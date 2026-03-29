"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A tensor parallel worker."""

import logging
import multiprocessing
import os
import pickle
import queue
import time
import threading
import warnings
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
import torch.distributed as dist

from sglang.global_config import global_config
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.policy_scheduler import PolicyScheduler, PrefillAdder
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.request_timeline import RUNNING_BATCH_WRITER, TIMELINE_WRITER
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode

DOC_POLICY_SNAPSHOT_DUMP_ENABLED = False
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.delta_fairness.time_estimation import pooled_prefill_time_estimation
from sglang.srt.delta_fairness.no_fairness_policy import NoFairnessPolicy
from sglang.srt.delta_fairness.static_fairness_policy import StaticFairnessPolicy
from sglang.srt.delta_fairness.delta_fairness_policy import DeltaFairnessPolicy
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang.srt.utils import (
    configure_logger,
    is_multimodal_model,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class AsyncCSVLogger:
    def __init__(self, csv_path: str, header: str):
        self.csv_path = csv_path
        self.header = header
        self._queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"async-csv-{os.path.basename(csv_path)}",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        header_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        pending: list[str] = []
        while not self._stop_event.is_set():
            try:
                line = self._queue.get(timeout=0.5)
                pending.append(line)
            except queue.Empty:
                pass

            if not pending:
                continue

            write_header = not header_written
            try:
                with open(self.csv_path, "a") as f:
                    if write_header:
                        f.write(self.header)
                        header_written = True
                    f.writelines(pending)
                pending.clear()
            except OSError:
                logger.exception("Failed to write telemetry CSV %s", self.csv_path)
                time.sleep(0.5)

        if pending:
            try:
                with open(self.csv_path, "a") as f:
                    if not header_written:
                        f.write(self.header)
                    f.writelines(pending)
            except OSError:
                logger.exception("Failed to flush telemetry CSV %s", self.csv_path)

    def log(self, line: str) -> None:
        self._queue.put(line)


class StepTimer:
    def __init__(self):
        self._last = time.perf_counter()
        self.parts: Dict[str, float] = {}

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        self.parts[name] = elapsed_ms
        self._last = now
        return elapsed_ms


def _normalize_single_delta_config(raw_deltas: Optional[Dict[str, int]]) -> Dict[str, int]:
    raw_deltas = raw_deltas or {}
    candidate_keys = (
        "delta",
        "decode",
        "first_decode",
        "prefill",
        "decode_running_batch",
        "first_decode_running_batch",
        "prefill_running_batch",
        "prefix_cache",
        "kv_cache",
    )
    effective_delta_us = 0
    for key in candidate_keys:
        value = raw_deltas.get(key)
        if value is None:
            continue
        effective_delta_us = max(effective_delta_us, int(value))

    return {
        "delta": effective_delta_us,
        "prefix_cache": effective_delta_us,
        "kv_cache": effective_delta_us,
        "prefill": effective_delta_us,
        "first_decode": effective_delta_us,
        "decode": effective_delta_us,
    }


crash_on_warning = os.getenv("SGLANG_IS_IN_CI", "false") == "true"
CLIP_MAX_NEW_TOKENS = int(os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS", "4096"))
PREFILL_TOKENS_PER_DECODE = 250


class ModelTpServer:
    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_override_args: dict,
    ):
        suppress_other_loggers()

        # Copy arguments
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=model_override_args,
        )

        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size - 1,
        )
        self.max_req_input_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )

        # Sync random seed
        server_args.random_seed = broadcast_recv_input(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(server_args.random_seed)

        # Print info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        self.max_kv_cache_tokens_per_user = None
        if server_args.static_reservation_n > 0:
            self.max_kv_cache_tokens_per_user = (
                self.max_total_num_tokens // server_args.static_reservation_n
            )

        self.fair_share_tokens_per_user = None
        self.delta_fairness_n = None
        self.delta_fairness_deltas_microseconds = None
        self.delta_fairness_quanta_us = 0
        self.delta_fairness_pooled_quanta_us = 0
        self.delta_fairness_exclusive_quanta_us = 0

        if server_args.delta_fairness_n > 0:
            self.delta_fairness_n = server_args.delta_fairness_n
            self.fair_share_tokens_per_user = (
                self.max_total_num_tokens // server_args.delta_fairness_n
            )
            delta_fairness_json_path = server_args.delta_fairness_config_file
            try:
                with open(delta_fairness_json_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except json.JSONDecodeError as err:
                logger.error(
                    "Could not read delta fairness config %s: %s",
                    delta_fairness_json_path,
                    err,
                )
                raise

            self.delta_fairness_deltas_microseconds = config.get(
                "delta_fairness_deltas_microseconds",
                {
                    "prefill_running_batch": 0,
                    "decode_running_batch": 150000,
                },
            )
            self.delta_fairness_deltas_microseconds = _normalize_single_delta_config(
                self.delta_fairness_deltas_microseconds,
            )
            logger.info(
                "Normalized delta fairness single-delta config: "
                "delta=%dus prefix_cache=%dus kv_cache=%dus prefill=%dus "
                "first_decode=%dus decode=%dus",
                self.delta_fairness_deltas_microseconds["delta"],
                self.delta_fairness_deltas_microseconds["prefix_cache"],
                self.delta_fairness_deltas_microseconds["kv_cache"],
                self.delta_fairness_deltas_microseconds["prefill"],
                self.delta_fairness_deltas_microseconds["first_decode"],
                self.delta_fairness_deltas_microseconds["decode"],
            )
            self.delta_fairness_quanta_us = int(
                config.get("delta_fairness_quanta_us", 0) or 0
            )
            self.delta_fairness_pooled_quanta_us = int(
                config.get(
                    "delta_fairness_pooled_quanta_us",
                    self.delta_fairness_quanta_us,
                )
                or 0
            )
            self.delta_fairness_exclusive_quanta_us = int(
                config.get("delta_fairness_exclusive_quanta_us", 0) or 0
            )

        # Init cache
        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
                static_max_per_user=self.max_kv_cache_tokens_per_user,
                fairinf_max_per_user=self.fair_share_tokens_per_user,
                fairinf_n=self.delta_fairness_n,
                fairinf_deltas_microseconds=self.delta_fairness_deltas_microseconds,
            )

        if self.delta_fairness_n:
            if server_args.delta_fairness_policy in (
                "earliest_deadline_first",
                "doc_policy",
            ):
                self.fairness_policy = DocPolicy(
                    delta_fairness_n=self.delta_fairness_n,
                    max_running_requests=self.max_running_requests,
                    delta_fairness_quanta_us=self.delta_fairness_quanta_us,
                    delta_fairness_pooled_quanta_us=self.delta_fairness_pooled_quanta_us,
                    delta_fairness_exclusive_quanta_us=self.delta_fairness_exclusive_quanta_us,
                    max_prefill_tokens=self.max_prefill_tokens,
                    isolated_kv_tokens_per_user=self.fair_share_tokens_per_user,
                    schedule_conservativeness=server_args.schedule_conservativeness,
                )
            else:
                self.fairness_policy = DeltaFairnessPolicy(
                    delta_fairness_n=self.delta_fairness_n,
                    max_running_requests=self.max_running_requests,
                )
        elif self.max_kv_cache_tokens_per_user:
            self.fairness_policy = StaticFairnessPolicy(
                static_reservation_n=server_args.static_reservation_n,
            )
        else:
            self.fairness_policy = NoFairnessPolicy()
        self.fairness_policy.set_tree_cache(self.tree_cache)
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.scheduler = PolicyScheduler(self.schedule_policy, self.tree_cache)
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool
        self.fairness_policy.token_to_kv_pool = self.token_to_kv_pool

        # Init running status
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduleBatch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        self.last_running_batch_snapshot_tic = 0.0
        self.last_model_forward_elapsed_ms = None
        self.last_decode_step_breakdown = None
        self._last_prepare_async_wait_ms = 0.0
        self._sglang_csv_logger = AsyncCSVLogger(
            "sglang_log.csv",
            "timestamp,type,time_elapsed,gpu_time_elapsed_ms,wall_time_elapsed_ms,"
            "model_forward_elapsed_ms,"
            "decode_sync_wait_ms,decode_after_sync_ms,"
            "decode_prepare_async_wait_ms,"
            "decode_check_mem_ms,decode_jump_forward_ms,decode_prepare_ms,"
            "decode_fairness_prepare_ms,"
            "decode_fairness_sync_live_user_tracking_ms,"
            "decode_fairness_logical_event_update_ms,"
            "decode_fairness_rebuild_from_real_state_ms,"
            "decode_fairness_build_deadline_candidates_ms,"
            "decode_build_input_ids_ms,decode_input_tensor_and_seq_lens_ms,"
            "decode_alloc_decode_output_slots_ms,decode_write_req_to_token_ms,"
            "decode_update_regex_vocab_mask_ms,"
            "decode_sample_postprocess_ms,decode_handle_finished_ms,"
            "running_reqs,num_tokens,token_usage,throughput,queue_reqs\n",
        )
        self._intermediate_gap_csv_logger = AsyncCSVLogger(
            "sglang_intermediate_gaps.csv",
            "timestamp,after_event_type,next_event_type,"
            "after_running_reqs,after_num_tokens,after_queue_reqs,"
            "current_running_reqs,current_num_tokens,current_queue_reqs,"
            "gap_total_ms,decision_overhead_ms,prepare_async_wait_ms,prepare_mutation_queue_backpressure_wait_ms,prepare_task_queue_backpressure_wait_ms,prepare_mutation_queue_drain_wait_ms,prepare_duplicate_state_wait_ms,force_prefill_check_ms,force_decode_check_ms,"
            "get_new_prefill_batch_ms,calc_priority_ms,prefill_adder_init_ms,"
            "remove_running_tokens_ms,doc_hot_path_refresh_ms,fairness_start_of_pass_ms,inflight_ms,"
            "force_prefill_reservations_ms,waiting_queue_prefills_ms,build_batch_ms,"
            "controller_send_queue_ms,controller_recv_requests_ms,request_handling_ms,"
            "doc_sync_live_user_tracking_ms,doc_rebuild_from_real_state_ms,"
            "doc_build_deadline_candidates_ms,doc_sort_waiting_prefills_ms,"
            "doc_safe_prefix_scan_ms,doc_build_pass_state_ms,doc_start_of_pass_total_ms\n",
        )
        self._doc_policy_snapshot_logger = (
            AsyncCSVLogger("doc_policy_pass_snapshots.jsonl", "")
            if DOC_POLICY_SNAPSHOT_DUMP_ENABLED
            else None
        )
        self._scheduler_pass_csv_logger = AsyncCSVLogger(
            "scheduler_passes.csv",
            "timestamp,running_reqs,waiting_reqs,current_num_tokens,"
            "force_prefill,force_decode,max_prefill_size,new_batch_size,chosen_event,"
            "get_new_prefill_reason,"
            "decision_force_prefill_check_ms,decision_force_decode_check_ms,decision_get_new_prefill_batch_ms,"
            "prefill_calc_priority_ms,prefill_adder_init_ms,prefill_remove_running_tokens_ms,"
            "prefill_doc_hot_path_refresh_ms,prefill_fairness_start_of_pass_ms,prefill_inflight_ms,prefill_force_prefill_reservations_ms,"
            "prefill_waiting_queue_prefills_ms,prefill_build_batch_ms,"
            "doc_forced_prefill_count,doc_safe_waiting_count,doc_deadline_queue_len,"
            "doc_has_decode_deadline,doc_max_safe_prefill_tokens,doc_waiting_deadline_count,"
            "doc_earliest_decode_start_deadline,doc_safe_prefix_now,doc_decode_deadline_slack_ms,"
            "doc_earliest_decode_rid,doc_earliest_decode_uid,doc_earliest_decode_completion_number,"
            "doc_earliest_decode_deadline,doc_earliest_decode_event_end_timestamp,"
            "doc_pass_state_source,doc_current_pass_id,doc_last_consumed_prepare_snapshot_seq,"
            "doc_first_waiting_rid,doc_first_waiting_prompt_tokens,doc_first_candidate_prefill_ms,doc_first_candidate_residual_slack_ms\n",
        )
        self._doc_policy_snapshot_threshold_ms = float(
            os.environ.get("DOC_POLICY_SNAPSHOT_THRESHOLD_MS", "200")
        )
        self._doc_policy_snapshot_limit = int(
            os.environ.get("DOC_POLICY_SNAPSHOT_LIMIT", "20")
        )
        self._doc_policy_snapshot_count = 0
        self._last_event_snapshot = None
        self._last_controller_send_queue_ms = 0.0
        self._last_controller_recv_requests_ms = 0.0
        self._last_request_handling_ms = 0.0

        # Chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.current_inflight_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the FSM cache for constrained generation
        if not server_args.skip_tokenizer_init:
            self.regex_fsm_cache = FSMCache(
                server_args.tokenizer_path,
                {
                    "tokenizer_mode": server_args.tokenizer_mode,
                    "trust_remote_code": server_args.trust_remote_code,
                },
                skip_tokenizer_init=server_args.skip_tokenizer_init,
                json_schema_mode=False,
            )
            self.json_fsm_cache = FSMCache(
                server_args.tokenizer_path,
                {
                    "tokenizer_mode": server_args.tokenizer_mode,
                    "trust_remote_code": server_args.trust_remote_code,
                },
                skip_tokenizer_init=server_args.skip_tokenizer_init,
                json_schema_mode=True,
            )
        self.jump_forward_cache = JumpForwardCache()

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay

    def exposed_step(self, recv_reqs: List):
        try:
            # Recv requests
            request_handling_start = time.perf_counter()
            for recv_req in recv_reqs:
                if isinstance(
                    recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                ):
                    self.handle_generate_request(recv_req)
                elif isinstance(recv_req, FlushCacheReq):
                    self.flush_cache()
                elif isinstance(recv_req, AbortReq):
                    self.abort_request(recv_req)
                elif isinstance(recv_req, UpdateWeightReqInput):
                    success, message = self.update_weights(recv_req)
                    self.out_pyobjs.append(UpdateWeightReqOutput(success, message))
                else:
                    raise ValueError(f"Invalid request: {recv_req}")
            self._last_request_handling_ms = (
                time.perf_counter() - request_handling_start
            ) * 1000.0

            # Forward
            self.forward_step()
        except Exception:
            logger.error("Exception in ModelTpServer:\n" + get_exception_traceback())
            raise

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret

    @torch.inference_mode()
    def forward_step(self):
        decision_timer = StepTimer()
        prefill_telemetry: Dict[str, float] = {}
        force_decode = False
        max_prefill_size = None
        new_batch = None
        force_prefill = False

        force_prefill_func = lambda: self.fairness_policy.fairinf_force_prefill_any_waiting(
                self.waiting_queue,
                delta_fairness_deltas_microseconds=self.delta_fairness_deltas_microseconds,
                running_batch=self.running_batch,
            )
        
        force_decode_func = lambda: self.fairness_policy.fairinf_force_decode(
                    self.running_batch,
                    delta_fairness_deltas_microseconds=self.delta_fairness_deltas_microseconds,
                )
        
        doc_pass_state_ready = False
        if isinstance(self.fairness_policy, DocPolicy):
            self.fairness_policy.start_of_pass(
                self.running_batch,
                self.waiting_queue,
                new_token_ratio=self.new_token_ratio,
                max_running_requests=self.max_running_requests,
            )
            prefill_telemetry["doc_hot_path_refresh_ms"] = decision_timer.mark(
                "doc_hot_path_refresh_ms"
            )
            doc_pass_state_ready = True

        if self.fairness_policy.fairinf_prioritize_force_prefill():
            # Force prefill is checked first.
            if isinstance(self.fairness_policy, DocPolicy):
                force_prefill = bool(self.fairness_policy._forced_prefill_rids)
                max_prefill_size = self.fairness_policy._max_safe_prefill_tokens
                decision_timer.parts["force_prefill_check_ms"] = 0.0
            else:
                force_prefill = force_prefill_func()
                decision_timer.mark("force_prefill_check_ms")

            if not force_prefill:
                force_decode, max_prefill_size = force_decode_func()
                decision_timer.mark("force_decode_check_ms")

            new_batch = (
                None
                if force_decode
                else self.get_new_prefill_batch(
                    max_prefill_size,
                    telemetry=prefill_telemetry,
                    pass_state_ready=doc_pass_state_ready,
                )
            )
            decision_timer.mark("get_new_prefill_batch_ms")
        
        else:
            # Force decode is checked first
            force_decode, max_prefill_size = force_decode_func()
            decision_timer.mark("force_decode_check_ms")
            
            if not force_decode:
                force_prefill = force_prefill_func()
                decision_timer.mark("force_prefill_check_ms")
                new_batch = (
                    None
                    if force_prefill
                    else self.get_new_prefill_batch(
                        max_prefill_size,
                        telemetry=prefill_telemetry,
                        pass_state_ready=doc_pass_state_ready,
                    )
                )
                decision_timer.mark("get_new_prefill_batch_ms")

        if "force_prefill_check_ms" not in decision_timer.parts:
            decision_timer.parts["force_prefill_check_ms"] = 0.0
        if "force_decode_check_ms" not in decision_timer.parts:
            decision_timer.parts["force_decode_check_ms"] = 0.0
        if "get_new_prefill_batch_ms" not in decision_timer.parts:
            decision_timer.parts["get_new_prefill_batch_ms"] = 0.0
        if "doc_hot_path_refresh_ms" not in prefill_telemetry:
            prefill_telemetry["doc_hot_path_refresh_ms"] = 0.0
        self._log_scheduler_pass(
            force_prefill=force_prefill,
            force_decode=force_decode,
            max_prefill_size=max_prefill_size,
            new_batch=new_batch,
            get_new_prefill_reason=str(prefill_telemetry.get("reason", "")),
            decision_parts=decision_timer.parts,
            prefill_parts=prefill_telemetry,
        )
        self._log_intermediate_gap(
            next_event_type=(
                "prefill" if new_batch is not None else "decode" if self.running_batch is not None else "idle"
            ),
            decision_parts=decision_timer.parts,
            prefill_parts=prefill_telemetry,
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if new_batch is not None:
            # Run a new prefill batch
            if self.running_batch is not None:
                add_wait = new_batch.total_size() / PREFILL_TOKENS_PER_DECODE
                for req in self.running_batch.reqs:
                    req.waiting_time_in_decodes += add_wait
            wall_start = time.perf_counter()
            start.record()
            self.forward_prefill_batch(new_batch)
            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
            self.fairness_policy.finished_prefill(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
            self.print_stats(
                decode=False,
                elapsed_time_ms=elapsed_time_ms,
                wall_time_ms=wall_time_ms,
            )

        else:
            # Run a decode batch
            if self.running_batch is not None:
                self.running_batch.max_running_requests = self.max_running_requests
                self.running_batch.delta_fairness_n = self.delta_fairness_n
                async_prepare_launched = self.fairness_policy.launch_async_decode_epoch_prepare(
                    running_batch=self.running_batch,
                    waiting_queue=list(self.waiting_queue),
                    selected_rids=None if not force_decode else self.fairness_policy.fairinf_overdue_decode_subset_rids(
                        self.running_batch
                    ),
                    decode_steps=global_config.num_continue_decode_steps,
                )
                # Run a few decode batches continuously for reducing overhead
                decoded_reqs_by_rid = {}
                decoded_steps_by_rid = {}
                decode_steps_run = 0
                for decode_step_idx in range(global_config.num_continue_decode_steps):
                    selected_rids = (
                        self.fairness_policy.fairinf_overdue_decode_subset_rids(
                            self.running_batch
                        )
                        if force_decode
                        else None
                    )
                    for req in self.running_batch.reqs:
                        if selected_rids is None or req.rid in selected_rids:
                            req.waiting_time_in_decodes = 0
                        else:
                            req.waiting_time_in_decodes += 1
                    for req in self.waiting_queue:
                        req.waiting_time_in_decodes += 1
                    generated_count = (
                        len(self.running_batch.reqs)
                        if selected_rids is None
                        else sum(1 for req in self.running_batch.reqs if req.rid in selected_rids)
                    )
                    self.num_generated_tokens += generated_count
                    wall_start = time.perf_counter()
                    start.record()
                    self.forward_decode_batch(
                        self.running_batch,
                        selected_rids=selected_rids,
                        prepare_pass_state=False,
                        decode_steps=0,
                    )
                    end.record()
                    if not end.query():
                        end.synchronize()
                    elapsed_time_ms = start.elapsed_time(end)
                    wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
                    for req in self.running_batch.reqs:
                        if selected_rids is None or req.rid in selected_rids:
                            decoded_reqs_by_rid[req.rid] = req
                            decoded_steps_by_rid[req.rid] = (
                                decoded_steps_by_rid.get(req.rid, 0) + 1
                            )
                    decode_steps_run += 1

                    # Print stats
                    self.print_stats(
                        decode=True,
                        elapsed_time_ms=elapsed_time_ms,
                        wall_time_ms=wall_time_ms,
                    )

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break
                if (
                    decoded_steps_by_rid
                    and self.running_batch is not None
                    and not async_prepare_launched
                    and hasattr(self.fairness_policy, "prepare_during_gpu_execution")
                ):
                    self.fairness_policy.prepare_during_gpu_execution(
                        event_type="decode",
                        running_batch=self.running_batch,
                        waiting_queue=list(self.waiting_queue),
                        scheduled_batch=None,
                        selected_rids=set(decoded_steps_by_rid),
                        prepare_pass_state=True,
                        decode_steps=0,
                        decode_steps_by_rid=dict(decoded_steps_by_rid),
                        output_ids_already_applied=True,
                        new_token_ratio=self.new_token_ratio,
                    )
                if decoded_reqs_by_rid:
                    self.fairness_policy.finished_decode(
                        SimpleNamespace(reqs=list(decoded_reqs_by_rid.values())),
                        decode_rounds=decode_steps_run,
                    )
                self._last_prepare_async_wait_ms = (
                    self.fairness_policy.wait_for_async_prepare()
                    if async_prepare_launched
                    else 0.0
                )

            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio

    def print_stats(self, decode=True, elapsed_time_ms=0, wall_time_ms=None):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()

        current_time = time.time()
        token_usage = num_used / self.max_total_num_tokens
        running_reqs = len(self.running_batch.reqs) if hasattr(self.running_batch, "reqs") else 0
        queue_reqs = len(self.waiting_queue)
        if (
            self.tp_rank == 0
            and self.running_batch is not None
            and hasattr(self.running_batch, "reqs")
            and current_time - self.last_running_batch_snapshot_tic >= 1.0
        ):
            RUNNING_BATCH_WRITER.write_snapshot(
                batch_type="decode" if decode else "prefill",
                running_reqs=self.running_batch.reqs,
            )
            self.last_running_batch_snapshot_tic = current_time
        
        # Log to console
        logger.info(
            f"{'Decode' if decode else 'Prefill'} batch. "
            f"#running-req: {running_reqs}, "
            f"#token: {num_used}, "
            f"token usage: {token_usage:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
        )
        
        # Log to CSV
        if self.tp_rank == 0:  # Only log from rank 0
            wall_time_ms = elapsed_time_ms if wall_time_ms is None else wall_time_ms
            model_forward_ms = (
                "" if self.last_model_forward_elapsed_ms is None else self.last_model_forward_elapsed_ms
            )
            breakdown = self.last_decode_step_breakdown or {}
            prepare_breakdown = breakdown.get("prepare_breakdown", {}) if breakdown else {}
            fairness_prepare_breakdown = (
                breakdown.get("fairness_prepare_breakdown", {}) if breakdown else {}
            )
            self._sglang_csv_logger.log(
                f"{current_time},{'Decode' if decode else 'Prefill'},{wall_time_ms},"
                f"{elapsed_time_ms},{wall_time_ms},{model_forward_ms},"
                f"{breakdown.get('sync_wait_ms', '')},{breakdown.get('after_sync_ms', '')},"
                f"{self._last_prepare_async_wait_ms if decode else ''},"
                f"{breakdown.get('check_mem_ms', '')},{breakdown.get('jump_forward_ms', '')},"
                f"{breakdown.get('prepare_ms', '')},"
                f"{breakdown.get('fairness_prepare_ms', '')},"
                f"{fairness_prepare_breakdown.get('sync_live_user_tracking_ms', '')},"
                f"{fairness_prepare_breakdown.get('logical_event_update_ms', '')},"
                f"{fairness_prepare_breakdown.get('rebuild_from_real_state_ms', '')},"
                f"{fairness_prepare_breakdown.get('build_deadline_candidates_ms', '')},"
                f"{prepare_breakdown.get('build_input_ids_ms', '')},"
                f"{prepare_breakdown.get('input_tensor_and_seq_lens_ms', '')},"
                f"{prepare_breakdown.get('alloc_decode_output_slots_ms', '')},"
                f"{prepare_breakdown.get('write_req_to_token_ms', '')},"
                f"{prepare_breakdown.get('update_regex_vocab_mask_ms', '')},"
                f"{breakdown.get('sample_postprocess_ms', '')},"
                f"{breakdown.get('handle_finished_ms', '')},{running_reqs},{num_used},"
                f"{token_usage:.4f},{throughput:.4f},{queue_reqs}\n"
            )
            self._last_event_snapshot = {
                "event_type": "decode" if decode else "prefill",
                "running_reqs": running_reqs,
                "num_tokens": num_used,
                "queue_reqs": queue_reqs,
                "perf_counter": time.perf_counter(),
                "timestamp": current_time,
            }

    def _current_num_used_tokens(self) -> int:
        return self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )

    def _log_intermediate_gap(
        self,
        *,
        next_event_type: str,
        decision_parts: Dict[str, float],
        prefill_parts: Dict[str, float],
    ) -> None:
        if (
            self.tp_rank != 0
            or self._last_event_snapshot is None
            or next_event_type == "idle"
        ):
            return

        snapshot = self._last_event_snapshot
        now_perf = time.perf_counter()
        gap_total_ms = (now_perf - snapshot["perf_counter"]) * 1000.0
        current_running_reqs = (
            len(self.running_batch.reqs) if self.running_batch is not None else 0
        )
        current_queue_reqs = len(self.waiting_queue)
        current_num_tokens = self._current_num_used_tokens()
        prepare_thread_wait_metrics = {}
        if hasattr(self.fairness_policy, "consume_prepare_thread_wait_metrics"):
            prepare_thread_wait_metrics = (
                self.fairness_policy.consume_prepare_thread_wait_metrics()
            )
        self._intermediate_gap_csv_logger.log(
            f"{time.time()},{snapshot['event_type']},{next_event_type},"
            f"{snapshot['running_reqs']},{snapshot['num_tokens']},{snapshot['queue_reqs']},"
            f"{current_running_reqs},{current_num_tokens},{current_queue_reqs},"
            f"{gap_total_ms},"
            f"{decision_parts.get('force_prefill_check_ms', 0.0) + decision_parts.get('force_decode_check_ms', 0.0) + decision_parts.get('get_new_prefill_batch_ms', 0.0)},"
            f"{self._last_prepare_async_wait_ms},"
            f"{prepare_thread_wait_metrics.get('prepare_mutation_queue_backpressure_wait_ms', 0.0)},"
            f"{prepare_thread_wait_metrics.get('prepare_task_queue_backpressure_wait_ms', 0.0)},"
            f"{prepare_thread_wait_metrics.get('prepare_mutation_queue_drain_wait_ms', 0.0)},"
            f"{prepare_thread_wait_metrics.get('prepare_duplicate_state_wait_ms', 0.0)},"
            f"{decision_parts.get('force_prefill_check_ms', 0.0)},"
            f"{decision_parts.get('force_decode_check_ms', 0.0)},"
            f"{decision_parts.get('get_new_prefill_batch_ms', 0.0)},"
            f"{prefill_parts.get('calc_priority_ms', 0.0)},"
            f"{prefill_parts.get('prefill_adder_init_ms', 0.0)},"
            f"{prefill_parts.get('remove_running_tokens_ms', 0.0)},"
            f"{prefill_parts.get('doc_hot_path_refresh_ms', 0.0)},"
            f"{prefill_parts.get('fairness_start_of_pass_ms', 0.0)},"
            f"{prefill_parts.get('inflight_ms', 0.0)},"
            f"{prefill_parts.get('force_prefill_reservations_ms', 0.0)},"
            f"{prefill_parts.get('waiting_queue_prefills_ms', 0.0)},"
            f"{prefill_parts.get('build_batch_ms', 0.0)},"
            f"{self._last_controller_send_queue_ms},"
            f"{self._last_controller_recv_requests_ms},"
            f"{self._last_request_handling_ms},"
            f"{prefill_parts.get('doc_sync_live_user_tracking_ms', 0.0)},"
            f"{prefill_parts.get('doc_rebuild_from_real_state_ms', 0.0)},"
            f"{prefill_parts.get('doc_build_deadline_candidates_ms', 0.0)},"
            f"{prefill_parts.get('doc_sort_waiting_prefills_ms', 0.0)},"
            f"{prefill_parts.get('doc_safe_prefix_scan_ms', 0.0)},"
            f"{prefill_parts.get('doc_build_pass_state_ms', 0.0)},"
            f"{prefill_parts.get('doc_start_of_pass_total_ms', 0.0)}\n"
        )
        self._last_prepare_async_wait_ms = 0.0
        self._last_controller_send_queue_ms = 0.0
        self._last_controller_recv_requests_ms = 0.0
        self._last_request_handling_ms = 0.0

    def _log_scheduler_pass(
        self,
        *,
        force_prefill: bool,
        force_decode: bool,
        max_prefill_size: Optional[int],
        new_batch: Optional[ScheduleBatch],
        get_new_prefill_reason: str,
        decision_parts: Dict[str, float],
        prefill_parts: Dict[str, float],
    ) -> None:
        if self.tp_rank != 0:
            return

        chosen_event = (
            "prefill"
            if new_batch is not None
            else "decode"
            if self.running_batch is not None
            else "idle"
        )
        if chosen_event == "idle" and self.running_batch is None and not self.waiting_queue:
            return
        doc_forced_prefill_count = 0
        doc_safe_waiting_count = 0
        doc_deadline_queue_len = 0
        doc_has_decode_deadline = 0
        doc_max_safe_prefill_tokens = ""
        doc_waiting_deadline_count = 0
        doc_earliest_decode_start_deadline = ""
        doc_safe_prefix_now = ""
        doc_decode_deadline_slack_ms = ""
        doc_earliest_decode_rid = ""
        doc_earliest_decode_uid = ""
        doc_earliest_decode_completion_number = ""
        doc_earliest_decode_deadline = ""
        doc_earliest_decode_event_end_timestamp = ""
        doc_pass_state_source = ""
        doc_current_pass_id = ""
        doc_last_consumed_prepare_snapshot_seq = ""
        doc_first_waiting_rid = ""
        doc_first_waiting_prompt_tokens = ""
        doc_first_candidate_prefill_ms = ""
        doc_first_candidate_residual_slack_ms = ""
        if isinstance(self.fairness_policy, DocPolicy):
            doc_forced_prefill_count = len(
                getattr(self.fairness_policy, "_forced_prefill_rids", ())
            )
            doc_safe_waiting_count = len(
                getattr(self.fairness_policy, "_safe_waiting_queue", ())
            )
            doc_deadline_queue_len = len(
                getattr(self.fairness_policy, "_deadline_queue", ())
            )
            doc_has_decode_deadline = int(
                bool(getattr(self.fairness_policy, "_has_decode_deadline", False))
            )
            max_safe = getattr(
                self.fairness_policy, "_max_safe_prefill_tokens", None
            )
            doc_max_safe_prefill_tokens = (
                "" if max_safe is None else int(max_safe)
            )
            doc_waiting_deadline_count = len(
                getattr(
                    self.fairness_policy,
                    "_waiting_prefill_start_deadline_by_rid",
                    {},
                )
            )
            earliest = getattr(
                self.fairness_policy, "_earliest_decode_start_deadline", None
            )
            safe_now = getattr(self.fairness_policy, "_safe_prefix_now", None)
            if earliest is not None:
                doc_earliest_decode_start_deadline = earliest
            if safe_now is not None:
                doc_safe_prefix_now = safe_now
            if earliest is not None and safe_now is not None:
                doc_decode_deadline_slack_ms = (earliest - safe_now) * 1000.0
            doc_earliest_decode_rid = getattr(
                self.fairness_policy, "_debug_earliest_decode_rid", ""
            ) or ""
            doc_earliest_decode_uid = getattr(
                self.fairness_policy, "_debug_earliest_decode_uid", ""
            ) or ""
            earliest_decode_candidate = min(
                (
                    candidate
                    for candidate in getattr(self.fairness_policy, "_deadline_queue", ())
                    if getattr(candidate, "event_type", None) == "decode"
                ),
                key=lambda candidate: candidate.start_deadline,
                default=None,
            )
            if earliest_decode_candidate is not None:
                if not doc_earliest_decode_rid:
                    doc_earliest_decode_rid = getattr(
                        getattr(earliest_decode_candidate, "req", None),
                        "rid",
                        "",
                    )
                if not doc_earliest_decode_uid:
                    doc_earliest_decode_uid = getattr(
                        getattr(earliest_decode_candidate, "req", None),
                        "uid",
                        "",
                    )
                doc_earliest_decode_completion_number = getattr(
                    getattr(earliest_decode_candidate, "event", None),
                    "completion_number",
                    "",
                )
                doc_earliest_decode_deadline = getattr(
                    earliest_decode_candidate, "deadline", ""
                )
                doc_earliest_decode_event_end_timestamp = getattr(
                    getattr(earliest_decode_candidate, "event", None),
                    "end_timestamp",
                    "",
                )
            doc_first_waiting_rid = getattr(
                self.fairness_policy, "_debug_first_waiting_rid", ""
            ) or ""
            first_prompt = getattr(
                self.fairness_policy, "_debug_first_waiting_prompt_tokens", None
            )
            doc_first_waiting_prompt_tokens = (
                "" if first_prompt is None else int(first_prompt)
            )
            first_prefill_ms = getattr(
                self.fairness_policy, "_debug_first_candidate_prefill_ms", None
            )
            doc_first_candidate_prefill_ms = (
                "" if first_prefill_ms is None else float(first_prefill_ms)
            )
            first_residual = getattr(
                self.fairness_policy, "_debug_first_candidate_residual_slack_ms", None
            )
            doc_first_candidate_residual_slack_ms = (
                "" if first_residual is None else float(first_residual)
            )
            doc_pass_state_source = getattr(
                self.fairness_policy, "_last_pass_state_source", ""
            ) or ""
            doc_current_pass_id = getattr(
                self.fairness_policy, "_current_pass_id", ""
            )
            doc_last_consumed_prepare_snapshot_seq = getattr(
                self.fairness_policy, "_last_consumed_prepare_snapshot_seq", ""
            )

        self._scheduler_pass_csv_logger.log(
            f"{time.time()},"
            f"{len(self.running_batch.reqs) if self.running_batch is not None else 0},"
            f"{len(self.waiting_queue)},"
            f"{self._current_num_used_tokens()},"
            f"{int(force_prefill)},{int(force_decode)},"
            f"{'' if max_prefill_size is None else max_prefill_size},"
            f"{0 if new_batch is None else len(new_batch.reqs)},{chosen_event},"
            f"{get_new_prefill_reason},"
            f"{decision_parts.get('force_prefill_check_ms', 0.0)},"
            f"{decision_parts.get('force_decode_check_ms', 0.0)},"
            f"{decision_parts.get('get_new_prefill_batch_ms', 0.0)},"
            f"{prefill_parts.get('calc_priority_ms', 0.0)},"
            f"{prefill_parts.get('prefill_adder_init_ms', 0.0)},"
            f"{prefill_parts.get('remove_running_tokens_ms', 0.0)},"
            f"{prefill_parts.get('doc_hot_path_refresh_ms', 0.0)},"
            f"{prefill_parts.get('fairness_start_of_pass_ms', 0.0)},"
            f"{prefill_parts.get('inflight_ms', 0.0)},"
            f"{prefill_parts.get('force_prefill_reservations_ms', 0.0)},"
            f"{prefill_parts.get('waiting_queue_prefills_ms', 0.0)},"
            f"{prefill_parts.get('build_batch_ms', 0.0)},"
            f"{doc_forced_prefill_count},{doc_safe_waiting_count},{doc_deadline_queue_len},"
            f"{doc_has_decode_deadline},{doc_max_safe_prefill_tokens},{doc_waiting_deadline_count},"
            f"{doc_earliest_decode_start_deadline},{doc_safe_prefix_now},{doc_decode_deadline_slack_ms},"
            f"{doc_earliest_decode_rid},{doc_earliest_decode_uid},{doc_earliest_decode_completion_number},"
            f"{doc_earliest_decode_deadline},{doc_earliest_decode_event_end_timestamp},"
            f"{doc_pass_state_source},{doc_current_pass_id},{doc_last_consumed_prepare_snapshot_seq},"
            f"{doc_first_waiting_rid},{doc_first_waiting_prompt_tokens},{doc_first_candidate_prefill_ms},{doc_first_candidate_residual_slack_ms}\n"
        )

    def _serialize_doc_policy_req(self, req: Req) -> Dict[str, Any]:
        return {
            "uid": req.uid,
            "rid": req.rid,
            "prompt_tokens": len(req.origin_input_ids),
            "output_tokens": len(req.output_ids),
            "fill_tokens": len(req.fill_ids) if req.fill_ids is not None else None,
            "max_new_tokens": (
                None
                if getattr(req, "sampling_params", None) is None
                else getattr(req.sampling_params, "max_new_tokens", None)
            ),
            "waiting_time_in_decodes": getattr(req, "waiting_time_in_decodes", None),
        }

    def _serialize_doc_policy_real_event(self, event) -> Optional[Dict[str, Any]]:
        if event is None:
            return None
        return {
            "type": event.__class__.__name__,
            "req_id": getattr(event, "req_id", None),
            "end_timestamp": getattr(event, "end_timestamp", None),
            "completion_number": getattr(event, "completion_number", None),
        }

    def _capture_doc_policy_pass_snapshot_state(self) -> Optional[Dict[str, Any]]:
        if self.tp_rank != 0:
            return None
        if not isinstance(self.fairness_policy, DocPolicy):
            return None

        simulator = self.fairness_policy.simulator
        return {
            "running_batch_size": 0
            if self.running_batch is None
            else len(self.running_batch.reqs),
            "waiting_queue_size": len(self.waiting_queue),
            "running_reqs": []
            if self.running_batch is None
            else [self._serialize_doc_policy_req(req) for req in self.running_batch.reqs],
            "waiting_reqs": [self._serialize_doc_policy_req(req) for req in self.waiting_queue],
            "most_recent_event_real": {
                rid: self._serialize_doc_policy_real_event(event)
                for rid, event in simulator.most_recent_event_real.items()
            },
            "tracked_requests": {
                rid: {
                    "uid": tracked.req.uid,
                    "arrival_timestamp": tracked.arrival_timestamp,
                    "most_recent_real_event": self._serialize_doc_policy_real_event(
                        simulator.most_recent_event_real.get(rid)
                    ),
                }
                for rid, tracked in simulator.requests.items()
            },
        }

    def _maybe_dump_doc_policy_pass_snapshot(
        self,
        fairness_start_of_pass_ms: float,
        pre_pass_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        if not DOC_POLICY_SNAPSHOT_DUMP_ENABLED:
            return
        if self.tp_rank != 0:
            return
        if not isinstance(self.fairness_policy, DocPolicy):
            return
        if fairness_start_of_pass_ms < self._doc_policy_snapshot_threshold_ms:
            return
        if self._doc_policy_snapshot_count >= self._doc_policy_snapshot_limit:
            return

        snapshot = {
            "timestamp": time.time(),
            "fairness_start_of_pass_ms": fairness_start_of_pass_ms,
            "pre": pre_pass_snapshot,
            "post": self._capture_doc_policy_pass_snapshot_state(),
        }
        assert self._doc_policy_snapshot_logger is not None
        self._doc_policy_snapshot_logger.log(json.dumps(snapshot) + "\n")
        self._doc_policy_snapshot_count += 1


    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            warnings.warn(
                "Warning: "
                f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
                "KV cache pool leak detected!"
            )
            exit(1) if crash_on_warning else None

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            warnings.warn(
                "Warning: "
                f"available req slots={len(self.req_to_token_pool.free_slots)}, "
                f"total slots={self.req_to_token_pool.size}\n"
                "Memory pool leak detected!"
            )
            exit(1) if crash_on_warning else None

    def handle_generate_request(
        self,
        recv_req: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        req = Req(recv_req.uid, recv_req.rid, recv_req.input_text, recv_req.input_ids)
        req.tokenizer = self.tokenizer
        req.sampling_params = recv_req.sampling_params
        if self.model_runner.is_generation:
            req.pixel_values = recv_req.pixel_values
            if req.pixel_values is not None:
                # Use image hash as fake token_ids, which is then used
                # for prefix matching
                image_hash = hash(tuple(recv_req.image_hashes))
                req.pad_value = [
                    (image_hash) % self.model_config.vocab_size,
                    (image_hash >> 16) % self.model_config.vocab_size,
                    (image_hash >> 32) % self.model_config.vocab_size,
                    (image_hash >> 64) % self.model_config.vocab_size,
                ]
                req.image_sizes = recv_req.image_sizes
                (
                    req.origin_input_ids,
                    req.image_offsets,
                ) = self.model_runner.model.pad_input_ids(
                    req.origin_input_ids_unpadded,
                    req.pad_value,
                    req.pixel_values,
                    req.image_sizes,
                )
            req.return_logprob = recv_req.return_logprob
            req.logprob_start_len = recv_req.logprob_start_len
            req.top_logprobs_num = recv_req.top_logprobs_num
            req.stream = recv_req.stream

            # Init regex fsm fron json
            if req.sampling_params.json_schema is not None:
                req.regex_fsm, computed_regex_string = self.json_fsm_cache.query(
                    req.sampling_params.json_schema
                )
                if not self.disable_regex_jump_forward:
                    req.jump_forward_map = self.jump_forward_cache.query(
                        computed_regex_string
                    )

            # Init regex fsm
            elif req.sampling_params.regex is not None:
                req.regex_fsm = self.regex_fsm_cache.query(req.sampling_params.regex)
                if not self.disable_regex_jump_forward:
                    req.jump_forward_map = self.jump_forward_cache.query(
                        req.sampling_params.regex
                    )

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warn(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        if self.model_runner.is_generation:
            req.sampling_params.max_new_tokens = min(
                (
                    req.sampling_params.max_new_tokens
                    if req.sampling_params.max_new_tokens is not None
                    else 1 << 30
                ),
                self.max_req_input_len - 1 - len(req.origin_input_ids),
            )

        self.fairness_policy.process_new_request(req)
        self.waiting_queue.append(req)

    def get_new_prefill_batch(
        self,
        max_prefill_token_size: Optional[int] = None,
        *,
        telemetry: Optional[Dict[str, float]] = None,
        pass_state_ready: bool = False,
    ) -> Optional[ScheduleBatch]:
        step_timer = StepTimer()
        telemetry = telemetry if telemetry is not None else {}
        telemetry["reason"] = ""
        pre_pass_snapshot = None
        running_bs = (
            len(self.running_batch.reqs) if self.running_batch is not None else 0
        )
        available_req_slots = len(self.req_to_token_pool.free_slots)
        allow_force_prefill_retraction = (
            isinstance(self.fairness_policy, DocPolicy)
            and bool(getattr(self.fairness_policy, "_forced_prefill_rids", None))
        )
        if not self.waiting_queue and self.current_inflight_req is None:
            telemetry["reason"] = "no_waiting_or_inflight"
            return None
        if running_bs >= self.max_running_requests and not allow_force_prefill_retraction:
            telemetry["reason"] = "running_batch_full"
            return None
        if available_req_slots <= 0 and not allow_force_prefill_retraction:
            telemetry["reason"] = "no_req_slots"
            return None
        if (
            self.running_batch is not None
            and max_prefill_token_size is not None
            and max_prefill_token_size <= 0
        ):
            telemetry["reason"] = "prefill_capped_to_zero_by_force_decode"
            return None

        # DocPolicy determines waiting-order itself and we only use it with FCFS,
        # so global scheduler priority work is unnecessary on this path.
        if isinstance(self.fairness_policy, DocPolicy):
            prefix_computed = False
            telemetry["calc_priority_ms"] = 0.0
        else:
            prefix_computed = self.scheduler.calc_priority(self.waiting_queue)
            telemetry["calc_priority_ms"] = step_timer.mark("calc_priority_ms")

        num_mixed_running = running_bs if self.is_mixed_chunk else 0
        max_input_size = (
            min(self.max_prefill_tokens, max_prefill_token_size)
            if max_prefill_token_size is not None
            else self.max_prefill_tokens
        )

        adder = PrefillAdder(
            self.tree_cache,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
            max_input_size,
            self.chunked_prefill_size,
            num_mixed_running,
            fairness_policy=self.fairness_policy,
        )
        telemetry["prefill_adder_init_ms"] = step_timer.mark("prefill_adder_init_ms")

        if self.running_batch is not None:
            adder.remove_running_tokens(self.running_batch, self.new_token_ratio)
        telemetry["remove_running_tokens_ms"] = step_timer.mark(
            "remove_running_tokens_ms"
        )

        if isinstance(self.fairness_policy, DocPolicy):
            self.fairness_policy._prefill_no_retraction_token_cap = int(
                max(0, adder.rem_total_tokens)
            )
        if not pass_state_ready:
            self.fairness_policy.start_of_pass(
                self.running_batch,
                self.waiting_queue,
                new_token_ratio=self.new_token_ratio,
                max_running_requests=self.max_running_requests,
            )
            telemetry["fairness_start_of_pass_ms"] = step_timer.mark(
                "fairness_start_of_pass_ms"
            )
        else:
            telemetry["fairness_start_of_pass_ms"] = 0.0
        if isinstance(self.fairness_policy, DocPolicy):
            self.fairness_policy._prefill_no_retraction_token_cap = None
        if isinstance(self.fairness_policy, DocPolicy):
            telemetry["doc_sync_live_user_tracking_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "sync_live_user_tracking_ms", 0.0
                )
            )
            telemetry["doc_rebuild_from_real_state_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "rebuild_from_real_state_ms", 0.0
                )
            )
            telemetry["doc_build_deadline_candidates_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "build_deadline_candidates_ms", 0.0
                )
            )
            telemetry["doc_sort_waiting_prefills_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "sort_waiting_prefills_ms", 0.0
                )
            )
            telemetry["doc_safe_prefix_scan_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "safe_prefix_scan_ms", 0.0
                )
            )
            telemetry["doc_build_pass_state_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "build_pass_state_ms", 0.0
                )
            )
            telemetry["doc_start_of_pass_total_ms"] = (
                self.fairness_policy._last_pass_breakdown_ms.get(
                    "doc_policy_start_of_pass_total_ms", 0.0
                )
            )
        self._maybe_dump_doc_policy_pass_snapshot(
            telemetry["fairness_start_of_pass_ms"], pre_pass_snapshot
        )

        has_inflight = self.current_inflight_req is not None
        token_counters_by_user: Dict[str, List[int]] = {}
        if self.current_inflight_req is not None:
            inflight_result = self.current_inflight_req.init_next_round_input(
                None if prefix_computed else self.tree_cache,
                fairness_policy=self.fairness_policy,
            )
            if inflight_result != "rejected":
                token_counters_by_user.setdefault(
                    self.current_inflight_req.uid, []
                ).append(self.current_inflight_req.extend_input_len)
                self.current_inflight_req = adder.add_inflight_req(
                    self.current_inflight_req
                )
            else:
                self.current_inflight_req = None
                has_inflight = False
        telemetry["inflight_ms"] = step_timer.mark("inflight_ms")

        extra_space, evicted_reqs = self.fairness_policy.force_prefill_reservations(
            self.waiting_queue,
            token_counters_by_user=token_counters_by_user,
            adder=adder,
            token_to_kv_pool=self.token_to_kv_pool,
            running_batch=self.running_batch,
            delta_fairness_deltas_microseconds=self.delta_fairness_deltas_microseconds,
            max_input_size=max_input_size,
            prefix_computed=prefix_computed,
            max_running_requests=self.max_running_requests,
        )
        telemetry["force_prefill_reservations_ms"] = step_timer.mark(
            "force_prefill_reservations_ms"
        )
        running_bs = len(self.running_batch.reqs) if self.running_batch is not None else 0
        available_req_slots = len(self.req_to_token_pool.free_slots)
        if running_bs >= self.max_running_requests or available_req_slots <= 0:
            telemetry["reason"] = (
                "running_batch_full"
                if running_bs >= self.max_running_requests
                else "no_req_slots"
            )
            return None
        if extra_space > 0 and evicted_reqs:
            logger.info(
                "Fairness reserved %s tokens (available=%s, evictable=%s)",
                extra_space,
                self.token_to_kv_pool.available_size(),
                self.tree_cache.evictable_size(),
            )

        waiting_queue_for_prefills = self.waiting_queue
        if evicted_reqs and isinstance(self.fairness_policy, DocPolicy):
            waiting_queue_for_prefills = []

        self.fairness_policy.process_waiting_queue_prefills(
            waiting_queue_for_prefills,
            adder=adder,
            token_counters_by_user=token_counters_by_user,
            prefix_computed=prefix_computed,
            running_batch=self.running_batch,
            running_batch_size=running_bs,
            max_running_requests=self.max_running_requests,
            available_req_slots=available_req_slots,
            max_input_size=max_input_size,
        )
        telemetry["waiting_queue_prefills_ms"] = step_timer.mark(
            "waiting_queue_prefills_ms"
        )

        can_run_list = adder.can_run_list

        if adder.new_inflight_req is not None:
            self.current_inflight_req = adder.new_inflight_req

        if len(can_run_list) == 0:
            telemetry["reason"] = "adder_can_run_list_empty"
            return None

        # Print stats
        if self.tp_rank == 0:
            if isinstance(self.tree_cache, RadixCache):
                self.tree_cache_metrics["total"] += (
                    adder.log_input_tokens + adder.log_hit_tokens
                ) / 10**9
                self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
                tree_cache_hit_rate = (
                    self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
                )
            else:
                tree_cache_hit_rate = 0.0

            if num_mixed_running > 0:
                logger.info(
                    f"Prefill batch"
                    f"(mixed #running-req: {num_mixed_running}). "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
                )
            else:
                fairness_msg = (
                    f"Max tokens imposed by fair decode {max_prefill_token_size}. "
                    if max_prefill_token_size is not None
                    else ""
                )
                logger.info(
                    f"Prefill batch. {fairness_msg}"
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"#running-req: {running_bs}, "
                    f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
                )

        # Return the new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            fairness_policy=self.fairness_policy,
        )
        new_batch.max_running_requests = self.max_running_requests
        new_batch.delta_fairness_n = self.delta_fairness_n
        if hasattr(self.fairness_policy, "note_scheduled_prefill_batch"):
            self.fairness_policy.note_scheduled_prefill_batch(new_batch)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_list]
        telemetry["build_batch_ms"] = step_timer.mark("build_batch_ms")
        telemetry["reason"] = "built_prefill_batch"
        return new_batch

    def forward_prefill_batch(self, batch: ScheduleBatch):
        prefill_wall_start = time.perf_counter()
        last_step_start = prefill_wall_start
        self.last_model_forward_elapsed_ms = None

        def _log_prefill_step(step_name: str) -> None:
            nonlocal last_step_start
            now = time.perf_counter()
            logger.info(
                "Prefill timing step=%s wall_ms=%.3f batch_size=%s extend_tokens=%s",
                step_name,
                (now - last_step_start) * 1000.0,
                batch.batch_size(),
                getattr(batch, "extend_num_tokens", "unset"),
            )
            last_step_start = now

        # Build batch tensors
        requesting_users = {req.uid for req in batch.reqs}
        try:
            removed_requests = batch.prepare_for_extend(
                self.model_config.vocab_size,
                running_batch=self.running_batch,
                requesting_users=list(requesting_users),
            )
        except RuntimeError as exc:
            if "Static prefill admission denied" not in str(exc):
                raise
            logger.info("Skipping prefill batch: %s", exc)
            if self.fairness_policy.uses_static_isolated_memory():
                self.waiting_queue = list(batch.reqs) + self.waiting_queue
            else:
                self.waiting_queue.extend(batch.reqs)
            batch.reqs = []
            return
        _log_prefill_step("prepare_for_extend")

        if removed_requests:
            logger.info(
                "Prefill displaced %s decodes; pushing back to waiting queue",
                len(removed_requests),
            )
            if self.fairness_policy.uses_static_isolated_memory():
                self.waiting_queue = list(removed_requests) + self.waiting_queue
            else:
                self.waiting_queue.extend(removed_requests)
        _log_prefill_step("requeue_removed_requests")

        for req in batch.reqs:
            TIMELINE_WRITER.mark_prefill_start(req.rid, req.uid)

        decoding_reqs = []
        if self.is_mixed_chunk and self.running_batch is not None:
            self.running_batch.prepare_for_decode()
            batch.mix_with_running(self.running_batch)
            decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        _log_prefill_step("mix_with_running")

        if self.model_runner.is_generation:
            # Forward and sample the next tokens
            if batch.extend_num_tokens != 0:
                model_forward_start = torch.cuda.Event(enable_timing=True)
                model_forward_end = torch.cuda.Event(enable_timing=True)
                model_forward_start.record()
                sample_output, logits_output = self.model_runner.forward(
                    batch, ForwardMode.EXTEND
                )
                self.fairness_policy.prepare_during_gpu_execution(
                    event_type="prefill",
                    running_batch=self.running_batch,
                    waiting_queue=list(self.waiting_queue),
                    scheduled_batch=batch,
                    selected_rids=None,
                    new_token_ratio=self.new_token_ratio,
                )
                model_forward_end.record()
                torch.cuda.synchronize()
                self.last_model_forward_elapsed_ms = model_forward_start.elapsed_time(
                    model_forward_end
                )
                _log_prefill_step("model_forward_extend")
                next_token_ids = batch.check_sample_results(sample_output)
                _log_prefill_step("check_sample_results")
                batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    next_token_ids
                )
                _log_prefill_step("cumulate_output_tokens")

                # Move logprobs to cpu
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs[
                            torch.arange(
                                len(next_token_ids), device=next_token_ids.device
                            ),
                            next_token_ids,
                        ].tolist()
                    )
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.tolist()
                    )
                    logits_output.normalized_prompt_logprobs = (
                        logits_output.normalized_prompt_logprobs.tolist()
                    )
                    _log_prefill_step("logprobs_to_cpu")

                next_token_ids = next_token_ids.tolist()
                _log_prefill_step("next_token_ids_to_list")
            else:
                if self.tokenizer is None:
                    next_token_ids = []
                    for req in batch.reqs:
                        next_token_ids.append(
                            next(iter(req.sampling_params.stop_token_ids))
                        )
                else:
                    next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)
                _log_prefill_step("zero_extend_token_fallback")

            # Check finish conditions
            pt = 0
            for i, req in enumerate(batch.reqs):
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    req.completion_tokens_wo_jump_forward += 1
                    req.output_ids.append(next_token_ids[i])
                    req.check_finished()

                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.get_next_state(
                        req.regex_fsm_state, next_token_ids[i]
                    )

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                elif req not in decoding_reqs:
                    # To reduce overhead, only cache prefill reqs
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)

                if req.return_logprob:
                    self.add_logprob_return_values(
                        i, req, pt, next_token_ids, logits_output
                    )
                    pt += req.extend_input_len
            _log_prefill_step("postprocess_generation")
        else:
            assert batch.extend_num_tokens != 0
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()
            logits_output = self.model_runner.forward(batch, ForwardMode.EXTEND)
            model_forward_end.record()
            torch.cuda.synchronize()
            self.last_model_forward_elapsed_ms = model_forward_start.elapsed_time(
                model_forward_end
            )
            _log_prefill_step("model_forward_extend_embedding")
            embeddings = logits_output.embeddings.tolist()
            _log_prefill_step("embeddings_to_cpu")

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                req.embedding = embeddings[i]
                if req is not self.current_inflight_req:
                    # Inflight reqs' prefill is not finished
                    # dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                else:
                    self.tree_cache.cache_unfinished_req(req)

                if req is self.current_inflight_req:
                    # Inflight request would get a new req idx
                    self.req_to_token_pool.free(req.req_pool_idx)
            _log_prefill_step("postprocess_embedding")

        self.handle_finished_requests(batch)
        _log_prefill_step("handle_finished_requests")
        logger.info(
            "Prefill timing total wall_ms=%.3f batch_size=%s extend_tokens=%s",
            (time.perf_counter() - prefill_wall_start) * 1000.0,
            batch.batch_size(),
            getattr(batch, "extend_num_tokens", "unset"),
        )

    def add_logprob_return_values(
        self,
        i,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitsProcessorOutput,
    ):
        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
            req.input_token_logprobs = list(
                zip(
                    output.input_token_logprobs[pt : pt + req.extend_input_len - 1],
                    req.fill_ids[-req.extend_input_len + 1 :],
                )
            )
            if req.logprob_start_len == 0:
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + req.extend_input_len
                            - req.last_update_decode_tokens : pt
                            + req.extend_input_len
                            - 1
                        ],
                        req.fill_ids[-req.last_update_decode_tokens + 1 :],
                    )
                )
            )

        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens + 1 :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

    def forward_decode_batch(
        self,
        batch: ScheduleBatch,
        *,
        selected_rids: Optional[set[str]] = None,
        prepare_pass_state: bool = True,
        decode_steps: int = 1,
    ):
        self.last_model_forward_elapsed_ms = None
        self.last_decode_step_breakdown = None
        decode_step_start = time.perf_counter()
        restore_state = None

        def apply_restricted_decode_subset() -> Optional[dict]:
            if not selected_rids:
                return None

            selected_indices = [i for i, req in enumerate(batch.reqs) if req.rid in selected_rids]
            if not selected_indices or len(selected_indices) == len(batch.reqs):
                return None

            deferred_indices = [i for i, req in enumerate(batch.reqs) if req.rid not in selected_rids]
            batch.reorder_batch(selected_indices + deferred_indices)
            selected_count = len(selected_indices)
            state = {
                "selected_count": selected_count,
                "deferred_reqs": list(batch.reqs[selected_count:]),
                "deferred_req_pool_indices": batch.req_pool_indices[selected_count:],
                "deferred_seq_lens": batch.seq_lens[selected_count:],
                "deferred_position_ids_offsets": batch.position_ids_offsets[selected_count:],
                "deferred_top_logprobs_nums": list(batch.top_logprobs_nums[selected_count:]),
                "full_sampling_attrs": {},
                "full_penalizer_attrs": {},
            }
            sampling_info = batch.sampling_info
            for attr in [
                "temperatures",
                "top_ps",
                "top_ks",
                "min_ps",
                "logit_bias",
                "vocab_mask",
                "linear_penalties",
                "scaling_penalties",
            ]:
                value = getattr(sampling_info, attr, None)
                state["full_sampling_attrs"][attr] = value
                if value is not None and hasattr(value, "shape") and value.shape[0] == len(batch.reqs):
                    setattr(sampling_info, attr, value[:selected_count])
            for penalizer in sampling_info.penalizer_orchestrator.penalizers.values():
                attrs = {}
                for attr, value in list(vars(penalizer).items()):
                    attrs[attr] = value
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim >= 1
                        and value.shape[0] == len(batch.reqs)
                    ):
                        setattr(penalizer, attr, value[:selected_count])
                state["full_penalizer_attrs"][id(penalizer)] = attrs
            logger.info(
                "EDF restricted decode batch to overdue-user requests only. selected=%s deferred=%s",
                selected_count,
                len(deferred_indices),
            )
            batch.reqs = list(batch.reqs[:selected_count])
            batch.req_pool_indices = batch.req_pool_indices[:selected_count]
            batch.seq_lens = batch.seq_lens[:selected_count]
            batch.position_ids_offsets = batch.position_ids_offsets[:selected_count]
            batch.top_logprobs_nums = list(batch.top_logprobs_nums[:selected_count])
            batch.return_logprob = any(req.return_logprob for req in batch.reqs)
            batch.input_ids = None
            batch.out_cache_loc = None
            return state

        def restore_restricted_decode_subset(state: dict) -> None:
            selected_survivors = len(batch.reqs)
            batch.reqs.extend(state["deferred_reqs"])
            batch.req_pool_indices = torch.concat(
                [batch.req_pool_indices[:selected_survivors], state["deferred_req_pool_indices"]]
            )
            batch.seq_lens = torch.concat(
                [batch.seq_lens[:selected_survivors], state["deferred_seq_lens"]]
            )
            batch.position_ids_offsets = torch.concat(
                [batch.position_ids_offsets[:selected_survivors], state["deferred_position_ids_offsets"]]
            )
            batch.top_logprobs_nums = (
                list(batch.top_logprobs_nums[:selected_survivors])
                + state["deferred_top_logprobs_nums"]
            )
            batch.return_logprob = any(req.return_logprob for req in batch.reqs)

            sampling_info = batch.sampling_info
            for attr, full_value in state["full_sampling_attrs"].items():
                cur_value = getattr(sampling_info, attr, None)
                if (
                    isinstance(full_value, torch.Tensor)
                    and full_value.ndim >= 1
                    and full_value.shape[0] == state["selected_count"] + len(state["deferred_reqs"])
                    and isinstance(cur_value, torch.Tensor)
                ):
                    setattr(
                        sampling_info,
                        attr,
                        torch.concat(
                            [cur_value[:selected_survivors], full_value[state["selected_count"] :]],
                            dim=0,
                        ),
                    )
                else:
                    setattr(sampling_info, attr, full_value)

            for penalizer in sampling_info.penalizer_orchestrator.penalizers.values():
                full_attrs = state["full_penalizer_attrs"][id(penalizer)]
                for attr, full_value in full_attrs.items():
                    cur_value = getattr(penalizer, attr, None)
                    if (
                        isinstance(full_value, torch.Tensor)
                        and full_value.ndim >= 1
                        and full_value.shape[0] == state["selected_count"] + len(state["deferred_reqs"])
                        and isinstance(cur_value, torch.Tensor)
                    ):
                        setattr(
                            penalizer,
                            attr,
                            torch.concat(
                                [cur_value[:selected_survivors], full_value[state["selected_count"] :]],
                                dim=0,
                            ),
                        )
                    else:
                        setattr(penalizer, attr, full_value)
            batch.input_ids = None
            batch.out_cache_loc = None

        restore_state = apply_restricted_decode_subset()
        # Check if decode out of memory
        try:
            if not batch.check_decode_mem():
                if restore_state is not None:
                    # Decode retraction must consider the full globally running batch, not the
                    # temporary EDF-restricted subset. Restore first, retract globally, then
                    # re-apply the subset on the surviving requests for the actual decode step.
                    restore_restricted_decode_subset(restore_state)
                    restore_state = None
                old_ratio = self.new_token_ratio

                retracted_reqs, new_token_ratio = batch.retract_decode()
                self.new_token_ratio = new_token_ratio

                logger.info(
                    "Decode out of memory happened. "
                    f"#retracted_reqs: {len(retracted_reqs)}, "
                    f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
                )
                if self.fairness_policy.uses_static_isolated_memory():
                    self.waiting_queue = list(retracted_reqs) + self.waiting_queue
                else:
                    self.waiting_queue.extend(retracted_reqs)
                self.fairness_policy.note_retracted_reqs(retracted_reqs)
                restore_state = apply_restricted_decode_subset()
            else:
                self.new_token_ratio = max(
                    self.new_token_ratio - self.new_token_ratio_decay,
                    self.min_new_token_ratio,
                )
            after_check_mem = time.perf_counter()

            if not self.disable_regex_jump_forward:
                # Check for jump-forward
                jump_forward_reqs = batch.check_for_jump_forward(self.model_runner)
                self.waiting_queue.extend(jump_forward_reqs)
                self.fairness_policy.note_retracted_reqs(jump_forward_reqs)
                if batch.is_empty():
                    self.last_decode_step_breakdown = {
                        "check_mem_ms": (after_check_mem - decode_step_start) * 1000.0,
                        "jump_forward_ms": (time.perf_counter() - after_check_mem) * 1000.0,
                        "prepare_ms": 0.0,
                        "prepare_breakdown": {},
                        "sync_wait_ms": 0.0,
                        "after_sync_ms": 0.0,
                        "sample_postprocess_ms": 0.0,
                        "handle_finished_ms": 0.0,
                    }
                    return
            after_jump_forward = time.perf_counter()

            # Update batch tensors
            self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
            batch.prepare_for_decode()
            prepare_breakdown = getattr(batch, "decode_prepare_breakdown", {})
            for req in batch.reqs:
                TIMELINE_WRITER.mark_first_decode_start(req.rid, req.uid)
            after_prepare = time.perf_counter()

            # Forward and sample the next tokens
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()
            sample_output, logits_output = self.model_runner.forward(
                batch, ForwardMode.DECODE
            )
            model_forward_end.record()
            fairness_prepare_start = time.perf_counter()
            self.fairness_policy.prepare_during_gpu_execution(
                event_type="decode",
                running_batch=batch,
                waiting_queue=list(self.waiting_queue),
                scheduled_batch=None,
                selected_rids=selected_rids,
                prepare_pass_state=prepare_pass_state,
                decode_steps=decode_steps,
                new_token_ratio=self.new_token_ratio,
            )
            fairness_prepare_end = time.perf_counter()
            next_token_ids = batch.check_sample_results(sample_output)
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                next_token_ids
            )

            # Move logprobs to cpu
            if logits_output.next_token_logprobs is not None:
                next_token_logprobs = logits_output.next_token_logprobs[
                    torch.arange(len(next_token_ids), device=next_token_ids.device),
                    next_token_ids,
                ].tolist()

            next_token_ids = next_token_ids.tolist()
            after_sync = time.perf_counter()
            self.last_model_forward_elapsed_ms = model_forward_start.elapsed_time(
                model_forward_end
            )
            after_sample_postprocess = time.perf_counter()

            # Check finish condition
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                req.completion_tokens_wo_jump_forward += 1
                req.output_ids.append(next_token_id)
                req.check_finished()

                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.get_next_state(
                        req.regex_fsm_state, next_token_id
                    )

                if req.finished():
                    self.tree_cache.cache_finished_req(req)

                if req.return_logprob:
                    req.output_token_logprobs.append(
                        (next_token_logprobs[i], next_token_id)
                    )
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs.append(logits_output.output_top_logprobs[i])

            self.handle_finished_requests(batch)
            after_handle_finished = time.perf_counter()
            fairness_prepare_breakdown = dict(
                getattr(self.fairness_policy, "_last_prepare_breakdown_ms", {})
            )
            self.last_decode_step_breakdown = {
                "check_mem_ms": (after_check_mem - decode_step_start) * 1000.0,
                "jump_forward_ms": (after_jump_forward - after_check_mem) * 1000.0,
                "prepare_ms": (after_prepare - after_jump_forward) * 1000.0,
                "prepare_breakdown": prepare_breakdown,
                "fairness_prepare_ms": (
                    fairness_prepare_end - fairness_prepare_start
                ) * 1000.0,
                "fairness_prepare_breakdown": fairness_prepare_breakdown,
                "sync_wait_ms": (after_sync - after_prepare) * 1000.0,
                "after_sync_ms": (after_sample_postprocess - after_sync) * 1000.0,
                "sample_postprocess_ms": (after_sample_postprocess - after_prepare) * 1000.0,
                "handle_finished_ms": (after_handle_finished - after_sample_postprocess) * 1000.0,
            }
        finally:
            if restore_state is not None:
                restore_restricted_decode_subset(restore_state)

    def handle_finished_requests(self, batch: ScheduleBatch):
        output_rids = []
        output_meta_info = []
        out_uids = []
        output_finished_reason: List[BaseFinishReason] = []
        if self.model_runner.is_generation:
            output_vids = []
            decoded_texts = []
            output_read_ids = []
            output_read_offsets = []
            output_skip_special_tokens = []
            output_spaces_between_special_tokens = []
        else:  # for embedding model
            output_embeddings = []
        unfinished_indices = []

        for i, req in enumerate(batch.reqs):
            if not req.finished() and req is not self.current_inflight_req:
                unfinished_indices.append(i)

            if req.finished() or (
                (
                    req.stream
                    and (
                        self.decode_forward_ct % self.stream_interval == 0
                        or len(req.output_ids) == 1
                    )
                )
            ):
                output_rids.append(req.rid)
                output_finished_reason.append(req.finished_reason)
                if req.finished():
                    self.fairness_policy.mark_request_finished(req)
                if self.model_runner.is_generation:
                    output_vids.append(req.vid)
                    out_uids.append(req.uid)
                    decoded_texts.append(req.decoded_text)
                    read_ids, read_offset = req.init_incremental_detokenize()
                    output_read_ids.append(read_ids)
                    output_read_offsets.append(read_offset)
                    output_skip_special_tokens.append(
                        req.sampling_params.skip_special_tokens
                    )
                    output_spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )

                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                        "completion_tokens": len(req.output_ids),
                        "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                        "finish_reason": str(req.finished_reason),
                    }
                    if req.return_logprob:
                        (
                            meta_info["input_token_logprobs"],
                            meta_info["output_token_logprobs"],
                            meta_info["input_top_logprobs"],
                            meta_info["output_top_logprobs"],
                            meta_info["normalized_prompt_logprob"],
                        ) = (
                            req.input_token_logprobs,
                            req.output_token_logprobs,
                            req.input_top_logprobs,
                            req.output_top_logprobs,
                            req.normalized_prompt_logprob,
                        )
                    output_meta_info.append(meta_info)
                else:  # for embedding model
                    output_embeddings.append(req.embedding)
                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                    }
                    output_meta_info.append(meta_info)

        # Send to detokenizer
        if output_rids:
            if self.model_runner.is_generation:
                self.out_pyobjs.append(
                    BatchTokenIDOut(
                        output_rids,
                        output_vids,
                        out_uids,
                        decoded_texts,
                        output_read_ids,
                        output_read_offsets,
                        output_skip_special_tokens,
                        output_spaces_between_special_tokens,
                        output_meta_info,
                        output_finished_reason,
                    )
                )
            else:  # for embedding model
                self.out_pyobjs.append(
                    BatchEmbeddingOut(
                        output_rids,
                        output_embeddings,
                        output_meta_info,
                        output_finished_reason,
                    )
                )

        # Remove finished reqs: update batch tensors
        batch.filter_batch(unfinished_indices)

    def flush_cache(self):
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def abort_request(self, recv_req):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid:
                    req.finished_reason = FINISH_ABORT()
                    break

    def update_weights(self, recv_req):
        success, message = self.model_runner.update_weights(
            recv_req.model_path, recv_req.load_format
        )
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        return success, message


def run_tp_server(
    gpu_id: int,
    tp_rank: int,
    server_args: ServerArgs,
    nccl_port: int,
    model_override_args: dict,
):
    """Run a tensor parallel model server."""
    configure_logger(server_args, prefix=f" TP{tp_rank}")

    try:
        model_server = ModelTpServer(
            gpu_id,
            tp_rank,
            server_args,
            nccl_port,
            model_override_args,
        )
        tp_cpu_group = model_server.model_runner.tp_group.cpu_group

        while True:
            recv_reqs = broadcast_recv_input(None, tp_rank, tp_cpu_group)
            model_server.exposed_step(recv_reqs)
    except Exception:
        logger.error("Exception in run_tp_server:\n" + get_exception_traceback())
        raise


def launch_tp_servers(
    gpu_ids: List[int],
    tp_rank_range: List[int],
    server_args: ServerArgs,
    nccl_port: int,
    model_override_args: dict,
):
    """Launch multiple tensor parallel servers."""
    procs = []
    for i in tp_rank_range:
        proc = multiprocessing.Process(
            target=run_tp_server,
            args=(gpu_ids[i], i, server_args, nccl_port, model_override_args),
        )
        proc.start()
        procs.append(proc)

    return procs


def broadcast_recv_input(
    data: Any, rank: int, dist_group: torch.distributed.ProcessGroup
):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(list(serialized_data))
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.tolist())
        data = pickle.loads(serialized_data)
        return data
