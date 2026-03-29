from __future__ import annotations

"""Design.md policy implementation."""

import logging
import os
import time
from copy import copy
from types import MappingProxyType, SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from sglang.global_config import global_config
from sglang.srt.request_timeline import TIMELINE_WRITER
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from .delta_fairness_policy import CLIP_MAX_NEW_TOKENS, DeltaFairnessPolicy
from .doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestEvent,
    RequestPrefillEvent,
    TrackedRequest,
)
from .simulator_thread import (
    _DocPolicyPrepareWorker,
    _FrozenPrepareCacheState,
    _FrozenPrepareInputs,
    _PreparedSnapshot,
)
from .time_estimation import (
    isolated_decode_time_estimation,
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

logger = logging.getLogger(__name__)
DOC_POLICY_TRACE_ENABLED = False


class DocPolicy(DeltaFairnessPolicy):
    def __init__(self, *args, **kwargs):
        self._pooled_quanta_us = int(
            kwargs.pop(
                "delta_fairness_pooled_quanta_us",
                kwargs.pop("delta_fairness_quanta_us", 0),
            )
            or 0
        )
        self._exclusive_quanta_us = int(
            kwargs.pop("delta_fairness_exclusive_quanta_us", 0) or 0
        )
        kwargs.pop("max_prefill_tokens", 0)
        isolated_kv_tokens_per_user = kwargs.pop("isolated_kv_tokens_per_user", None)
        schedule_conservativeness = float(kwargs.pop("schedule_conservativeness", 1.0))
        super().__init__(*args, **kwargs)
        min_new_token_ratio = min(
            global_config.base_min_new_token_ratio * schedule_conservativeness,
            1.0,
        )
        self.simulator = AlternateHistorySimulator(
            max_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
            enable_timeline_logging=False,
        )
        self._deltas_us = {"prefill": 0, "first_decode": 0, "decode": 0}
        self._deadline_queue = []
        self._waiting_prefill_start_deadline_by_rid: Dict[str, float] = {}
        self._safe_waiting_queue: List[Req] = []
        self._safe_waiting_rids: set[str] = set()
        self._forced_prefill_queue: List[Req] = []
        self._forced_prefill_rids: set[str] = set()
        self._max_safe_prefill_tokens: Optional[int] = None
        self._has_fair_waiting = False
        self._has_decode_deadline = False
        self._last_pass_breakdown_ms: Dict[str, float] = {}
        self._simulator_rebuild_prepared = False
        self._earliest_decode_start_deadline: Optional[float] = None
        self._safe_prefix_now: Optional[float] = None
        self._debug_first_waiting_rid: Optional[str] = None
        self._debug_earliest_decode_rid: Optional[str] = None
        self._debug_earliest_decode_uid: Optional[str] = None
        self._debug_first_waiting_prompt_tokens: Optional[int] = None
        self._debug_first_candidate_prefill_ms: Optional[float] = None
        self._debug_first_candidate_residual_slack_ms: Optional[float] = None
        self._prepared_deadline_queue = []
        self._prepared_waiting_prefill_start_deadline_by_rid: Dict[str, float] = {}
        self._prepared_safe_waiting_queue: List[Req] = []
        self._prepared_safe_waiting_rids: set[str] = set()
        self._prepared_forced_prefill_queue: List[Req] = []
        self._prepared_forced_prefill_rids: set[str] = set()
        self._prepared_max_safe_prefill_tokens: Optional[int] = None
        self._prepared_has_fair_waiting = False
        self._prepared_has_decode_deadline = False
        self._prepared_earliest_decode_start_deadline: Optional[float] = None
        self._prepared_safe_prefix_now: Optional[float] = None
        self._prepared_running_sig: Optional[Tuple[str, ...]] = None
        self._prepared_waiting_sig: Optional[Tuple[str, ...]] = None
        self._prepared_pass_state = None
        self._current_pass_running_sig: Optional[Tuple[str, ...]] = None
        self._current_pass_waiting_sig: Optional[Tuple[str, ...]] = None
        self._current_pass_id = 0
        self._last_prepare_breakdown_ms: Dict[str, float] = {}
        self._last_pass_state_source = "init"
        self._prefill_no_retraction_token_cap: Optional[int] = None
        self._last_consumed_prepare_snapshot_seq = 0
        self._last_prepare_task_seq = 0
        self._prepare_worker = _DocPolicyPrepareWorker(
            self,
            isolated_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
        )
        self._trace_csv_path = os.path.join(os.getcwd(), "doc_policy_trace.csv")

    def _trace(self, action: str, **fields) -> None:
        if not DOC_POLICY_TRACE_ENABLED:
            return
        keys = [
            "ts",
            "action",
            "pass_id",
            "task_seq",
            "snapshot_seq",
            "event_type",
            "prepare_pass_state",
            "waiting_len",
            "running_len",
            "scheduled_len",
            "selected_count",
            "decode_steps",
            "reason",
            "waiting_sig_match",
            "running_sig_match",
            "earliest_rid",
            "earliest_completion",
            "earliest_deadline",
            "last_consumed_snapshot_seq",
            "current_pass_id",
        ]
        row = {"ts": f"{time.time():.6f}", "action": action}
        row.update(fields)
        write_header = not os.path.exists(self._trace_csv_path) or os.path.getsize(
            self._trace_csv_path
        ) == 0
        with open(self._trace_csv_path, "a") as f:
            if write_header:
                f.write(",".join(keys) + "\n")
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    def _snapshot_req_for_prepare(self, req: Req) -> Req:
        return self._prepare_worker.snapshot_req(req)

    def _snapshot_batch_for_prepare(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        return self._prepare_worker.snapshot_batch(batch)

    def _enqueue_prepare_mutation(self, kind: str, payload) -> None:
        self._prepare_worker.enqueue_mutation(kind, payload)

    def _enqueue_prepare_task(self, task: tuple) -> int:
        return self._prepare_worker.enqueue_task(task)

    def _raise_prepare_thread_exception_if_any(self) -> None:
        self._prepare_worker.raise_exception_if_any()

    def _request_live_prepare_snapshot(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> Tuple[int, bool]:
        return self._prepare_worker.request_live_snapshot(
            waiting_queue,
            running_batch,
            last_consumed_snapshot_seq=self._last_consumed_prepare_snapshot_seq,
        )

    def _wait_for_matching_prepare_snapshot(
        self,
        *,
        min_task_seq: int,
        record_duplicate_wait: bool = False,
        timeout_s: float = 1.000,
    ) -> bool:
        return self._prepare_worker.wait_for_snapshot(
            min_task_seq=min_task_seq,
            record_duplicate_wait=record_duplicate_wait,
            timeout_s=timeout_s,
        )

    def _read_deltas(self, delta_fairness_deltas_microseconds: Optional[Dict[str, int]]) -> None:
        deltas = delta_fairness_deltas_microseconds or {}
        self._deltas_us = {
            "prefill": deltas.get("prefill", deltas.get("prefill_running_batch", 0)),
            "first_decode": deltas.get(
                "first_decode",
                deltas.get("first_decode_running_batch", deltas.get("decode_running_batch", 0)),
            ),
            "decode": deltas.get("decode", deltas.get("decode_running_batch", 0)),
        }

    def _freeze_prepare_cache_state(self) -> Optional[_FrozenPrepareCacheState]:
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None:
            return None
        total_counters = getattr(tree_cache, "total_user_counters", None)
        evictable_counters = getattr(tree_cache, "evictable_total_user_counters", None)
        total_user_tokens = (
            dict(total_counters.snapshot())
            if total_counters is not None and hasattr(total_counters, "snapshot")
            else {}
        )
        evictable_user_tokens = (
            dict(evictable_counters.snapshot())
            if evictable_counters is not None and hasattr(evictable_counters, "snapshot")
            else {}
        )
        unevictable_limit = None
        if hasattr(tree_cache, "calculate_delta_fair_reservation_size") and hasattr(
            tree_cache, "fairinf_delta_unevictable"
        ):
            unevictable_limit = int(
                tree_cache.calculate_delta_fair_reservation_size(
                    tree_cache.fairinf_delta_unevictable
                )
            )
        return _FrozenPrepareCacheState(
            total_user_tokens=total_user_tokens,
            evictable_user_tokens=evictable_user_tokens,
            fairinf_max_per_user=getattr(tree_cache, "fairinf_max_per_user", None),
            unevictable_limit=unevictable_limit,
        )

    def _freeze_prepare_inputs(
        self, *, new_token_ratio: float = 0.0
    ) -> _FrozenPrepareInputs:
        return _FrozenPrepareInputs(
            deltas_us=dict(self._deltas_us),
            no_retraction_cap=self._prefill_no_retraction_token_cap,
            new_token_ratio=float(new_token_ratio),
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
        )

    def _event_delta_seconds(self, tracked_req: TrackedRequest, event: RequestEvent) -> float:
        req_deltas = tracked_req.deltas_in_microseconds or self._deltas_us
        if isinstance(event, RequestPrefillEvent):
            delta_us = req_deltas.get("prefill", self._deltas_us["prefill"])
        elif isinstance(event, RequestDecodeEvent):
            if event.completion_number <= 1:
                delta_us = req_deltas.get("first_decode", self._deltas_us["first_decode"])
            else:
                delta_us = req_deltas.get("decode", self._deltas_us["decode"])
        else:
            delta_us = 0
        return float(delta_us) / 1_000_000.0

    def _user_is_fair_for_tracking(
        self, user_id: str, running_batch: Optional[ScheduleBatch]
    ) -> bool:
        del running_batch
        return super().user_is_fair_prefill(user_id, running_batch=None)

    def _user_is_fair_prefill_from_frozen(
        self,
        user_id: str,
        *,
        this_user_sum: int = 0,
        frozen_cache_state: Optional[_FrozenPrepareCacheState],
    ) -> bool:
        if not self.delta_fairness_n:
            return False
        if frozen_cache_state is None or frozen_cache_state.unevictable_limit is None:
            return True
        total_tokens = frozen_cache_state.total_user_tokens.get(user_id, 0)
        evictable_tokens = frozen_cache_state.evictable_user_tokens.get(user_id, 0)
        unevictable_used = total_tokens - evictable_tokens + this_user_sum
        return unevictable_used < frozen_cache_state.unevictable_limit

    def user_is_fair_prefill(
        self,
        user_id: str,
        *,
        running_batch: Optional[ScheduleBatch],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        del running_batch
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None or hasattr(
            tree_cache, "user_unevictable_kv_is_under_fair_share_reservation"
        ):
            return super().user_is_fair_prefill(
                user_id,
                running_batch=None,
                this_user_len=this_user_len,
                this_user_sum=this_user_sum,
            )
        original_tree_cache = self.tree_cache
        try:
            self.tree_cache = None
            return super().user_is_fair_prefill(
                user_id,
                running_batch=None,
                this_user_len=this_user_len,
                this_user_sum=this_user_sum,
            )
        finally:
            self.tree_cache = original_tree_cache

    def _force_prefill_within_user_headroom_from_frozen(
        self,
        req: Req,
        *,
        running_batch: Optional[ScheduleBatch],
        pending_prefill_tokens: int = 0,
        frozen_cache_state: Optional[_FrozenPrepareCacheState],
        new_token_ratio: float,
    ) -> bool:
        if frozen_cache_state is None or frozen_cache_state.fairinf_max_per_user is None:
            return True
        total_tokens = frozen_cache_state.total_user_tokens.get(req.uid, 0)
        evictable_tokens = frozen_cache_state.evictable_user_tokens.get(req.uid, 0)
        cached_unevictable_tokens = total_tokens - evictable_tokens
        uncached_running_tokens = 0
        if running_batch is not None and getattr(running_batch, "seq_lens", None) is not None:
            seq_lens_cpu = running_batch.seq_lens.cpu().tolist()
            for i, running_req in enumerate(running_batch.reqs):
                if running_req.uid != req.uid:
                    continue
                uncached_running_tokens += max(
                    0, int(seq_lens_cpu[i]) - len(running_req.prefix_indices)
                )
        ratio = max(0.0, float(new_token_ratio))
        decode_headroom = 0
        if running_batch is not None:
            for running_req in running_batch.reqs:
                if running_req.uid != req.uid:
                    continue
                remaining = max(
                    0,
                    running_req.sampling_params.max_new_tokens
                    - len(running_req.output_ids),
                )
                decode_headroom += int(min(remaining, 4096) * ratio)
        protected_tokens = (
            cached_unevictable_tokens + uncached_running_tokens + pending_prefill_tokens
        )
        return (
            protected_tokens + decode_headroom + req.extend_input_len
            <= frozen_cache_state.fairinf_max_per_user
        )

    def _force_prefill_within_user_headroom(
        self,
        req: Req,
        *,
        running_batch: Optional[ScheduleBatch],
        pending_prefill_tokens: int = 0,
    ) -> bool:
        return super()._force_prefill_within_user_headroom(
            req,
            running_batch=running_batch,
            pending_prefill_tokens=pending_prefill_tokens,
        )

    def _memory_pressure_active_for_prefill(self) -> bool:
        return self._prefill_no_retraction_token_cap is not None

    def _no_retraction_prefill_token_cap(
        self, running_batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        del running_batch
        return self._prefill_no_retraction_token_cap

    def _prefill_deadline_waiting_queue(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> List[Req]:
        if not self._memory_pressure_active_for_prefill():
            return waiting_queue
        return [
            req
            for req in waiting_queue
            if self.user_is_fair_prefill(
                req.uid,
                running_batch=running_batch,
                this_user_sum=req.get_estimated_prefill_impact(),
            )
        ]

    def _pooled_prefill_seconds(self, reqs: List[Req]) -> float:
        if not reqs:
            return 0.0
        prompt_tokens = [len(req.origin_input_ids) for req in reqs]
        return pooled_prefill_time_estimation(
            sum(prompt_tokens),
            max(prompt_tokens),
            len(prompt_tokens),
            max(int(self.delta_fairness_n or 1), 1),
        )

    def _current_pooled_decode_seconds(
        self, running_batch: Optional[ScheduleBatch]
    ) -> float:
        if running_batch is None or not running_batch.reqs:
            return 0.0
        token_counts = [
            len(req.fill_ids) if req.fill_ids is not None else len(req.origin_input_ids) + len(req.output_ids)
            for req in running_batch.reqs
        ]
        decode_s = pooled_decode_time_estimation(
            sum(token_counts),
            max(token_counts),
            len(token_counts),
            max(int(self.delta_fairness_n or 1), 1),
        )
        return decode_s

    def _compute_safe_prefix_state(
        self,
        deadline_queue,
        waiting_prefill_start_deadline_by_rid: Dict[str, float],
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        ordered_waiting_queue: Optional[Tuple[Req, ...]] = None,
        frozen_cache_state: Optional[_FrozenPrepareCacheState] = None,
        frozen_inputs: Optional[_FrozenPrepareInputs] = None,
    ) -> Dict[str, object]:
        if ordered_waiting_queue is None:
            indexed = list(enumerate(waiting_queue))
            indexed.sort(
                key=lambda item: (
                    0 if item[1].rid in waiting_prefill_start_deadline_by_rid else 1,
                    waiting_prefill_start_deadline_by_rid.get(item[1].rid, float("inf")),
                    item[0],
                )
            )
            safe_waiting_queue = [req for _, req in indexed]
        else:
            safe_waiting_queue = list(ordered_waiting_queue)
        safe_waiting_rids = {req.rid for req in safe_waiting_queue}
        forced_prefill_queue: List[Req] = []
        forced_prefill_rids: set[str] = set()
        max_safe_prefill_tokens: Optional[int] = None
        has_fair_waiting = bool(safe_waiting_queue)
        has_decode_deadline = False
        earliest_decode_start_deadline: Optional[float] = None
        safe_prefix_now: Optional[float] = None

        earliest_decode_candidate = min(
            (
                candidate
                for candidate in deadline_queue
                if candidate.event_type == "decode"
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        if earliest_decode_candidate is None:
            return {
                "safe_waiting_queue": safe_waiting_queue,
                "safe_waiting_rids": safe_waiting_rids,
                "forced_prefill_queue": forced_prefill_queue,
                "forced_prefill_rids": forced_prefill_rids,
                "max_safe_prefill_tokens": max_safe_prefill_tokens,
                "has_fair_waiting": has_fair_waiting,
                "has_decode_deadline": has_decode_deadline,
                "earliest_decode_start_deadline": earliest_decode_start_deadline,
                "safe_prefix_now": safe_prefix_now,
            }

        has_decode_deadline = True
        earliest_decode_start_deadline = earliest_decode_candidate.start_deadline
        now = time.time()
        safe_prefix_now = now
        selected_batch: List[Req] = []
        safe_prompt_tokens = 0
        no_retraction_cap = (
            frozen_inputs.no_retraction_cap
            if frozen_inputs is not None
            else self._no_retraction_prefill_token_cap(running_batch)
        )
        pending_prefill_sum_by_user: Dict[str, int] = {}
        pending_prefill_len_by_user: Dict[str, int] = {}
        fair_user_by_uid: Dict[str, bool] = {}
        for req in safe_waiting_queue:
            req_prefill_tokens = len(req.origin_input_ids)
            next_safe_prompt_tokens = safe_prompt_tokens + req_prefill_tokens
            beyond_no_retraction_cap = (
                no_retraction_cap is not None
                and next_safe_prompt_tokens > no_retraction_cap
            )
            if beyond_no_retraction_cap:
                is_fair = fair_user_by_uid.get(req.uid)
                if is_fair is None:
                    this_user_sum = pending_prefill_sum_by_user.get(req.uid, 0)
                    is_fair = self._user_is_fair_prefill_from_frozen(
                        req.uid,
                        this_user_sum=req.get_estimated_prefill_impact() + this_user_sum,
                        frozen_cache_state=frozen_cache_state,
                    )
                    fair_user_by_uid[req.uid] = is_fair
                if not is_fair:
                    continue
                if not self._force_prefill_within_user_headroom_from_frozen(
                    req,
                    running_batch=running_batch,
                    pending_prefill_tokens=pending_prefill_sum_by_user.get(req.uid, 0),
                    frozen_cache_state=frozen_cache_state,
                    new_token_ratio=(
                        frozen_inputs.new_token_ratio if frozen_inputs is not None else 0.0
                    ),
                ):
                    continue
            candidate_batch = selected_batch + [req]
            prompt_tokens = [len(item.origin_input_ids) for item in candidate_batch]
            pooled_prefill_s = pooled_prefill_time_estimation(
                sum(prompt_tokens),
                max(prompt_tokens),
                len(prompt_tokens),
                (
                    frozen_inputs.fairinf_n
                    if frozen_inputs is not None
                    else max(int(self.delta_fairness_n or 1), 1)
                ),
            )
            if now + pooled_prefill_s <= earliest_decode_start_deadline:
                safe_prompt_tokens = next_safe_prompt_tokens
                selected_batch.append(req)
                forced_prefill_queue.append(req)
                forced_prefill_rids.add(req.rid)
                pending_prefill_sum_by_user[req.uid] = (
                    pending_prefill_sum_by_user.get(req.uid, 0) + req_prefill_tokens
                )
                pending_prefill_len_by_user[req.uid] = (
                    pending_prefill_len_by_user.get(req.uid, 0) + 1
                )
                continue
            break

        max_safe_prefill_tokens = safe_prompt_tokens
        return {
            "safe_waiting_queue": safe_waiting_queue,
            "safe_waiting_rids": safe_waiting_rids,
            "forced_prefill_queue": forced_prefill_queue,
            "forced_prefill_rids": forced_prefill_rids,
            "max_safe_prefill_tokens": max_safe_prefill_tokens,
            "has_fair_waiting": has_fair_waiting,
            "has_decode_deadline": has_decode_deadline,
            "earliest_decode_start_deadline": earliest_decode_start_deadline,
            "safe_prefix_now": safe_prefix_now,
        }

    def _recompute_safe_prefix(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        phase_start = time.perf_counter()
        indexed = list(enumerate(waiting_queue))
        indexed.sort(
            key=lambda item: (
                0
                if item[1].rid in self._waiting_prefill_start_deadline_by_rid
                else 1,
                self._waiting_prefill_start_deadline_by_rid.get(
                    item[1].rid, float("inf")
                ),
                item[0],
            )
        )
        after_waiting_sort = time.perf_counter()
        self._safe_waiting_queue = [req for _, req in indexed]
        self._safe_waiting_rids = {req.rid for req in self._safe_waiting_queue}
        self._has_fair_waiting = bool(self._safe_waiting_queue)
        self._forced_prefill_queue = []
        self._max_safe_prefill_tokens = None
        self._forced_prefill_rids = set()
        self._has_decode_deadline = False
        self._earliest_decode_start_deadline = None
        self._safe_prefix_now = None
        self._debug_first_waiting_rid = None
        self._debug_earliest_decode_rid = None
        self._debug_earliest_decode_uid = None
        self._debug_first_waiting_prompt_tokens = None
        self._debug_first_candidate_prefill_ms = None
        self._debug_first_candidate_residual_slack_ms = None

        earliest_decode_candidate = min(
            (
                candidate
                for candidate in self._deadline_queue
                if candidate.event_type == "decode"
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        if earliest_decode_candidate is None:
            if timing_breakdown is not None:
                timing_breakdown["sort_waiting_prefills_ms"] = (
                    after_waiting_sort - phase_start
                ) * 1000.0
                timing_breakdown["safe_prefix_scan_ms"] = 0.0
                timing_breakdown["build_pass_state_ms"] = (
                    after_waiting_sort - phase_start
                ) * 1000.0
            return
        self._has_decode_deadline = True
        earliest_decode_deadline = earliest_decode_candidate.start_deadline
        self._earliest_decode_start_deadline = earliest_decode_deadline
        self._debug_earliest_decode_rid = earliest_decode_candidate.req.rid
        self._debug_earliest_decode_uid = earliest_decode_candidate.req.uid

        now = time.time()
        self._safe_prefix_now = now
        selected_batch: List[Req] = []
        safe_prompt_tokens = 0
        no_retraction_cap = self._no_retraction_prefill_token_cap(running_batch)
        pending_prefill_sum_by_user: Dict[str, int] = {}
        pending_prefill_len_by_user: Dict[str, int] = {}
        for req in self._safe_waiting_queue:
            req_prefill_tokens = len(req.origin_input_ids)
            next_safe_prompt_tokens = safe_prompt_tokens + req_prefill_tokens
            beyond_no_retraction_cap = (
                no_retraction_cap is not None
                and next_safe_prompt_tokens > no_retraction_cap
            )
            if beyond_no_retraction_cap:
                this_user_sum = pending_prefill_sum_by_user.get(req.uid, 0)
                this_user_len = pending_prefill_len_by_user.get(req.uid, 0)
                if not self.user_is_fair_prefill(
                    req.uid,
                    running_batch=running_batch,
                    this_user_len=this_user_len,
                    this_user_sum=this_user_sum,
                ):
                    continue
                if not self._force_prefill_within_user_headroom(
                    req,
                    running_batch=running_batch,
                    pending_prefill_tokens=this_user_sum,
                ):
                    continue
            candidate_batch = selected_batch + [req]
            pooled_prefill_s = self._pooled_prefill_seconds(candidate_batch)
            if len(selected_batch) == 0:
                self._debug_first_waiting_rid = req.rid
                self._debug_first_waiting_prompt_tokens = len(req.origin_input_ids)
                self._debug_first_candidate_prefill_ms = pooled_prefill_s * 1000.0
                self._debug_first_candidate_residual_slack_ms = (
                    earliest_decode_deadline - (now + pooled_prefill_s)
                ) * 1000.0
            if now + pooled_prefill_s <= earliest_decode_deadline:
                safe_prompt_tokens = next_safe_prompt_tokens
                selected_batch.append(req)
                self._forced_prefill_queue.append(req)
                self._forced_prefill_rids.add(req.rid)
                pending_prefill_sum_by_user[req.uid] = (
                    pending_prefill_sum_by_user.get(req.uid, 0) + req_prefill_tokens
                )
                pending_prefill_len_by_user[req.uid] = (
                    pending_prefill_len_by_user.get(req.uid, 0) + 1
                )
                continue
            break

        self._max_safe_prefill_tokens = safe_prompt_tokens
        after_safe_scan = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["sort_waiting_prefills_ms"] = (
                after_waiting_sort - phase_start
            ) * 1000.0
            timing_breakdown["safe_prefix_scan_ms"] = (
                after_safe_scan - after_waiting_sort
            ) * 1000.0
            timing_breakdown["build_pass_state_ms"] = (
                after_safe_scan - phase_start
            ) * 1000.0

    def _build_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]],
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        phase_start = time.perf_counter()
        self._read_deltas(delta_fairness_deltas_microseconds)
        self.simulator.get_live_users(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=self._deltas_us,
        )
        deadline_waiting_queue = self._prefill_deadline_waiting_queue(waiting_queue, running_batch)
        deadline_result = self.simulator.build_deadline_candidates(
                deadline_waiting_queue,
                running_batch,
                include_ordered_waiting_queue=True,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=self._event_delta_seconds,
                pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                    rb
                ),
            )
        if len(deadline_result) == 3:
            (
                self._deadline_queue,
                self._waiting_prefill_start_deadline_by_rid,
                ordered_waiting_queue,
            ) = deadline_result
        else:
            self._deadline_queue, self._waiting_prefill_start_deadline_by_rid = deadline_result
            ordered_waiting_queue = None
        after_deadline_build = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["build_deadline_candidates_ms"] = (
                after_deadline_build - phase_start
            ) * 1000.0
        self._recompute_safe_prefix(waiting_queue, running_batch, timing_breakdown=timing_breakdown)

    def _queue_sig(self, reqs: List[Req]) -> Tuple[str, ...]:
        return tuple(req.rid for req in reqs)

    def _running_sig(self, running_batch: Optional[ScheduleBatch]) -> Tuple[str, ...]:
        if running_batch is None:
            return ()
        return tuple(req.rid for req in running_batch.reqs)

    def _predicted_decode_running_batch(
        self,
        running_batch: ScheduleBatch,
        selected_rids: Optional[set[str]],
    ) -> SimpleNamespace:
        predicted_reqs = []
        for req in running_batch.reqs:
            predicted_req = copy(req)
            predicted_req.output_ids = list(req.output_ids)
            if selected_rids is None or req.rid in selected_rids:
                predicted_req.output_ids = list(req.output_ids) + [0]
            predicted_reqs.append(predicted_req)
        return SimpleNamespace(reqs=predicted_reqs)

    def _prepare_deadline_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> None:
        target_simulator = self.simulator if simulator is None else simulator
        target_simulator.get_live_users(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=self._deltas_us,
        )
        deadline_waiting_queue = self._prefill_deadline_waiting_queue(waiting_queue, running_batch)
        deadline_result = target_simulator.build_deadline_candidates(
                deadline_waiting_queue,
                running_batch,
                include_ordered_waiting_queue=True,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=self._event_delta_seconds,
                pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                    rb
                ),
            )
        if len(deadline_result) == 3:
            (
                self._prepared_deadline_queue,
                self._prepared_waiting_prefill_start_deadline_by_rid,
                _prepared_ordered_waiting_queue,
            ) = deadline_result
        else:
            (
                self._prepared_deadline_queue,
                self._prepared_waiting_prefill_start_deadline_by_rid,
            ) = deadline_result
        self._prepared_waiting_sig = self._queue_sig(waiting_queue)
        self._prepared_running_sig = self._running_sig(running_batch)
        old_deadline_queue = self._deadline_queue
        old_waiting_deadlines = self._waiting_prefill_start_deadline_by_rid
        old_safe_waiting_queue = self._safe_waiting_queue
        old_safe_waiting_rids = self._safe_waiting_rids
        old_forced_prefill_queue = self._forced_prefill_queue
        old_forced_prefill_rids = self._forced_prefill_rids
        old_max_safe_prefill_tokens = self._max_safe_prefill_tokens
        old_has_fair_waiting = self._has_fair_waiting
        old_has_decode_deadline = self._has_decode_deadline
        old_earliest_decode_start_deadline = self._earliest_decode_start_deadline
        old_safe_prefix_now = self._safe_prefix_now
        try:
            self._deadline_queue = list(self._prepared_deadline_queue)
            self._waiting_prefill_start_deadline_by_rid = dict(
                self._prepared_waiting_prefill_start_deadline_by_rid
            )
            self._recompute_safe_prefix(waiting_queue, running_batch)
            self._prepared_safe_waiting_queue = list(self._safe_waiting_queue)
            self._prepared_safe_waiting_rids = set(self._safe_waiting_rids)
            self._prepared_forced_prefill_queue = list(self._forced_prefill_queue)
            self._prepared_forced_prefill_rids = set(self._forced_prefill_rids)
            self._prepared_max_safe_prefill_tokens = self._max_safe_prefill_tokens
            self._prepared_has_fair_waiting = self._has_fair_waiting
            self._prepared_has_decode_deadline = self._has_decode_deadline
            self._prepared_earliest_decode_start_deadline = (
                self._earliest_decode_start_deadline
            )
            self._prepared_safe_prefix_now = self._safe_prefix_now
            self._prepared_pass_state = True
        finally:
            self._deadline_queue = old_deadline_queue
            self._waiting_prefill_start_deadline_by_rid = old_waiting_deadlines
            self._safe_waiting_queue = old_safe_waiting_queue
            self._safe_waiting_rids = old_safe_waiting_rids
            self._forced_prefill_queue = old_forced_prefill_queue
            self._forced_prefill_rids = old_forced_prefill_rids
            self._max_safe_prefill_tokens = old_max_safe_prefill_tokens
            self._has_fair_waiting = old_has_fair_waiting
            self._has_decode_deadline = old_has_decode_deadline
            self._earliest_decode_start_deadline = old_earliest_decode_start_deadline
            self._safe_prefix_now = old_safe_prefix_now

    def _build_prepare_snapshot(
        self,
        simulator: AlternateHistorySimulator,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        task_seq: int,
        mutation_seq: int,
        breakdown: Optional[Dict[str, float]] = None,
        frozen_cache_state: Optional[_FrozenPrepareCacheState] = None,
        frozen_inputs: Optional[_FrozenPrepareInputs] = None,
    ) -> _PreparedSnapshot:
        target_breakdown: Dict[str, float] = {} if breakdown is None else breakdown
        build_start = time.perf_counter()
        simulator.get_live_users(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=(
                frozen_inputs.deltas_us if frozen_inputs is not None else self._deltas_us
            ),
        )
        after_sync = time.perf_counter()
        if frozen_inputs is not None and frozen_inputs.no_retraction_cap is not None:
            deadline_waiting_queue = [
                req
                for req in waiting_queue
                if self._user_is_fair_prefill_from_frozen(
                    req.uid,
                    this_user_sum=req.get_estimated_prefill_impact(),
                    frozen_cache_state=frozen_cache_state,
                )
            ]
        else:
            deadline_waiting_queue = waiting_queue
        fairinf_n = (
            frozen_inputs.fairinf_n
            if frozen_inputs is not None
            else max(int(self.delta_fairness_n or 1), 1)
        )
        deadline_result = simulator.build_deadline_candidates(
            deadline_waiting_queue,
            running_batch,
            include_ordered_waiting_queue=True,
            req_is_fair_prefill=lambda req, rb: self._user_is_fair_prefill_from_frozen(
                req.uid,
                this_user_sum=req.get_estimated_prefill_impact(),
                frozen_cache_state=frozen_cache_state,
            ),
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked_req, event: float(
                (
                    tracked_req.deltas_in_microseconds
                    or (
                        frozen_inputs.deltas_us
                        if frozen_inputs is not None
                        else self._deltas_us
                    )
                ).get(
                    "prefill"
                    if isinstance(event, RequestPrefillEvent)
                    else (
                        "first_decode"
                        if isinstance(event, RequestDecodeEvent)
                        and event.completion_number <= 1
                        else "decode"
                    ),
                    0,
                )
            )
            / 1_000_000.0,
            pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                len(req.origin_input_ids),
                len(req.origin_input_ids),
                1,
                fairinf_n,
            ),
            pooled_decode_estimate_seconds=lambda req, rb: (
                0.0
                if rb is None or not rb.reqs
                else pooled_decode_time_estimation(
                    sum(
                        len(item.fill_ids)
                        if item.fill_ids is not None
                        else len(item.origin_input_ids) + len(item.output_ids)
                        for item in rb.reqs
                    ),
                    max(
                        len(item.fill_ids)
                        if item.fill_ids is not None
                        else len(item.origin_input_ids) + len(item.output_ids)
                        for item in rb.reqs
                    ),
                    len(rb.reqs),
                    fairinf_n,
                )
            ),
        )
        if len(deadline_result) == 3:
            (
                deadline_queue,
                waiting_prefill_deadline_by_rid,
                ordered_waiting_queue,
            ) = deadline_result
        else:
            deadline_queue, waiting_prefill_deadline_by_rid = deadline_result
            ordered_waiting_queue = None
        after_deadline = time.perf_counter()
        safe_state = self._compute_safe_prefix_state(
            deadline_queue,
            waiting_prefill_deadline_by_rid,
            waiting_queue,
            running_batch,
            ordered_waiting_queue=ordered_waiting_queue,
            frozen_cache_state=frozen_cache_state,
            frozen_inputs=frozen_inputs,
        )
        after_safe = time.perf_counter()
        target_breakdown["sync_live_user_tracking_ms"] = (
            after_sync - build_start
        ) * 1000.0
        target_breakdown["build_deadline_candidates_ms"] = (
            after_deadline - after_sync
        ) * 1000.0
        target_breakdown["sort_waiting_prefills_ms"] = 0.0
        target_breakdown["safe_prefix_scan_ms"] = (
            after_safe - after_deadline
        ) * 1000.0
        target_breakdown["build_pass_state_ms"] = (
            after_safe - build_start
        ) * 1000.0
        return _PreparedSnapshot(
            task_seq=task_seq,
            mutation_seq=mutation_seq,
            waiting_sig=self._queue_sig(waiting_queue),
            running_sig=self._running_sig(running_batch),
            deadline_queue=tuple(deadline_queue),
            waiting_prefill_deadlines=MappingProxyType(
                dict(waiting_prefill_deadline_by_rid)
            ),
            safe_waiting_queue=tuple(safe_state["safe_waiting_queue"]),
            safe_waiting_rids=frozenset(safe_state["safe_waiting_rids"]),
            forced_prefill_queue=tuple(safe_state["forced_prefill_queue"]),
            forced_prefill_rids=frozenset(safe_state["forced_prefill_rids"]),
            max_safe_prefill_tokens=safe_state["max_safe_prefill_tokens"],
            has_fair_waiting=bool(safe_state["has_fair_waiting"]),
            has_decode_deadline=bool(safe_state["has_decode_deadline"]),
            earliest_decode_start_deadline=safe_state["earliest_decode_start_deadline"],
            earliest_decode_rid=self._debug_earliest_decode_rid,
            earliest_decode_uid=self._debug_earliest_decode_uid,
            safe_prefix_now=safe_state["safe_prefix_now"],
            breakdown_items=tuple(target_breakdown.items()),
        )

    def consume_prepare_thread_wait_metrics(self) -> Dict[str, float]:
        return self._prepare_worker.consume_wait_metrics()

    def _decode_candidate_for_req(
        self,
        req: Req,
        running_batch: Optional[ScheduleBatch],
    ):
        tracked = self.simulator.requests.get(req.rid)
        real_event = self.simulator.most_recent_event_real.get(req.rid)
        if tracked is None or real_event is None:
            return None
        if not isinstance(real_event, (RequestPrefillEvent, RequestDecodeEvent)):
            return None
        event = None
        for upcoming in tracked.earliest_events_after_real_time(real_event) or []:
            if isinstance(upcoming, RequestDecodeEvent):
                event = upcoming
                break
        if event is None:
            token_count = len(req.origin_input_ids) + max(1, len(req.output_ids))
            event = RequestDecodeEvent(
                req_id=req.rid,
                end_timestamp=real_event.end_timestamp
                + isolated_decode_time_estimation(
                    token_count,
                    token_count,
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                completion_number=len(req.output_ids) + 1,
            )
        deadline = event.end_timestamp + self._event_delta_seconds(tracked, event)
        return SimpleNamespace(
            deadline=deadline,
            start_deadline=deadline - self._current_pooled_decode_seconds(running_batch),
            event_type="decode",
            req=req,
            event=event,
        )

    def _async_snapshot_req(self, req: Req, *, output_delta: int = 0) -> Req:
        cloned = copy(req)
        cloned.origin_input_ids = list(req.origin_input_ids)
        cloned.output_ids = list(req.output_ids) + ([0] * output_delta)
        cloned.fill_ids = None if req.fill_ids is None else list(req.fill_ids)
        return cloned

    def _async_prepare_decode_epoch(
        self,
        *,
        running_batch: ScheduleBatch,
        waiting_queue: List[Req],
        selected_rids: Optional[set[str]],
        decode_steps: int,
    ) -> None:
        try:
            predicted_running = SimpleNamespace(reqs=list(running_batch.reqs))
            predicted_waiting = list(waiting_queue)
            self.simulator.get_live_users(
                predicted_running,
                predicted_waiting,
                deltas_in_microseconds=self._deltas_us,
            )
            affected_user_ids = sorted(
                {
                    req.uid
                    for req in running_batch.reqs
                    if selected_rids is None or req.rid in selected_rids
                }
            )
            self._apply_logical_decode_updates(
                self.simulator,
                predicted_running,
                selected_rids=selected_rids,
                decode_steps=decode_steps,
            )
            deadline_start = time.perf_counter()
            deadline_result = self.simulator.build_deadline_candidates(
                    predicted_waiting,
                    predicted_running,
                    req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                        req.uid,
                        running_batch=rb,
                        this_user_sum=req.get_estimated_prefill_impact(),
                    ),
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=self._event_delta_seconds,
                    pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                        len(req.origin_input_ids),
                        len(req.origin_input_ids),
                        1,
                        max(int(self.delta_fairness_n or 1), 1),
                    ),
                    pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                        rb
                    ),
                )
            if len(deadline_result) == 3:
                deadline_queue, waiting_prefill_start_deadline_by_rid, _ = deadline_result
            else:
                deadline_queue, waiting_prefill_start_deadline_by_rid = deadline_result
            deadline_elapsed_ms = (time.perf_counter() - deadline_start) * 1000.0
            result = {
                "deadline_queue": deadline_queue,
                "waiting_prefill_start_deadline_by_rid": waiting_prefill_start_deadline_by_rid,
                "waiting_sig": self._queue_sig(predicted_waiting),
                "running_sig": self._running_sig(predicted_running),
                "breakdown": {
                    "sync_live_user_tracking_ms": 0.0,
                    "logical_event_update_ms": 0.0,
                    "rebuild_from_real_state_ms": 0.0,
                    "rebuild_live_states_ms": 0.0,
                    "rebuild_state_setup_ms": 0.0,
                    "rebuild_scheduler_loop_ms": 0.0,
                    "rebuild_scheduler_step_count": 0,
                    "rebuild_prefill_step_count": 0,
                    "rebuild_decode_step_count": 0,
                    "build_deadline_candidates_ms": deadline_elapsed_ms,
                },
            }
            with self._async_prepare_lock:
                self._async_prepare_result = result
                self._async_prepare_exception = None
        except BaseException as exc:
            with self._async_prepare_lock:
                self._async_prepare_exception = exc
        finally:
            self._async_prepare_request = None

    def launch_async_decode_epoch_prepare(
        self,
        *,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        selected_rids: Optional[set[str]] = None,
        decode_steps: int = 1,
    ) -> bool:
        del running_batch, waiting_queue, selected_rids, decode_steps
        return False

    def wait_for_async_prepare(self) -> float:
        return 0.0

    def _consume_prepared_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        allow_waiting_sig_mismatch: bool = False,
    ) -> bool:
        snapshot = self._prepare_worker.latest_snapshot()
        if snapshot is None:
            self._trace(
                "consume_reject",
                reason="no_snapshot",
                waiting_len=len(waiting_queue),
                running_len=0 if running_batch is None else len(running_batch.reqs),
                last_consumed_snapshot_seq=self._last_consumed_prepare_snapshot_seq,
                current_pass_id=self._current_pass_id,
            )
            return False
        if snapshot.task_seq <= self._last_consumed_prepare_snapshot_seq:
            self._trace(
                "consume_reject",
                reason="old_snapshot_seq",
                snapshot_seq=snapshot.task_seq,
                last_consumed_snapshot_seq=self._last_consumed_prepare_snapshot_seq,
                current_pass_id=self._current_pass_id,
            )
            return False
        # Require a snapshot built from the current pass's prepare task so we
        # don't consume a snapshot from a previous pass that doesn't reflect
        # the mutations from the just-completed GPU step.
        if (
            self._last_prepare_task_seq > 0
            and snapshot.task_seq < self._last_prepare_task_seq
        ):
            self._trace(
                "consume_reject",
                reason="stale_prepare_task",
                snapshot_seq=snapshot.task_seq,
                last_prepare_task_seq=self._last_prepare_task_seq,
                current_pass_id=self._current_pass_id,
            )
            return False
        waiting_sig = self._queue_sig(waiting_queue)
        waiting_sig_match = snapshot.waiting_sig == waiting_sig
        # We always consume the freshest snapshot regardless of sig changes.
        # Candidates from requests no longer in the live batch are dropped
        # during remapping below, so a running-batch change is handled safely.
        waiting_by_rid = {req.rid: req for req in waiting_queue}
        running_by_rid = (
            {} if running_batch is None else {req.rid: req for req in running_batch.reqs}
        )
        live_req_by_rid = dict(waiting_by_rid)
        live_req_by_rid.update(running_by_rid)

        remapped_deadline_queue = []
        for candidate in snapshot.deadline_queue:
            live_req = live_req_by_rid.get(candidate.req.rid)
            if live_req is None:
                continue
            remapped_candidate = copy(candidate)
            remapped_candidate.req = live_req
            remapped_deadline_queue.append(remapped_candidate)

        self._deadline_queue = tuple(remapped_deadline_queue)
        if waiting_sig_match:
            remapped_waiting_prefill_deadlines = {
                rid: deadline
                for rid, deadline in snapshot.waiting_prefill_deadlines.items()
                if rid in waiting_by_rid
            }
            remapped_safe_order = tuple(
                waiting_by_rid[req.rid]
                for req in snapshot.safe_waiting_queue
                if req.rid in waiting_by_rid
            )
            safe_state = self._compute_safe_prefix_state(
                remapped_deadline_queue,
                remapped_waiting_prefill_deadlines,
                waiting_queue,
                running_batch,
                ordered_waiting_queue=remapped_safe_order,
            )
            self._waiting_prefill_start_deadline_by_rid = remapped_waiting_prefill_deadlines
            self._safe_waiting_queue = tuple(safe_state["safe_waiting_queue"])
            self._safe_waiting_rids = frozenset(safe_state["safe_waiting_rids"])
            self._forced_prefill_queue = tuple(safe_state["forced_prefill_queue"])
            self._forced_prefill_rids = frozenset(safe_state["forced_prefill_rids"])
            self._max_safe_prefill_tokens = safe_state["max_safe_prefill_tokens"]
            self._has_fair_waiting = bool(safe_state["has_fair_waiting"])
            self._has_decode_deadline = bool(safe_state["has_decode_deadline"])
            self._earliest_decode_start_deadline = safe_state[
                "earliest_decode_start_deadline"
            ]
            self._safe_prefix_now = safe_state["safe_prefix_now"]
        else:
            self._waiting_prefill_start_deadline_by_rid = {
                rid: deadline
                for rid, deadline in snapshot.waiting_prefill_deadlines.items()
                if rid in waiting_by_rid
            }
            preserved_safe = [
                waiting_by_rid[req.rid]
                for req in snapshot.safe_waiting_queue
                if req.rid in waiting_by_rid
            ]
            seen_safe = {req.rid for req in preserved_safe}
            preserved_safe.extend(req for req in waiting_queue if req.rid not in seen_safe)
            self._safe_waiting_queue = tuple(preserved_safe)
            self._safe_waiting_rids = frozenset(req.rid for req in self._safe_waiting_queue)
            self._forced_prefill_queue = tuple(
                waiting_by_rid[req.rid]
                for req in snapshot.forced_prefill_queue
                if req.rid in waiting_by_rid
            )
            self._forced_prefill_rids = frozenset(req.rid for req in self._forced_prefill_queue)
            self._max_safe_prefill_tokens = snapshot.max_safe_prefill_tokens
            self._has_fair_waiting = bool(self._safe_waiting_queue)
            self._has_decode_deadline = any(
                candidate.event_type == "decode" for candidate in self._deadline_queue
            )
            self._earliest_decode_start_deadline = min(
                (
                    candidate.start_deadline
                    for candidate in self._deadline_queue
                    if candidate.event_type == "decode"
                ),
                default=None,
            )
            self._safe_prefix_now = snapshot.safe_prefix_now
        earliest_decode_candidate = min(
            (
                candidate
                for candidate in self._deadline_queue
                if candidate.event_type == "decode"
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        self._debug_earliest_decode_rid = (
            None if earliest_decode_candidate is None else earliest_decode_candidate.req.rid
        )
        self._debug_earliest_decode_uid = (
            None if earliest_decode_candidate is None else earliest_decode_candidate.req.uid
        )
        self._simulator_rebuild_prepared = False
        self._last_consumed_prepare_snapshot_seq = snapshot.task_seq
        self._trace(
            "consume_accept",
            snapshot_seq=snapshot.task_seq,
            reason="" if waiting_sig_match else "running_only",
            waiting_len=len(waiting_queue),
            running_len=0 if running_batch is None else len(running_batch.reqs),
            earliest_rid=self._debug_earliest_decode_rid,
            earliest_completion=""
            if earliest_decode_candidate is None
            else getattr(getattr(earliest_decode_candidate, "event", None), "completion_number", ""),
            earliest_deadline=self._earliest_decode_start_deadline,
            last_consumed_snapshot_seq=self._last_consumed_prepare_snapshot_seq,
            current_pass_id=self._current_pass_id,
        )
        return True

    def _ensure_current_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
    ) -> None:
        del delta_fairness_deltas_microseconds
        waiting_sig = self._queue_sig(waiting_queue)
        running_sig = self._running_sig(running_batch)
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "ensure_consume_prepared"
            self._current_pass_waiting_sig = waiting_sig
            self._current_pass_running_sig = running_sig
            return
        if (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
        ):
            self._last_pass_state_source = "ensure_current_reuse"
            return
        task_seq, duplicate_wait = self._request_live_prepare_snapshot(
            waiting_queue, running_batch
        )
        if self._wait_for_matching_prepare_snapshot(
            min_task_seq=task_seq,
            record_duplicate_wait=duplicate_wait,
        ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "ensure_wait_prepare"
        else:
            self._last_pass_state_source = "ensure_no_prepared"
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig

    def refresh_decode_hot_path_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        if self._consume_prepared_pass_state(
            waiting_queue,
            running_batch,
            allow_waiting_sig_mismatch=True,
        ):
            self._last_pass_state_source = "hot_path_consume_prepared"

    def start_of_pass(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        new_token_ratio: float = 0.0,
        max_running_requests: Optional[int] = None,
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        self._prepare_worker.wait_for_mutation_queue_below_limit(100)
        super().start_of_pass(
            running_batch,
            waiting_queue,
            new_token_ratio=new_token_ratio,
            max_running_requests=max_running_requests,
        )
        breakdown: Dict[str, float] = {}
        pass_start = time.perf_counter()
        waiting_sig = self._queue_sig(waiting_queue)
        running_sig = self._running_sig(running_batch)

        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "start_of_pass_consume_prepared"
        else:
            self._last_pass_state_source = "start_of_pass_no_prepared"

        after_simulator = time.perf_counter()
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig
        self._current_pass_id += 1
        after_build = time.perf_counter()
        breakdown["doc_policy_after_super_ms"] = (after_simulator - pass_start) * 1000.0
        breakdown["doc_policy_start_of_pass_total_ms"] = (
            after_build - pass_start
        ) * 1000.0
        self._last_pass_breakdown_ms = breakdown

    def _logical_event_timestamp(
        self,
        req: Req,
        *,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> float:
        target = self.simulator if simulator is None else simulator
        real_event = target.most_recent_event_real.get(req.rid)
        if real_event is not None:
            return float(real_event.end_timestamp)
        tracked = target.requests.get(req.rid)
        if tracked is not None:
            return float(tracked.arrival_timestamp)
        return 0.0

    def _logical_next_event_timestamp(
        self,
        req: Req,
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> float:
        target = self.simulator if simulator is None else simulator
        tracked = target.requests.get(req.rid)
        real_event = target.most_recent_event_real.get(req.rid)
        if tracked is not None and real_event is not None:
            upcoming = tracked.earliest_events_after_real_time(real_event) or []
            for event in upcoming:
                if event_type == "prefill" and isinstance(event, RequestPrefillEvent):
                    return float(event.end_timestamp)
                if (
                    event_type == "decode"
                    and isinstance(event, RequestDecodeEvent)
                    and (
                        completion_number is None
                        or event.completion_number == completion_number
                    )
                ):
                    return float(event.end_timestamp)
        return self._logical_event_timestamp(req, simulator=target)

    def _apply_logical_decode_updates(
        self,
        simulator: AlternateHistorySimulator,
        running_batch,
        *,
        selected_rids: Optional[set[str]],
        decode_steps: int,
        decode_steps_by_rid: Optional[Dict[str, int]] = None,
        output_ids_already_applied: bool = False,
    ) -> None:
        chosen = selected_rids
        for req in running_batch.reqs:
            if chosen is not None and req.rid not in chosen:
                continue
            tracked = simulator.requests.get(req.rid)
            if tracked is None:
                continue
            req_decode_steps = (
                int(decode_steps_by_rid.get(req.rid, 0))
                if decode_steps_by_rid is not None
                else int(decode_steps)
            )
            if req_decode_steps <= 0:
                continue
            final_logical_ts = None
            previous_logical_ts = self._logical_event_timestamp(
                req, simulator=simulator
            )
            if output_ids_already_applied:
                start_completion_number = max(
                    1, len(req.output_ids) - req_decode_steps + 1
                )
            else:
                start_completion_number = len(req.output_ids) + 1
            for completion_number in range(
                start_completion_number, start_completion_number + req_decode_steps
            ):
                context_tokens = len(req.origin_input_ids) + completion_number
                logical_ts = previous_logical_ts + isolated_decode_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                )
                simulator.most_recent_event_real[req.rid] = RequestDecodeEvent(
                    req_id=req.rid,
                    end_timestamp=logical_ts,
                    completion_number=completion_number,
                )
                final_logical_ts = logical_ts
                previous_logical_ts = logical_ts
            if final_logical_ts is None:
                continue
            next_completion_number = start_completion_number + req_decode_steps
            context_tokens = len(req.origin_input_ids) + next_completion_number
            decode_duration = isolated_decode_time_estimation(
                context_tokens,
                context_tokens,
                1,
                max(int(self.delta_fairness_n or 1), 1),
            )
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestDecodeEvent(
                    req_id=req.rid,
                    duration=decode_duration,
                    end_timestamp=final_logical_ts + decode_duration,
                    completion_number=next_completion_number,
                )
            ]

    def _predicted_running_batch(
        self,
        running_batch: Optional[ScheduleBatch],
        scheduled_batch: Optional[ScheduleBatch],
    ) -> Optional[SimpleNamespace]:
        running_reqs = [] if running_batch is None else list(running_batch.reqs)
        if scheduled_batch is not None:
            running_reqs = running_reqs + list(scheduled_batch.reqs)
        if not running_reqs:
            return None
        return SimpleNamespace(reqs=running_reqs)

    def prepare_during_gpu_execution(
        self,
        *,
        event_type: str,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        scheduled_batch: Optional[ScheduleBatch] = None,
        selected_rids: Optional[set[str]] = None,
        prepare_pass_state: bool = True,
        decode_steps: int = 1,
        decode_steps_by_rid: Optional[Dict[str, int]] = None,
        output_ids_already_applied: bool = False,
        new_token_ratio: float = 0.0,
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        with torch.inference_mode():
            phase_start = time.perf_counter()
            running_snapshot = self._snapshot_batch_for_prepare(running_batch)
            waiting_snapshot = [
                self._snapshot_req_for_prepare(req) for req in waiting_queue
            ]
            scheduled_snapshot = self._snapshot_batch_for_prepare(scheduled_batch)
            if event_type == "decode" and running_batch is not None and (
                decode_steps > 0 or decode_steps_by_rid
            ):
                chosen_decode_rids = (
                    {
                        rid
                        for rid, steps in (decode_steps_by_rid or {}).items()
                        if int(steps) > 0
                    }
                    if decode_steps_by_rid is not None
                    else selected_rids
                )
                self._apply_logical_decode_updates(
                    self.simulator,
                    running_batch,
                    selected_rids=chosen_decode_rids,
                    decode_steps=decode_steps,
                    decode_steps_by_rid=decode_steps_by_rid,
                    output_ids_already_applied=output_ids_already_applied,
                )
                self._enqueue_prepare_mutation(
                    "logical_decode_update",
                    (self._snapshot_batch_for_prepare(running_batch),),
                )
                chosen = chosen_decode_rids
                now_ts = time.time()
                for req in running_batch.reqs:
                    if chosen is None or req.rid in chosen:
                        pass
            task_seq = self._enqueue_prepare_task(
                (
                    event_type,
                    running_snapshot,
                    waiting_snapshot,
                    scheduled_snapshot,
                    None if event_type != "decode" or selected_rids is None else set(selected_rids),
                    prepare_pass_state,
                    decode_steps,
                    None if decode_steps_by_rid is None else dict(decode_steps_by_rid),
                    output_ids_already_applied,
                    new_token_ratio,
                    self._freeze_prepare_cache_state(),
                    self._freeze_prepare_inputs(new_token_ratio=new_token_ratio),
                    self._current_pass_id,
                    self._prepare_worker.mutation_seq,
                )
            )
            if prepare_pass_state:
                self._last_prepare_task_seq = task_seq
            self._trace(
                "enqueue_prepare",
                pass_id=self._current_pass_id,
                task_seq=task_seq,
                event_type=event_type,
                prepare_pass_state=int(bool(prepare_pass_state)),
                waiting_len=len(waiting_snapshot),
                running_len=0 if running_snapshot is None else len(running_snapshot.reqs),
                scheduled_len=0 if scheduled_snapshot is None else len(scheduled_snapshot.reqs),
                selected_count=0 if selected_rids is None else len(selected_rids),
                decode_steps=decode_steps if decode_steps_by_rid is None else sum(decode_steps_by_rid.values()),
                current_pass_id=self._current_pass_id,
            )
            if not prepare_pass_state:
                elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
                self._last_prepare_breakdown_ms = {
                    "sync_live_user_tracking_ms": 0.0,
                    "logical_event_update_ms": elapsed_ms,
                    "rebuild_from_real_state_ms": 0.0,
                    "build_deadline_candidates_ms": 0.0,
                    "prepare_during_gpu_execution_total_ms": elapsed_ms,
                }
                return
            elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
            self._last_prepare_breakdown_ms = {
                "sync_live_user_tracking_ms": 0.0,
                "logical_event_update_ms": 0.0,
                "rebuild_from_real_state_ms": 0.0,
                "build_deadline_candidates_ms": 0.0,
                "prepare_during_gpu_execution_total_ms": elapsed_ms,
            }                

    def process_new_request(self, req: Req) -> None:
        arrival_ts = time.time()
        super().process_new_request(req)
        self._enqueue_prepare_mutation(
            "process_new_request",
            (self._snapshot_req_for_prepare(req), dict(self._deltas_us), arrival_ts),
        )

    def note_scheduled_prefill_batch(self, batch: ScheduleBatch) -> None:
        self._enqueue_prepare_mutation(
            "note_scheduled_prefill_batch",
            [self._snapshot_req_for_prepare(req) for req in batch.reqs],
        )

    def note_retracted_reqs(self, reqs) -> None:
        # for req in reqs:
        #     self.simulator.process_new_request(req, self._deltas_us)
        self._enqueue_prepare_mutation(
            "note_retracted_reqs",
            (
                [self._snapshot_req_for_prepare(req) for req in reqs],
                dict(self._deltas_us),
            ),
        )

    def fairinf_force_prefill(
        self,
        req: Req,
        token_counters_by_user: Dict[str, List[int]],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional[ScheduleBatch] = None,
        decode_time_us: int = 20000,
    ) -> bool:
        del delta_fairness_deltas_microseconds, decode_time_us
        if req.rid not in self._forced_prefill_rids:
            return False
        this_users_extras = token_counters_by_user.get(req.uid, [])
        extra_sum = sum(this_users_extras)
        return self.user_is_fair_prefill(
            req.uid,
            running_batch=running_batch,
            this_user_len=len(this_users_extras),
            this_user_sum=extra_sum,
        ) and self._force_prefill_within_user_headroom(
            req,
            running_batch=running_batch,
            pending_prefill_tokens=extra_sum,
        )

    def fairinf_prioritize_force_prefill(self):
        return True

    def fairinf_force_prefill_any_waiting(
        self,
        waiting_queue: List[Req],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional[ScheduleBatch] = None,
    ) -> bool:
        del waiting_queue, delta_fairness_deltas_microseconds, running_batch
        return bool(self._forced_prefill_rids)

    def fairinf_force_decode(
        self,
        running_batch: Optional[ScheduleBatch],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        decode_time_us: int = 20000,
    ) -> Tuple[bool, Optional[int]]:
        del decode_time_us
        if running_batch is None:
            return False, self._max_safe_prefill_tokens
        if self._has_decode_deadline and (self._max_safe_prefill_tokens or 0) <= 0:
            return True, 0
        return False, self._max_safe_prefill_tokens

    def force_prefill_reservations(
        self,
        waiting_queue: List[Req],
        *,
        token_counters_by_user: Dict[str, List[int]],
        adder,
        token_to_kv_pool=None,
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        max_input_size: Optional[int] = None,
        prefix_computed: bool = False,
        max_running_requests: Optional[int] = None,
    ) -> Tuple[int, Optional[List[Req]]]:
        if (
            not self._forced_prefill_rids
            or token_to_kv_pool is None
            or self.tree_cache is None
            or running_batch is None
        ):
            return 0, None
        prioritized_waiting = []
        safe_prefill_cap = self._max_safe_prefill_tokens or 0
        used_safe_prefill_tokens = 0
        total_admission_tokens = 0
        local_token_counters_by_user = {
            user_id: list(tokens) for user_id, tokens in token_counters_by_user.items()
        }
        for req in self._forced_prefill_queue:
            if req.rid not in self._forced_prefill_rids:
                continue
            req_prefill_tokens = getattr(req, "extend_input_len", len(req.origin_input_ids))
            if (
                safe_prefill_cap > 0
                and used_safe_prefill_tokens + req_prefill_tokens > safe_prefill_cap
            ):
                break
            this_users_extras = local_token_counters_by_user.get(req.uid, [])
            extra_sum = sum(this_users_extras)
            if not self.user_is_fair_prefill(
                req.uid,
                running_batch=running_batch,
                this_user_len=len(this_users_extras),
                this_user_sum=extra_sum,
            ):
                continue
            if not self._force_prefill_within_user_headroom(
                req,
                running_batch=running_batch,
                pending_prefill_tokens=extra_sum,
            ):
                continue
            prioritized_waiting.append(req)
            used_safe_prefill_tokens += req_prefill_tokens
            total_admission_tokens += len(req.origin_input_ids) + min(
                req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
            )
            local_token_counters_by_user.setdefault(req.uid, []).append(
                req_prefill_tokens
            )
        if not prioritized_waiting:
            return 0, None

        tree_cache = self.tree_cache
        all_evicted: List[Req] = []

        if max_running_requests is not None:
            slots_needed = max(
                0,
                running_batch.batch_size()
                + len(prioritized_waiting)
                - max_running_requests,
            )
            if slots_needed > 0:
                try:
                    evicted_for_slots, _ = running_batch.retract_decode_for_slots(
                        slots_needed
                    )
                except RuntimeError as exc:
                    if "Delta fairness retraction blocked" in str(exc):
                        logger.info(
                            "Aggregate forced prefill slot reservation blocked: %s", exc
                        )
                        evicted_for_slots = []
                    else:
                        raise
                if not evicted_for_slots:
                    return 0, None
                waiting_queue.extend(evicted_for_slots)
                self.note_retracted_reqs(evicted_for_slots)
                all_evicted.extend(evicted_for_slots)

        current_capacity = (
            token_to_kv_pool.available_size() + tree_cache.evictable_size()
        )
        if current_capacity < total_admission_tokens:
            try:
                evicted_for_tokens, _ = running_batch.retract_decode(
                    total_admission_tokens
                )
            except RuntimeError as exc:
                if "Delta fairness retraction blocked" in str(exc):
                    logger.info(
                        "Aggregate forced prefill token reservation blocked: %s", exc
                    )
                    evicted_for_tokens = []
                else:
                    raise
            if not evicted_for_tokens:
                return 0, all_evicted or None
            waiting_queue.extend(evicted_for_tokens)
            self.note_retracted_reqs(evicted_for_tokens)
            all_evicted.extend(evicted_for_tokens)
            adder.expand_capacity(
                token_to_kv_pool.available_size() + tree_cache.evictable_size()
            )
            if max_input_size is not None:
                adder.rem_input_tokens = min(
                    adder.rem_input_tokens,
                    max(0, max_input_size - adder.log_input_tokens),
                )

        return total_admission_tokens, all_evicted or None

    def fairinf_overdue_decode_subset_rids(
        self,
        running_batch: Optional[ScheduleBatch],
    ) -> Optional[set[str]]:
        del running_batch
        return None

    def sorted_waiting_queue(self, waiting_queue: List[Req]):
        if not self._safe_waiting_queue:
            return waiting_queue

        prioritized = {req.rid: i for i, req in enumerate(self._safe_waiting_queue)}
        indexed = list(enumerate(waiting_queue))
        indexed.sort(
            key=lambda item: (
                0 if item[1].rid in prioritized else 1,
                prioritized.get(item[1].rid, 0),
                item[0],
            )
        )
        return [req for _, req in indexed]

    def process_waiting_queue_prefills(
        self,
        waiting_queue: List[Req],
        *,
        adder,
        token_counters_by_user: Dict[str, List[int]],
        prefix_computed: bool,
        running_batch: Optional[ScheduleBatch],
        running_batch_size: int,
        max_running_requests: int,
        available_req_slots: int,
        max_input_size: Optional[int],
    ) -> None:
        effective_max_running_requests = (
            self.max_running_requests
            if self.max_running_requests is not None
            else max_running_requests
        )
        if effective_max_running_requests is None:
            raise ValueError("max_running_requests must be set for waiting-queue processing.")
        if not self._has_delta_limit():
            return super().process_waiting_queue_prefills(
                waiting_queue,
                adder=adder,
                token_counters_by_user=token_counters_by_user,
                prefix_computed=prefix_computed,
                running_batch=running_batch,
                running_batch_size=running_batch_size,
                max_running_requests=effective_max_running_requests,
                available_req_slots=available_req_slots,
                max_input_size=max_input_size,
            )

        effective_running_limit = min(
            effective_max_running_requests,
            running_batch_size + max(0, available_req_slots),
        )
        if running_batch_size >= effective_running_limit:
            return

        if not self._safe_waiting_queue:
            return super().process_waiting_queue_prefills(
                waiting_queue,
                adder=adder,
                token_counters_by_user=token_counters_by_user,
                prefix_computed=prefix_computed,
                running_batch=running_batch,
                running_batch_size=running_batch_size,
                max_running_requests=effective_max_running_requests,
                available_req_slots=available_req_slots,
                max_input_size=max_input_size,
            )

        waiting_by_rid = {req.rid: req for req in waiting_queue}
        target_tree_cache = None if prefix_computed else self.tree_cache
        pending_prefill_by_user = {
            user_id: sum(tokens) for user_id, tokens in token_counters_by_user.items()
        }

        for prepared_req in self._safe_waiting_queue:
            req = waiting_by_rid.get(prepared_req.rid)
            if req is None:
                continue
            if max_input_size is not None and adder.log_input_tokens > max_input_size:
                break
            if max_input_size is not None:
                adder.rem_input_tokens = max_input_size - adder.log_input_tokens
            if req in adder.can_run_list:
                continue

            extra_tokens = pending_prefill_by_user.get(req.uid, 0)
            res = req.init_next_round_input(
                target_tree_cache,
                fairness_policy=self,
                fair=False,
                extra_tokens=extra_tokens,
            )
            if res == "rejected":
                logger.info(
                    "Prefill request uid=%s rid=%s rejected due to KV cache user limit.",
                    req.uid,
                    req.rid,
                )
                continue

            is_forced = req.rid in self._forced_prefill_rids
            if is_forced:
                self._ignore_global_prefill_budget = True
            try:
                add_res = adder.add_one_req(req, extra_tokens)
            finally:
                if is_forced:
                    self._ignore_global_prefill_budget = False
            if add_res == "rejected":
                logger.info(
                    "Prefill request uid=%s rid=%s rejected during admission.",
                    req.uid,
                    req.rid,
                )
                continue

            token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)
            pending_prefill_by_user[req.uid] = extra_tokens + req.extend_input_len

            if (
                not add_res
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= effective_running_limit
            ):
                break

    def _mark_violation_if_executed_after_deadline(
        self,
        req: Req,
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        tracked = self.simulator.requests.get(req.rid)
        real_event = self.simulator.most_recent_event_real.get(req.rid)
        if tracked is None or real_event is None:
            return

        now_ts = time.time() if now is None else now
        upcoming_events = tracked.earliest_events_after_real_time(real_event) or []
        matched_event = None
        for event in upcoming_events:
            if event_type == "prefill" and isinstance(event, RequestPrefillEvent):
                matched_event = event
                break
            if (
                event_type == "decode"
                and isinstance(event, RequestDecodeEvent)
                and (completion_number is None or event.completion_number == completion_number)
            ):
                matched_event = event
                break

        if matched_event is None:
            return

        deadline = matched_event.end_timestamp + self._event_delta_seconds(
            tracked, matched_event
        )
        if now_ts > deadline:
            TIMELINE_WRITER.mark_delta_violation(req.rid, req.uid, event_type=event_type)

    def finished_prefill(self, batch: ScheduleBatch) -> None:
        super().finished_prefill(batch)
        now = time.time()
        for req in batch.reqs:
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="prefill",
                now=now,
            )
        # self.simulator.finished_prefill(batch)
        # Worker state is advanced from prepare_during_gpu_execution().

    def finished_decode(self, batch: ScheduleBatch, decode_rounds: int = 1) -> None:
        super().finished_decode(batch, decode_rounds=decode_rounds)
        # needs_simulator_update = []
        # for req in batch.reqs:
        #     real_event = self.simulator.most_recent_event_real.get(req.rid)
        #     if not isinstance(real_event, RequestDecodeEvent) or (
        #         real_event.completion_number < len(req.output_ids)
        #     ):
        #         needs_simulator_update.append(req)
        # if needs_simulator_update:
        #     self.simulator.finished_decode(
        #         SimpleNamespace(reqs=needs_simulator_update),
        #         decode_rounds=decode_rounds,
        #     )
        # Worker state is advanced from prepare_during_gpu_execution().

    def mark_request_finished(self, req: Req) -> None:
        super().mark_request_finished(req)
        self.simulator.mark_request_finished(req)
        self._enqueue_prepare_mutation(
            "mark_request_finished",
            (self._snapshot_req_for_prepare(req), self._current_pass_id),
        )
