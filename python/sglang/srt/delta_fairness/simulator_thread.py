from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from .doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
)

if TYPE_CHECKING:
    from .doc_policy import DocPolicy


logger = logging.getLogger(__name__)
DOC_POLICY_WORKER_TRACE_ENABLED = False


class _LenOnlySeq:
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


class _PrepareReq:
    def __init__(
        self,
        *,
        uid: str,
        rid: str,
        prompt_len: int,
        output_len: int,
        fill_len: Optional[int],
        prefix_len: int,
        extend_input_len: int,
        waiting_time_in_decodes: int,
        first_time_in_waiting_queue: bool,
        max_new_tokens: int,
    ) -> None:
        self.uid = uid
        self.rid = rid
        self.origin_input_ids = _LenOnlySeq(prompt_len)
        self.output_ids = _LenOnlySeq(output_len)
        self.fill_ids = None if fill_len is None else _LenOnlySeq(fill_len)
        self.extend_input_len = int(extend_input_len)
        self.prefix_indices = _LenOnlySeq(prefix_len)
        self.waiting_time_in_decodes = int(waiting_time_in_decodes)
        self.first_time_in_waiting_queue = bool(first_time_in_waiting_queue)
        self.sampling_params = SimpleNamespace(max_new_tokens=int(max_new_tokens))

    def get_estimated_prefill_impact(self) -> int:
        return len(self.origin_input_ids) + 2


def _advance_req_output(req: _PrepareReq, steps: int) -> _PrepareReq:
    return _PrepareReq(
        uid=req.uid,
        rid=req.rid,
        prompt_len=len(req.origin_input_ids),
        output_len=len(req.output_ids) + max(int(steps), 0),
        fill_len=None if req.fill_ids is None else len(req.fill_ids),
        prefix_len=len(req.prefix_indices),
        extend_input_len=int(req.extend_input_len),
        waiting_time_in_decodes=int(req.waiting_time_in_decodes),
        first_time_in_waiting_queue=bool(req.first_time_in_waiting_queue),
        max_new_tokens=int(req.sampling_params.max_new_tokens),
    )


@dataclass(frozen=True)
class _PreparedSnapshot:
    task_seq: int
    mutation_seq: int
    waiting_sig: Tuple[str, ...]
    running_sig: Tuple[str, ...]
    deadline_queue: Tuple[object, ...]
    waiting_prefill_deadlines: MappingProxyType
    safe_waiting_queue: Tuple[Req, ...]
    safe_waiting_rids: frozenset[str]
    forced_prefill_queue: Tuple[Req, ...]
    forced_prefill_rids: frozenset[str]
    max_safe_prefill_tokens: Optional[int]
    has_fair_waiting: bool
    has_decode_deadline: bool
    earliest_decode_start_deadline: Optional[float]
    earliest_decode_rid: Optional[str]
    earliest_decode_uid: Optional[str]
    safe_prefix_now: Optional[float]
    breakdown_items: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class _FrozenPrepareCacheState:
    total_user_tokens: Dict[str, int]
    evictable_user_tokens: Dict[str, int]
    fairinf_max_per_user: Optional[int]
    unevictable_limit: Optional[int]


@dataclass(frozen=True)
class _FrozenPrepareInputs:
    deltas_us: Dict[str, int]
    no_retraction_cap: Optional[int]
    new_token_ratio: float
    fairinf_n: int


class _DocPolicyPrepareWorker:
    def __init__(
        self,
        owner: "DocPolicy",
        *,
        isolated_kv_tokens_per_user: Optional[int],
        fairinf_n: int,
        min_new_token_ratio: float,
    ) -> None:
        self._owner = owner
        self._mutation_queue: Deque[tuple] = deque()
        self._task_queue: Deque[tuple] = deque()
        self._simulator = AlternateHistorySimulator(
            max_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=fairinf_n,
            min_new_token_ratio=min_new_token_ratio,
            enable_timeline_logging=True,
        )
        self._published_snapshot: Optional[_PreparedSnapshot] = None
        self._published_snapshot_lock = threading.Lock()
        self._pause_lock = threading.Lock()
        self._pause_cond = threading.Condition(self._pause_lock)
        self._pause_depth = 0
        self._active_work = 0
        self._thread_exception: Optional[BaseException] = None
        self._worker_stop = False
        self._task_seq = 0
        self._mutation_seq = 0
        self._applied_mutation_seq = 0
        self._inflight_observe_seq: Optional[int] = None
        self._pending_mutation_queue_backpressure_wait_ms = 0.0
        self._pending_prepare_task_queue_backpressure_wait_ms = 0.0
        self._pending_mutation_queue_drain_wait_ms = 0.0
        self._pending_duplicate_state_wait_ms = 0.0
        self._queue_csv_path = os.path.join(os.getcwd(), "doc_policy_prepare_queue.csv")
        self._queue_csv_header_written = False
        self._trace_csv_path = os.path.join(os.getcwd(), "doc_policy_worker_trace.csv")
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="doc-policy-prepare",
            daemon=True,
        )
        self._worker_thread.start()

    def _trace(self, action: str, **fields) -> None:
        if not DOC_POLICY_WORKER_TRACE_ENABLED:
            return
        keys = [
            "ts",
            "action",
            "task_seq",
            "pass_id",
            "event_type",
            "prepare_pass_state",
            "selected_count",
            "decode_steps",
            "snapshot_seq",
            "earliest_rid",
            "earliest_completion",
            "earliest_deadline",
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

    def snapshot_req(self, req: Req) -> dict:
        return req.to_prepare_snapshot()

    def snapshot_batch(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        if batch is None:
            return None
        return SimpleNamespace(reqs=[self.snapshot_req(req) for req in batch.reqs])

    def _restore_req(self, req_data: dict) -> _PrepareReq:
        return _PrepareReq(
            uid=req_data["uid"],
            rid=req_data["rid"],
            prompt_len=int(req_data.get("prompt_len", 0)),
            output_len=int(req_data.get("output_len", 0)),
            fill_len=req_data.get("fill_len"),
            prefix_len=int(req_data.get("prefix_len", 0)),
            extend_input_len=int(req_data.get("extend_input_len", 0)),
            waiting_time_in_decodes=int(req_data.get("waiting_time_in_decodes", 0)),
            first_time_in_waiting_queue=bool(
                req_data.get("first_time_in_waiting_queue", False)
            ),
            max_new_tokens=int(req_data.get("max_new_tokens", 0) or 0),
        )

    def _restore_batch(
        self, batch_data: Optional[SimpleNamespace]
    ) -> Optional[SimpleNamespace]:
        if batch_data is None:
            return None
        return SimpleNamespace(reqs=[self._restore_req(req) for req in batch_data.reqs])

    def raise_exception_if_any(self) -> None:
        if self._thread_exception is not None:
            raise RuntimeError("doc policy prepare worker failed") from self._thread_exception

    def latest_snapshot(self) -> Optional[_PreparedSnapshot]:
        with self._published_snapshot_lock:
            return self._published_snapshot

    def _log_queue_backpressure(
        self, *, queue_name: str, queue_len: int, wait_ms: float
    ) -> None:
        if queue_name == "mutation":
            self._pending_mutation_queue_backpressure_wait_ms += wait_ms
        else:
            self._pending_prepare_task_queue_backpressure_wait_ms += wait_ms
        line = f"{time.time()},{queue_name},{queue_len},{wait_ms}\n"
        if not self._queue_csv_header_written:
            with open(self._queue_csv_path, "a") as f:
                if f.tell() == 0:
                    f.write("timestamp,queue_name,queue_len,wait_ms\n")
                f.write(line)
            self._queue_csv_header_written = True
            return
        with open(self._queue_csv_path, "a") as f:
            f.write(line)

    def _maybe_wait_for_queue_capacity(
        self, queue_ref: Deque[tuple], *, queue_name: str
    ) -> None:
        if len(queue_ref) < 100:
            return
        wait_start = time.perf_counter()
        while len(queue_ref) >= 100:
            time.sleep(0.001)
        self._log_queue_backpressure(
            queue_name=queue_name,
            queue_len=len(queue_ref),
            wait_ms=(time.perf_counter() - wait_start) * 1000.0,
        )

    def enqueue_mutation(self, kind: str, payload) -> None:
        self._maybe_wait_for_queue_capacity(self._mutation_queue, queue_name="mutation")
        self._mutation_seq += 1
        self._mutation_queue.append((kind, payload))

    def enqueue_task(self, task: tuple) -> int:
        self._maybe_wait_for_queue_capacity(self._task_queue, queue_name="prepare")
        self._task_seq += 1
        seq = self._task_seq
        self._task_queue.append((seq, task))
        return seq

    def enqueue_priority_task(self, task: tuple) -> int:
        self._maybe_wait_for_queue_capacity(self._task_queue, queue_name="prepare")
        self._task_seq += 1
        seq = self._task_seq
        self._task_queue.appendleft((seq, task))
        return seq

    def wait_for_mutation_queue_below_limit(self, limit: int = 100) -> None:
        if len(self._mutation_queue) < limit:
            return
        wait_start = time.perf_counter()
        while len(self._mutation_queue) >= limit:
            self.raise_exception_if_any()
            time.sleep(0.001)
        self._pending_mutation_queue_drain_wait_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0

    @property
    def mutation_seq(self) -> int:
        return self._mutation_seq

    def request_live_snapshot(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        last_consumed_snapshot_seq: int,
        ) -> Tuple[int, bool]:
        task = (
            "observe",
            self.snapshot_batch(running_batch),
            [self.snapshot_req(req) for req in waiting_queue],
            None,
            None,
            True,
            0,
            None,
            False,
            float(getattr(self._owner, "_pass_new_token_ratio", 0.0)),
            self._owner._freeze_prepare_cache_state(),
            self._owner._freeze_prepare_inputs(
                new_token_ratio=float(getattr(self._owner, "_pass_new_token_ratio", 0.0))
            ),
            0,
            self._mutation_seq,
        )
        return self.enqueue_task(task), False

    def wait_for_snapshot(
        self,
        *,
        min_task_seq: int,
        record_duplicate_wait: bool = False,
        timeout_s: Optional[float] = 0.050,
    ) -> bool:
        if min_task_seq <= 0:
            return False
        wait_start = time.perf_counter()
        while True:
            self.raise_exception_if_any()
            snapshot = self.latest_snapshot()
            if snapshot is not None and snapshot.task_seq >= min_task_seq:
                if record_duplicate_wait:
                    self._pending_duplicate_state_wait_ms += (
                        time.perf_counter() - wait_start
                    ) * 1000.0
                return True
            if timeout_s is not None and (time.perf_counter() - wait_start) >= timeout_s:
                break
            time.sleep(0.001)
        if record_duplicate_wait:
            self._pending_duplicate_state_wait_ms += (
                time.perf_counter() - wait_start
            ) * 1000.0
        return False

    def consume_wait_metrics(self) -> Dict[str, float]:
        metrics = {
            "prepare_mutation_queue_backpressure_wait_ms": self._pending_mutation_queue_backpressure_wait_ms,
            "prepare_task_queue_backpressure_wait_ms": self._pending_prepare_task_queue_backpressure_wait_ms,
            "prepare_mutation_queue_drain_wait_ms": self._pending_mutation_queue_drain_wait_ms,
            "prepare_duplicate_state_wait_ms": self._pending_duplicate_state_wait_ms,
        }
        self._pending_mutation_queue_backpressure_wait_ms = 0.0
        self._pending_prepare_task_queue_backpressure_wait_ms = 0.0
        self._pending_mutation_queue_drain_wait_ms = 0.0
        self._pending_duplicate_state_wait_ms = 0.0
        return metrics

    class _PauseContext:
        def __init__(self, worker: "_DocPolicyPrepareWorker") -> None:
            self._worker = worker

        def __enter__(self):
            self._worker._pause_for_critical_section()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._worker._resume_after_critical_section()
            return False

    def pause(self):
        return self._PauseContext(self)

    def _pause_for_critical_section(self) -> None:
        with self._pause_cond:
            self._pause_depth += 1
            while self._active_work > 0:
                self._pause_cond.wait(timeout=0.001)

    def _resume_after_critical_section(self) -> None:
        with self._pause_cond:
            if self._pause_depth > 0:
                self._pause_depth -= 1
            self._pause_cond.notify_all()

    def _wait_until_resumed(self) -> None:
        with self._pause_cond:
            while self._pause_depth > 0 and not self._worker_stop:
                self._pause_cond.wait(timeout=0.001)

    def _mark_work_start(self) -> None:
        with self._pause_cond:
            while self._pause_depth > 0 and not self._worker_stop:
                self._pause_cond.wait(timeout=0.001)
            self._active_work += 1

    def _mark_work_done(self) -> None:
        with self._pause_cond:
            if self._active_work > 0:
                self._active_work -= 1
            self._pause_cond.notify_all()

    def _publish_snapshot(self, snapshot: _PreparedSnapshot) -> None:
        with self._published_snapshot_lock:
            self._published_snapshot = snapshot
            self._owner._simulator_rebuild_prepared = True
            self._owner._last_prepare_breakdown_ms = dict(snapshot.breakdown_items)
        earliest_decode = next(
            (c for c in snapshot.deadline_queue if getattr(c, "event_type", "") == "decode"),
            None,
        )
        self._trace(
            "publish_snapshot",
            snapshot_seq=snapshot.task_seq,
            earliest_rid="" if earliest_decode is None else earliest_decode.req.rid,
            earliest_completion=""
            if earliest_decode is None
            else getattr(getattr(earliest_decode, "event", None), "completion_number", ""),
            earliest_deadline=snapshot.earliest_decode_start_deadline,
        )

    def _apply_task_overlay(
        self,
        *,
        event_type: str,
        running_batch,
        waiting_queue,
        scheduled_batch,
        selected_rids,
        decode_steps: int,
        decode_steps_by_rid,
        output_ids_already_applied: bool,
        frozen_inputs: _FrozenPrepareInputs,
        pass_id: int,
    ) -> None:
        if (
            event_type == "decode"
            and running_batch is not None
            and (decode_steps > 0 or decode_steps_by_rid)
        ):
            if selected_rids is None:
                selected_set = {req.rid for req in running_batch.reqs}
            else:
                selected_set = set(selected_rids)
            effective_reqs = []
            max_steps = 0
            for req in running_batch.reqs:
                if req.rid not in selected_set:
                    continue
                steps = (
                    int((decode_steps_by_rid or {}).get(req.rid, 0))
                    if decode_steps_by_rid is not None
                    else int(decode_steps)
                )
                if steps <= 0:
                    continue
                max_steps = max(max_steps, steps)
                effective_reqs.append(
                    req if output_ids_already_applied else _advance_req_output(req, steps)
                )
            if effective_reqs:
                self._trace(
                    "apply_overlay_decode",
                    pass_id=pass_id,
                    event_type=event_type,
                    selected_count=len(effective_reqs),
                    decode_steps=max_steps,
                )
                effective_batch = SimpleNamespace(reqs=effective_reqs)
                self._owner._apply_logical_decode_updates(
                    self._simulator,
                    effective_batch,
                    selected_rids=None,
                    decode_steps=max_steps,
                    decode_steps_by_rid=decode_steps_by_rid,
                    output_ids_already_applied=output_ids_already_applied,
                )
            return
        if event_type == "prefill":
            if scheduled_batch is not None:
                scheduled_rids = {req.rid for req in scheduled_batch.reqs}
                predicted_waiting = [
                    req for req in waiting_queue if req.rid not in scheduled_rids
                ]
                predicted_running = self._owner._predicted_running_batch(
                    running_batch, scheduled_batch
                )
            else:
                predicted_waiting = list(waiting_queue)
                predicted_running = running_batch
            self._simulator.get_live_users(
                predicted_running,
                predicted_waiting,
                deltas_in_microseconds=frozen_inputs.deltas_us,
            )
            if scheduled_batch is not None:
                self._trace(
                    "apply_overlay_prefill",
                    pass_id=pass_id,
                    event_type=event_type,
                    selected_count=len(scheduled_batch.reqs),
                    decode_steps=0,
                )
                self._simulator.finished_prefill(scheduled_batch)
            return

    # def _reconcile_live_progress(
    #     self,
    #     *,
    #     running_batch,
    # ) -> None:
    #     if running_batch is None:
    #         return
    #     for req in running_batch.reqs:
    #         self._simulator.sync_request_progress_from_live(req)

    def _apply_mutation(self, kind: str, payload) -> None:
        simulator = self._simulator
        if kind == "process_new_request":
            req, deltas_us, *rest = payload
            arrival_ts = rest[0] if rest else None
            simulator.process_new_request(
                self._restore_req(req), deltas_us, arrival_timestamp=arrival_ts
            )
            return
        if kind == "note_retracted_reqs":
            reqs, deltas_us = payload
            for req in reqs:
                simulator.process_new_request(self._restore_req(req), deltas_us)
            return
        if kind == "finished_prefill":
            return
        if kind == "finished_decode":
            return
        if kind == "logical_decode_update":
            (
                running_batch,
            ) = payload
            restored_running = self._restore_batch(running_batch)
            if restored_running is not None:
 
                
                # the main thread already added the fake decode events into the req timelines
                
                for user_timeline in self._simulator.users.values():
                    user_timeline.rebuild_from_real_state(
                        self._simulator.most_recent_event_real
                    )

                # self._owner._apply_logical_decode_updates(
                #     simulator,
                #     restored_running,
                #     selected_rids=None if selected_rids is None else set(selected_rids),
                #     decode_steps=int(decode_steps),
                #     decode_steps_by_rid=None
                #     if decode_steps_by_rid is None
                #     else dict(decode_steps_by_rid),
                #     output_ids_already_applied=bool(output_ids_already_applied),
                # )
                # self._latest_prepare_pass_seen = max(
                #     self._latest_prepare_pass_seen, int(pass_id)
                # )
            return
        if kind == "mark_request_finished":
            req, pass_id = payload
            simulator.mark_request_finished(self._restore_req(req))
            return
        if kind == "note_scheduled_prefill_batch":
            return
        raise ValueError(f"unknown prepare mutation kind: {kind}")

    def _worker_loop(self) -> None:
        torch.set_grad_enabled(False)
        while not self._worker_stop:
            if self._thread_exception is not None:
                return
            if self._mutation_queue:
                self._mark_work_start()
                
                # for now take it out of the try
                with torch.inference_mode():
                        kind, payload = self._mutation_queue.popleft()
                        self._apply_mutation(kind, payload)
                try:
                    pass
                except BaseException as exc:
                    self._thread_exception = exc
                    logger.exception("prepare worker mutation failed")
                    self._mark_work_done()
                    return
                self._mark_work_done()
                continue
            if not self._task_queue:
                self._wait_until_resumed()
                time.sleep(0.001)
                continue
            self._mark_work_start()
            event_type = None
            try:
                with torch.inference_mode():
                    task_seq, task = self._task_queue.popleft()
                    (
                        event_type,
                        running_batch,
                        waiting_queue,
                        scheduled_batch,
                        selected_rids,
                        prepare_pass_state,
                        decode_steps,
                        decode_steps_by_rid,
                        output_ids_already_applied,
                        _new_token_ratio,
                        frozen_cache_state,
                        frozen_inputs,
                        pass_id,
                        _requested_mutation_seq,
                    ) = task
                    self._trace(
                        "task_start",
                        task_seq=task_seq,
                        pass_id=pass_id,
                        event_type=event_type,
                        prepare_pass_state=int(bool(prepare_pass_state)),
                        selected_count=0 if selected_rids is None else len(selected_rids),
                        decode_steps=decode_steps if decode_steps_by_rid is None else sum(decode_steps_by_rid.values()),
                    )
                    running_batch = self._restore_batch(running_batch)
                    scheduled_batch = self._restore_batch(scheduled_batch)
                    waiting_queue = [self._restore_req(req) for req in waiting_queue]
                    if event_type == "observe":
                        self._inflight_observe_seq = task_seq
                    while self._mutation_queue:
                        kind, payload = self._mutation_queue.popleft()
                        self._apply_mutation(kind, payload)
                        self._applied_mutation_seq += 1
                    if event_type != "observe":
                        self._apply_task_overlay(
                            event_type=event_type,
                            running_batch=running_batch,
                            waiting_queue=waiting_queue,
                            scheduled_batch=scheduled_batch,
                            selected_rids=selected_rids,
                            decode_steps=decode_steps,
                            decode_steps_by_rid=decode_steps_by_rid,
                            output_ids_already_applied=output_ids_already_applied,
                            frozen_inputs=frozen_inputs,
                            pass_id=int(pass_id),
                        )
                    #self._reconcile_live_progress(running_batch=running_batch)
                    if not prepare_pass_state:
                        continue
                    # frozen_simulator = self._simulator
                    # self._simulator = frozen_simulator.clone(
                    #     enable_timeline_logging=True
                    # )
                    if event_type == "prefill" and scheduled_batch is not None:
                        scheduled_rids = {req.rid for req in scheduled_batch.reqs}
                        predicted_waiting = [
                            req for req in waiting_queue if req.rid not in scheduled_rids
                        ]
                        predicted_running = self._owner._predicted_running_batch(
                            running_batch, scheduled_batch
                        )
                    else:
                        predicted_waiting = list(waiting_queue)
                        predicted_running = running_batch
                    breakdown: Dict[str, float] = {
                        "logical_event_update_ms": 0.0,
                        "rebuild_from_real_state_ms": 0.0,
                        "prepare_during_gpu_execution_total_ms": 0.0,
                    }
                    snapshot = self._owner._build_prepare_snapshot(
                        self._simulator,
                        predicted_waiting,
                        predicted_running,
                        task_seq=task_seq,
                        mutation_seq=self._applied_mutation_seq,
                        breakdown=breakdown,
                        frozen_cache_state=frozen_cache_state,
                        frozen_inputs=frozen_inputs,
                    )
                    snapshot = _PreparedSnapshot(
                        task_seq=snapshot.task_seq,
                        mutation_seq=snapshot.mutation_seq,
                        waiting_sig=snapshot.waiting_sig,
                        running_sig=snapshot.running_sig,
                        deadline_queue=snapshot.deadline_queue,
                        waiting_prefill_deadlines=snapshot.waiting_prefill_deadlines,
                        safe_waiting_queue=snapshot.safe_waiting_queue,
                        safe_waiting_rids=snapshot.safe_waiting_rids,
                        forced_prefill_queue=snapshot.forced_prefill_queue,
                        forced_prefill_rids=snapshot.forced_prefill_rids,
                        max_safe_prefill_tokens=snapshot.max_safe_prefill_tokens,
                        has_fair_waiting=snapshot.has_fair_waiting,
                        has_decode_deadline=snapshot.has_decode_deadline,
                        earliest_decode_start_deadline=snapshot.earliest_decode_start_deadline,
                        earliest_decode_rid=snapshot.earliest_decode_rid,
                        earliest_decode_uid=snapshot.earliest_decode_uid,
                        safe_prefix_now=snapshot.safe_prefix_now,
                        breakdown_items=snapshot.breakdown_items,
                    )
                    self._publish_snapshot(snapshot)
            except BaseException as exc:
                self._thread_exception = exc
                logger.exception("prepare worker task failed")
                return
            finally:
                if event_type == "observe":
                    self._inflight_observe_seq = None
                self._mark_work_done()
