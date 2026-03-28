from __future__ import annotations

"""Design.md policy implementation."""

import logging
import os
import threading
import time
from collections import deque
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Deque, Dict, List, Optional, Tuple

from sglang.global_config import global_config
from sglang.srt.request_timeline import TIMELINE_WRITER
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from .delta_fairness_policy import DeltaFairnessPolicy
from .doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestEvent,
    RequestPrefillEvent,
    TrackedRequest,
)
from .time_estimation import (
    isolated_decode_time_estimation,
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

logger = logging.getLogger(__name__)


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
    safe_prefix_now: Optional[float]
    breakdown_items: Tuple[Tuple[str, float], ...]


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
            enable_timeline_logging=False,
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
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="doc-policy-prepare",
            daemon=True,
        )
        self._worker_thread.start()

    def snapshot_req(self, req: Req) -> Req:
        cloned = copy(req)
        cloned.origin_input_ids = list(req.origin_input_ids)
        cloned.output_ids = list(req.output_ids)
        cloned.fill_ids = None if req.fill_ids is None else list(req.fill_ids)
        return cloned

    def snapshot_batch(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        if batch is None:
            return None
        return SimpleNamespace(reqs=[self.snapshot_req(req) for req in batch.reqs])

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
        del waiting_queue, running_batch
        snapshot = self.latest_snapshot()
        if snapshot is not None and snapshot.task_seq > last_consumed_snapshot_seq:
            return snapshot.task_seq, True
        return last_consumed_snapshot_seq, False

    def wait_for_snapshot(
        self,
        *,
        min_task_seq: int,
        record_duplicate_wait: bool = False,
        timeout_s: float = 0.050,
    ) -> bool:
        if min_task_seq <= 0:
            return False
        deadline = time.perf_counter() + timeout_s
        wait_start = time.perf_counter()
        while time.perf_counter() < deadline:
            self.raise_exception_if_any()
            snapshot = self.latest_snapshot()
            if snapshot is not None and snapshot.task_seq >= min_task_seq:
                if record_duplicate_wait:
                    self._pending_duplicate_state_wait_ms += (
                        time.perf_counter() - wait_start
                    ) * 1000.0
                return True
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

    def _apply_mutation(self, kind: str, payload) -> None:
        simulator = self._simulator
        if kind == "process_new_request":
            simulator.process_new_request(payload, self._owner._deltas_us)
            return
        if kind == "note_retracted_reqs":
            for req in payload:
                simulator.process_new_request(req, self._owner._deltas_us)
            return
        if kind == "finished_prefill":
            simulator.finished_prefill(SimpleNamespace(reqs=payload))
            return
        if kind == "finished_decode":
            reqs, decode_rounds = payload
            simulator.finished_decode(SimpleNamespace(reqs=reqs), decode_rounds=decode_rounds)
            return
        if kind == "mark_request_finished":
            simulator.mark_request_finished(payload)
            return
        if kind == "note_scheduled_prefill_batch":
            return
        raise ValueError(f"unknown prepare mutation kind: {kind}")

    def _worker_loop(self) -> None:
        while not self._worker_stop:
            if self._thread_exception is not None:
                return
            if self._mutation_queue:
                self._mark_work_start()
                try:
                    kind, payload = self._mutation_queue.popleft()
                    self._apply_mutation(kind, payload)
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
                task_seq, task = self._task_queue.popleft()
                (
                    event_type,
                    running_batch,
                    waiting_queue,
                    scheduled_batch,
                    selected_rids,
                    prepare_pass_state,
                    decode_steps,
                    _new_token_ratio,
                    _requested_mutation_seq,
                ) = task
                if event_type == "observe":
                    self._inflight_observe_seq = task_seq
                while self._mutation_queue:
                    kind, payload = self._mutation_queue.popleft()
                    self._apply_mutation(kind, payload)
                    self._applied_mutation_seq += 1
                if event_type == "decode" and running_batch is not None and decode_steps > 0:
                    self._owner._apply_logical_decode_updates(
                        self._simulator,
                        running_batch,
                        selected_rids=selected_rids,
                        decode_steps=decode_steps,
                    )
                if not prepare_pass_state:
                    continue
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
        self._last_prepare_breakdown_ms: Dict[str, float] = {}
        self._last_pass_state_source = "init"
        self._pending_new_requests: List[Req] = []
        self._pending_finished_rids: set[str] = set()
        self._pending_decoded_reqs: Dict[str, Req] = {}
        self._pending_finished_prefill_reqs: Dict[str, Req] = {}
        self._pending_scheduled_prefill_reqs: Dict[str, Req] = {}
        self._prefill_no_retraction_token_cap: Optional[int] = None
        self._last_consumed_prepare_snapshot_seq = 0
        self._prepare_worker = _DocPolicyPrepareWorker(
            self,
            isolated_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
        )

    def _has_pending_mutations(self) -> bool:
        return bool(
            self._pending_new_requests
            or self._pending_finished_rids
            or self._pending_decoded_reqs
            or self._pending_finished_prefill_reqs
            or self._pending_scheduled_prefill_reqs
        )

    def _clear_pending_mutations(self) -> None:
        self._pending_new_requests = []
        self._pending_finished_rids.clear()
        self._pending_decoded_reqs.clear()
        self._pending_finished_prefill_reqs.clear()
        self._pending_scheduled_prefill_reqs.clear()

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
        timeout_s: float = 0.050,
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
        no_retraction_cap = self._no_retraction_prefill_token_cap(running_batch)
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
                    this_user_len = pending_prefill_len_by_user.get(req.uid, 0)
                    is_fair = self.user_is_fair_prefill(
                        req.uid,
                        running_batch=running_batch,
                        this_user_len=this_user_len,
                        this_user_sum=req.get_estimated_prefill_impact() + this_user_sum,
                    )
                    fair_user_by_uid[req.uid] = is_fair
                if not is_fair:
                    continue
                if not self._force_prefill_within_user_headroom(
                    req,
                    running_batch=running_batch,
                    pending_prefill_tokens=pending_prefill_sum_by_user.get(req.uid, 0),
                ):
                    continue
            candidate_batch = selected_batch + [req]
            pooled_prefill_s = self._pooled_prefill_seconds(candidate_batch)
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
        self.simulator.sync_live_user_tracking(
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
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
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

    def _queue_sig_excluding_pending_new(self, reqs: List[Req]) -> Tuple[str, ...]:
        pending_new_rids = {req.rid for req in self._pending_new_requests}
        if not pending_new_rids:
            return self._queue_sig(reqs)
        return tuple(req.rid for req in reqs if req.rid not in pending_new_rids)

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
        target_simulator.sync_live_user_tracking(
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
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
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
    ) -> _PreparedSnapshot:
        target_breakdown: Dict[str, float] = {} if breakdown is None else breakdown
        build_start = time.perf_counter()
        simulator.sync_live_user_tracking(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=self._deltas_us,
        )
        after_sync = time.perf_counter()
        deadline_waiting_queue = self._prefill_deadline_waiting_queue(waiting_queue, running_batch)
        deadline_result = simulator.build_deadline_candidates(
            deadline_waiting_queue,
            running_batch,
            include_ordered_waiting_queue=True,
            req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                req.uid,
                running_batch=rb,
                this_user_sum=req.get_estimated_prefill_impact(),
            ),
            req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                req, running_batch=rb
            ),
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

    def _merge_pending_pass_state_mutations(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> None:
        waiting_rids = {req.rid for req in waiting_queue}
        running_rids = (
            set() if running_batch is None else {req.rid for req in running_batch.reqs}
        )
        finished_rids = set(self._pending_finished_rids)
        scheduled_prefill_rids = set(self._pending_scheduled_prefill_reqs)
        self._deadline_queue = [
            candidate
            for candidate in self._deadline_queue
            if (
                candidate.event_type != "prefill"
                or candidate.req.rid in waiting_rids
            )
            and (
                candidate.event_type != "decode"
                or candidate.req.rid in running_rids
            )
            if candidate.req.rid not in finished_rids
            and candidate.req.rid not in scheduled_prefill_rids
            and not (
                candidate.event_type == "decode"
                and candidate.req.rid in self._pending_decoded_reqs
            )
            and not (
                candidate.event_type == "decode"
                and candidate.req.rid in self._pending_finished_prefill_reqs
            )
        ]
        self._waiting_prefill_start_deadline_by_rid = {
            rid: deadline
            for rid, deadline in self._waiting_prefill_start_deadline_by_rid.items()
            if rid in waiting_rids
            and rid not in finished_rids
            and rid not in scheduled_prefill_rids
        }
        if self._pending_new_requests:
            deadline_result = self.simulator.build_deadline_candidates(
                self._pending_new_requests,
                running_batch,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
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
                new_candidates, new_waiting_deadlines, _ = deadline_result
            else:
                new_candidates, new_waiting_deadlines = deadline_result
            self._deadline_queue.extend(new_candidates)
            self._waiting_prefill_start_deadline_by_rid.update(new_waiting_deadlines)
        for req in self._pending_finished_prefill_reqs.values():
            candidate = self._decode_candidate_for_req(req, running_batch)
            if candidate is not None:
                self._deadline_queue.append(candidate)
        for req in self._pending_decoded_reqs.values():
            candidate = self._decode_candidate_for_req(req, running_batch)
            if candidate is not None:
                self._deadline_queue.append(candidate)
        latest_decode = {}
        others = []
        for candidate in self._deadline_queue:
            if candidate.event_type != "decode":
                others.append(candidate)
                continue
            prev = latest_decode.get(candidate.req.rid)
            if prev is None or candidate.event.completion_number >= prev.event.completion_number:
                latest_decode[candidate.req.rid] = candidate
        self._deadline_queue = others + list(latest_decode.values())
        self._deadline_queue.sort(
            key=lambda candidate: (
                candidate.start_deadline,
                0 if candidate.event_type == "decode" else 1,
                candidate.deadline,
            )
        )
        self._recompute_safe_prefix(waiting_queue, running_batch)
        self._clear_pending_mutations()

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
            self.simulator.sync_live_user_tracking(
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
                    req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                        req, running_batch=rb
                    ),
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
    ) -> bool:
        snapshot = self._prepare_worker.latest_snapshot()
        if snapshot is None:
            return False
        if snapshot.task_seq <= self._last_consumed_prepare_snapshot_seq:
            return False
        self._deadline_queue = snapshot.deadline_queue
        self._waiting_prefill_start_deadline_by_rid = snapshot.waiting_prefill_deadlines
        self._safe_waiting_queue = snapshot.safe_waiting_queue
        self._safe_waiting_rids = snapshot.safe_waiting_rids
        self._forced_prefill_queue = snapshot.forced_prefill_queue
        self._forced_prefill_rids = snapshot.forced_prefill_rids
        self._max_safe_prefill_tokens = snapshot.max_safe_prefill_tokens
        self._has_fair_waiting = snapshot.has_fair_waiting
        self._has_decode_deadline = snapshot.has_decode_deadline
        self._earliest_decode_start_deadline = snapshot.earliest_decode_start_deadline
        self._safe_prefix_now = snapshot.safe_prefix_now
        self._simulator_rebuild_prepared = False
        self._last_consumed_prepare_snapshot_seq = snapshot.task_seq
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
            and not self._has_pending_mutations()
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
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "hot_path_consume_prepared"
            return
        task_seq, duplicate_wait = self._request_live_prepare_snapshot(
            waiting_queue, running_batch
        )
        if self._wait_for_matching_prepare_snapshot(
            min_task_seq=task_seq,
            record_duplicate_wait=duplicate_wait,
        ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "hot_path_wait_prepare"

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
        elif (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
            and not self._has_pending_mutations()
        ):
            self._last_pass_state_source = "start_of_pass_current_reuse"
        else:
            task_seq, duplicate_wait = self._request_live_prepare_snapshot(
                waiting_queue, running_batch
            )
            if self._wait_for_matching_prepare_snapshot(
                min_task_seq=task_seq,
                record_duplicate_wait=duplicate_wait,
            ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
                self._last_pass_state_source = "start_of_pass_wait_prepare"
            else:
                self._last_pass_state_source = "start_of_pass_no_prepared"

        after_simulator = time.perf_counter()
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig
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
    ) -> None:
        chosen = selected_rids
        for req in running_batch.reqs:
            if chosen is not None and req.rid not in chosen:
                continue
            tracked = simulator.requests.get(req.rid)
            if tracked is None:
                continue
            final_logical_ts = None
            previous_logical_ts = self._logical_event_timestamp(req, simulator=simulator)
            most_recent_event = tracked.most_recent_event()
            if most_recent_event is not None:
                previous_logical_ts = max(
                    previous_logical_ts, float(most_recent_event.end_timestamp)
                )
            if tracked.alternate_history_timeline.anticipated_future_events:
                previous_logical_ts = max(
                    previous_logical_ts,
                    max(
                        float(event.end_timestamp)
                        for event in tracked.alternate_history_timeline.anticipated_future_events
                    ),
                )
            start_completion_number = len(req.output_ids) + 1
            for completion_number in range(
                start_completion_number, start_completion_number + decode_steps
            ):
                logical_ts = self._logical_next_event_timestamp(
                    req,
                    event_type="decode",
                    completion_number=completion_number,
                    simulator=simulator,
                )
                if logical_ts <= previous_logical_ts:
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
            next_completion_number = start_completion_number + decode_steps
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
        new_token_ratio: float = 0.0,
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        phase_start = time.perf_counter()
        if event_type == "decode" and running_batch is not None and decode_steps > 0:
            self._apply_logical_decode_updates(
                self.simulator,
                running_batch,
                selected_rids=selected_rids,
                decode_steps=decode_steps,
            )
            chosen = selected_rids
            for req in running_batch.reqs:
                if chosen is None or req.rid in chosen:
                    self._pending_decoded_reqs[req.rid] = req
        running_snapshot = self._snapshot_batch_for_prepare(running_batch)
        waiting_snapshot = [
            self._snapshot_req_for_prepare(req) for req in waiting_queue
        ]
        scheduled_snapshot = self._snapshot_batch_for_prepare(scheduled_batch)
        self._enqueue_prepare_task(
            (
                event_type,
                running_snapshot,
                waiting_snapshot,
                scheduled_snapshot,
                None if selected_rids is None else set(selected_rids),
                prepare_pass_state,
                decode_steps,
                new_token_ratio,
                self._prepare_worker.mutation_seq,
            )
        )
        elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        self._last_prepare_breakdown_ms = {
            "sync_live_user_tracking_ms": 0.0,
            "logical_event_update_ms": 0.0,
            "rebuild_from_real_state_ms": 0.0,
            "build_deadline_candidates_ms": 0.0,
            "prepare_during_gpu_execution_total_ms": elapsed_ms,
        }

    def process_new_request(self, req: Req) -> None:
        super().process_new_request(req)
        self.simulator.process_new_request(req, self._deltas_us)
        self._pending_new_requests.append(req)
        self._enqueue_prepare_mutation(
            "process_new_request", self._snapshot_req_for_prepare(req)
        )

    def note_scheduled_prefill_batch(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            self._pending_scheduled_prefill_reqs[req.rid] = req
        self._enqueue_prepare_mutation(
            "note_scheduled_prefill_batch",
            [self._snapshot_req_for_prepare(req) for req in batch.reqs],
        )

    def note_retracted_reqs(self, reqs) -> None:
        for req in reqs:
            self.simulator.process_new_request(req, self._deltas_us)
            self._pending_new_requests.append(req)
        self._enqueue_prepare_mutation(
            "note_retracted_reqs",
            [self._snapshot_req_for_prepare(req) for req in reqs],
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
        with self._prepare_worker.pause():
            if not self._forced_prefill_rids:
                return 0, None
            prioritized_waiting = []
            safe_prefill_cap = self._max_safe_prefill_tokens or 0
            used_safe_prefill_tokens = 0
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
                local_token_counters_by_user.setdefault(req.uid, []).append(
                    req_prefill_tokens
                )
            if not prioritized_waiting:
                return 0, None
            return super().force_prefill_reservations(
                prioritized_waiting,
                token_counters_by_user=token_counters_by_user,
                adder=adder,
                token_to_kv_pool=token_to_kv_pool,
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=delta_fairness_deltas_microseconds,
                max_input_size=max_input_size,
                prefix_computed=prefix_computed,
                max_running_requests=max_running_requests,
                exact_forced_prefills=True,
                reservation_token_cap=used_safe_prefill_tokens,
            )

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
        self.simulator.finished_prefill(batch)
        for req in batch.reqs:
            self._pending_finished_prefill_reqs[req.rid] = req
            self._pending_scheduled_prefill_reqs.pop(req.rid, None)
        self._enqueue_prepare_mutation(
            "finished_prefill",
            [self._snapshot_req_for_prepare(req) for req in batch.reqs],
        )

    def finished_decode(self, batch: ScheduleBatch, decode_rounds: int = 1) -> None:
        super().finished_decode(batch, decode_rounds=decode_rounds)
        needs_simulator_update = []
        for req in batch.reqs:
            self._pending_decoded_reqs[req.rid] = req
            real_event = self.simulator.most_recent_event_real.get(req.rid)
            if not isinstance(real_event, RequestDecodeEvent) or (
                real_event.completion_number < len(req.output_ids)
            ):
                needs_simulator_update.append(req)
        if needs_simulator_update:
            self.simulator.finished_decode(
                SimpleNamespace(reqs=needs_simulator_update),
                decode_rounds=decode_rounds,
            )
        self._enqueue_prepare_mutation(
            "finished_decode",
            (
                [self._snapshot_req_for_prepare(req) for req in batch.reqs],
                decode_rounds,
            ),
        )

    def mark_request_finished(self, req: Req) -> None:
        super().mark_request_finished(req)
        self.simulator.mark_request_finished(req)
        self._pending_finished_rids.add(req.rid)
        self._pending_decoded_reqs.pop(req.rid, None)
        self._pending_finished_prefill_reqs.pop(req.rid, None)
        self._pending_scheduled_prefill_reqs.pop(req.rid, None)
        self._enqueue_prepare_mutation(
            "mark_request_finished", self._snapshot_req_for_prepare(req)
        )
