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

# ---------------------------------------------------------------------------
# USE_C_SIM: when True, UserTimeline.rebuild_from_real_state() runs in the C
# extension (_fairinf_sim.so) instead of pure Python.  The C path releases the
# GIL for the entire simulation loop, allowing GPU kernel launches on the main
# thread to proceed in parallel.  Set to False to fall back to pure Python
# (e.g. for debugging or when the .so has not been built yet).
# ---------------------------------------------------------------------------
USE_C_SIM = True

try:
    from sglang.srt.delta_fairness import _fairinf_sim as _sim_c  # type: ignore[import]
    _C_SIM_AVAILABLE = True
except ImportError:
    _C_SIM_AVAILABLE = False
    if USE_C_SIM:
        logging.getLogger(__name__).warning(
            "_fairinf_sim C extension not found — falling back to pure-Python "
            "simulator. Build it with: python setup_fairinf_sim.py build_ext --inplace"
        )


if USE_C_SIM and _C_SIM_AVAILABLE:
    print("Using C extension for prepare snapshot simulation")

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.request_timeline import ISOLATED_SIM_TIMELINE_WRITER

from .doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
    RequestStatusReal,
)
from .time_estimation import (
    isolated_prefill_time_estimation,
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

if TYPE_CHECKING:
    from .doc_policy import DocPolicy


logger = logging.getLogger(__name__)
DOC_POLICY_WORKER_TRACE_ENABLED = False

# When True: only consider fair clients' decode deadlines when deciding whether
# to force a decode pass. Must match DECODE_PRIORITIZE_FAIR in doc_policy.py.
DECODE_PRIORITIZE_FAIR = True


class _LenOnlySeq:
    """Stores a length as a plain Python int so the worker never touches torch tensors."""
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


class _PrepareReq:
    """
    A torch-safe snapshot of a Req. Stores only plain Python ints — no tensor
    references — so the worker thread cannot accidentally touch the autograd graph.
    Constructed directly from a live Req on the main thread (no dict roundtrip).
    """
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

    @staticmethod
    def from_req(req: Req) -> "_PrepareReq":
        """Construct directly from a live Req — no dict serialization."""
        max_new_tokens = 0
        if req.sampling_params is not None:
            max_new_tokens = int(req.sampling_params.max_new_tokens or 0)
        return _PrepareReq(
            uid=req.uid,
            rid=req.rid,
            prompt_len=int(len(req.origin_input_ids)),
            output_len=int(len(req.output_ids)),
            fill_len=None if req.fill_ids is None else int(len(req.fill_ids)),
            prefix_len=int(len(req.prefix_indices)),
            extend_input_len=int(req.extend_input_len),
            waiting_time_in_decodes=int(req.waiting_time_in_decodes),
            first_time_in_waiting_queue=bool(req.first_time_in_waiting_queue),
            max_new_tokens=max_new_tokens,
        )


@dataclass(frozen=True)
class _PreparedSnapshot:
    task_seq: int
    mutation_seq: int
    deadline_queue: Tuple[object, ...]
    waiting_prefill_deadlines: MappingProxyType
    safe_waiting_queue: Tuple[Req, ...]
    safe_waiting_rids: frozenset
    forced_prefill_queue: Tuple[Req, ...]
    forced_prefill_rids: frozenset
    max_safe_prefill_tokens: Optional[int]
    has_fair_waiting: bool
    has_decode_deadline: bool
    earliest_decode_start_deadline: Optional[float]
    earliest_decode_rid: Optional[str]
    earliest_decode_uid: Optional[str]
    safe_prefix_now: Optional[float]
    skipped_rids: Tuple[str, ...]
    skipped_reasons: Tuple[str, ...]
    breakdown_items: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class _FrozenPrepareCacheState:
    total_user_tokens: Dict[str, int]
    evictable_user_tokens: Dict[str, int]
    fairinf_max_per_user: Optional[int]
    unevictable_limit: Optional[int]
    # Pre-computed set of UIDs that pass user_is_fair_prefill on the main thread.
    # None means "all users are fair" (e.g. fairness disabled).
    known_fair_uids: Optional[frozenset] = None


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
        if USE_C_SIM and _C_SIM_AVAILABLE:
            _sim_c.patch_simulator(self._simulator)
        self._published_snapshot: Optional[_PreparedSnapshot] = None
        self._published_snapshot_lock = threading.Lock()
        self._thread_exception: Optional[BaseException] = None
        self._worker_stop = False
        self._task_seq = 0
        self._mutation_seq = 0
        self._applied_mutation_seq = 0
        self._pending_mutation_queue_backpressure_wait_ms = 0.0
        self._pending_prepare_task_queue_backpressure_wait_ms = 0.0
        self._pending_mutation_queue_drain_wait_ms = 0.0
        self._pending_duplicate_state_wait_ms = 0.0
        self._queue_csv_path = os.path.join(os.getcwd(), "doc_policy_prepare_queue.csv")
        self._queue_csv_header_written = False
        self._trace_csv_path = os.path.join(os.getcwd(), "doc_policy_worker_trace.csv")
        self._worker_activity_csv_path = os.path.join(os.getcwd(), "doc_policy_worker_activity.csv")
        self._worker_activity_csv_lock = threading.Lock()
        self._worker_activity_csv_header_written = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="doc-policy-prepare",
            daemon=True,
        )
        self._worker_thread.start()

    def _log_worker_activity(self, event: str, **fields) -> None:
        """Append one row to doc_policy_worker_activity.csv (thread-safe)."""
        keys = [
            "ts", "event", "kind", "task_seq", "mutation_seq",
            "waiting_len", "running_len",
            "mutation_queue_len", "task_queue_len",
            "elapsed_ms",
        ]
        row = {"ts": f"{time.time():.6f}", "event": event}
        row.update(fields)
        line = ",".join(str(row.get(k, "")) for k in keys) + "\n"
        with self._worker_activity_csv_lock:
            with open(self._worker_activity_csv_path, "a") as f:
                if not self._worker_activity_csv_header_written:
                    f.write(",".join(keys) + "\n")
                    self._worker_activity_csv_header_written = True
                f.write(line)

    def _trace(self, action: str, **fields) -> None:
        if not DOC_POLICY_WORKER_TRACE_ENABLED:
            return
        keys = [
            "ts", "action", "task_seq", "pass_id", "event_type",
            "selected_count", "snapshot_seq", "earliest_rid",
            "earliest_completion", "earliest_deadline",
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

    def make_prepare_req(self, req: Req) -> _PrepareReq:
        return _PrepareReq.from_req(req)

    def make_prepare_batch(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        if batch is None:
            return None
        return SimpleNamespace(reqs=[_PrepareReq.from_req(req) for req in batch.reqs])

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
        self._log_worker_activity(
            "enqueue_mutation",
            kind=kind,
            mutation_seq=self._mutation_seq,
            mutation_queue_len=len(self._mutation_queue),
            task_queue_len=len(self._task_queue),
        )

    def enqueue_task(self, task: tuple) -> int:
        self._maybe_wait_for_queue_capacity(self._task_queue, queue_name="prepare")
        self._task_seq += 1
        seq = self._task_seq
        waiting_len = len(task[0]) if task and task[0] is not None else 0
        running_len = len(task[1].reqs) if task and task[1] is not None else 0
        self._task_queue.append((seq, task))
        self._log_worker_activity(
            "enqueue_task",
            task_seq=seq,
            mutation_seq=self._mutation_seq,
            waiting_len=waiting_len,
            running_len=running_len,
            mutation_queue_len=len(self._mutation_queue),
            task_queue_len=len(self._task_queue),
        )
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

    @property
    def simulator(self) -> "AlternateHistorySimulator":
        return self._simulator

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

    def _publish_snapshot(self, snapshot: _PreparedSnapshot, elapsed_ms: float = 0.0) -> None:
        with self._published_snapshot_lock:
            self._published_snapshot = snapshot
            self._owner._last_prepare_breakdown_ms = dict(snapshot.breakdown_items)
        self._log_worker_activity(
            "publish_snapshot",
            task_seq=snapshot.task_seq,
            mutation_seq=snapshot.mutation_seq,
            waiting_len=len(snapshot.safe_waiting_queue),
            running_len=sum(1 for c in snapshot.deadline_queue if getattr(c, "event_type", "") == "decode"),
            mutation_queue_len=len(self._mutation_queue),
            task_queue_len=len(self._task_queue),
            elapsed_ms=f"{elapsed_ms:.2f}",
        )
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

    def _build_prepare_snapshot(
        self,
        waiting_queue: List[_PrepareReq],
        running_batch,
        *,
        task_seq: int,
        mutation_seq: int,
        breakdown: Optional[Dict[str, float]] = None,
        frozen_cache_state: Optional[_FrozenPrepareCacheState] = None,
        frozen_inputs: Optional[_FrozenPrepareInputs] = None,
    ) -> _PreparedSnapshot:
        owner = self._owner
        simulator = self._simulator
        target_breakdown: Dict[str, float] = {} if breakdown is None else breakdown
        build_start = time.perf_counter()
        simulator.get_live_users(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=(
                frozen_inputs.deltas_us if frozen_inputs is not None else owner._deltas_us
            ),
        )
        # Skip rebuild for users that are definitively unfair (over their KV
        # fair-share reservation).  Their decode deadlines are filtered out by
        # req_is_fair_decode and their prefill deadlines are set to inf, so
        # running the full simulation for them produces no usable output.
        # known_fair_uids=None means "no KV info available → rebuild everyone".
        _known_fair = (
            frozen_cache_state.known_fair_uids
            if frozen_cache_state is not None
            else None
        )
        for uid, user_timeline in simulator.users.items():
            if _known_fair is not None and uid not in _known_fair:
                continue
            user_timeline.rebuild_from_real_state(
                timing_breakdown=target_breakdown,
            )
        ISOLATED_SIM_TIMELINE_WRITER.write_snapshot(simulator)
        after_sync = time.perf_counter()

        fairinf_n = (
            frozen_inputs.fairinf_n
            if frozen_inputs is not None
            else max(int(owner.delta_fairness_n or 1), 1)
        )

        # Pre-compute per-user fairness once for all unique users in waiting_queue.
        # Used in both the deadline filter and the safe-prefix scan below.
        under_memory_pressure = (
            frozen_inputs is not None and frozen_inputs.no_retraction_cap is not None
        )
        if under_memory_pressure:
            fair_uids: Optional[set] = set()
            for req in waiting_queue:
                if req.uid not in fair_uids and owner._user_is_fair_prefill_from_frozen(
                    req.uid,
                    this_user_sum=req.get_estimated_prefill_impact(),
                    frozen_cache_state=frozen_cache_state,
                ):
                    fair_uids.add(req.uid)
            deadline_waiting_queue = [req for req in waiting_queue if req.uid in fair_uids]
        else:
            fair_uids = None
            deadline_waiting_queue = waiting_queue

        # Compute fair decode UIDs from the main-thread-computed known_fair_uids.
        # Only their decode deadlines drive forced-decode decisions (when DECODE_PRIORITIZE_FAIR).
        # If known_fair_uids is unavailable (no tree_cache), fall back to allowing all users.
        if (
            DECODE_PRIORITIZE_FAIR
            and frozen_cache_state is not None
            and frozen_cache_state.known_fair_uids is not None
        ):
            fair_decode_uids: Optional[frozenset] = frozen_cache_state.known_fair_uids
        else:
            fair_decode_uids = None  # None = all users allowed

        # Pre-compute running-batch token sums once — used for every decode candidate.
        if running_batch is not None and running_batch.reqs:
            _rb_token_lens = [
                len(item.fill_ids)
                if item.fill_ids is not None
                else len(item.origin_input_ids) + len(item.output_ids)
                for item in running_batch.reqs
            ]
            _rb_total_tokens = sum(_rb_token_lens)
            _rb_max_tokens = max(_rb_token_lens)
            _rb_count = len(running_batch.reqs)
            _pooled_decode_s = pooled_decode_time_estimation(
                _rb_total_tokens, _rb_max_tokens, _rb_count, fairinf_n
            )
        else:
            _pooled_decode_s = 0.0

        # Pre-compute default deltas dict (used when tracked_req has no per-req deltas).
        _default_deltas = frozen_inputs.deltas_us if frozen_inputs is not None else owner._deltas_us

        deadline_result = simulator.build_deadline_candidates(
            deadline_waiting_queue,
            running_batch,
            include_ordered_waiting_queue=True,
            req_is_fair_prefill=lambda req, rb: fair_uids is None or req.uid in fair_uids,
            req_is_fair_decode=lambda req, rb: fair_decode_uids is None or req.uid in fair_decode_uids,
            event_delta_seconds=lambda tracked_req, event: float(
                (tracked_req.deltas_in_microseconds or _default_deltas).get(
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
            pooled_decode_estimate_seconds=lambda req, rb: _pooled_decode_s,
        )
        if len(deadline_result) == 3:
            deadline_queue, waiting_prefill_deadline_by_rid, ordered_waiting_queue = deadline_result
        else:
            deadline_queue, waiting_prefill_deadline_by_rid = deadline_result
            ordered_waiting_queue = None
        after_deadline = time.perf_counter()
        safe_state = owner._compute_safe_prefix_state(
            deadline_queue,
            waiting_prefill_deadline_by_rid,
            waiting_queue,
            running_batch,
            ordered_waiting_queue=ordered_waiting_queue,
            frozen_cache_state=frozen_cache_state,
            frozen_inputs=frozen_inputs,
            fair_uids=fair_uids,
        )
        after_safe = time.perf_counter()
        target_breakdown["sync_live_user_tracking_ms"] = (after_sync - build_start) * 1000.0
        target_breakdown["build_deadline_candidates_ms"] = (after_deadline - after_sync) * 1000.0
        target_breakdown["sort_waiting_prefills_ms"] = 0.0
        target_breakdown["safe_prefix_scan_ms"] = (after_safe - after_deadline) * 1000.0
        target_breakdown["build_pass_state_ms"] = (after_safe - build_start) * 1000.0
        return _PreparedSnapshot(
            task_seq=task_seq,
            mutation_seq=mutation_seq,
            deadline_queue=tuple(deadline_queue),
            waiting_prefill_deadlines=MappingProxyType(dict(waiting_prefill_deadline_by_rid)),
            safe_waiting_queue=tuple(safe_state["safe_waiting_queue"]),
            safe_waiting_rids=frozenset(safe_state["safe_waiting_rids"]),
            forced_prefill_queue=tuple(safe_state["forced_prefill_queue"]),
            forced_prefill_rids=frozenset(safe_state["forced_prefill_rids"]),
            max_safe_prefill_tokens=safe_state["max_safe_prefill_tokens"],
            has_fair_waiting=bool(safe_state["has_fair_waiting"]),
            has_decode_deadline=bool(safe_state["has_decode_deadline"]),
            earliest_decode_start_deadline=safe_state["earliest_decode_start_deadline"],
            earliest_decode_rid=owner._debug_earliest_decode_rid,
            earliest_decode_uid=owner._debug_earliest_decode_uid,
            safe_prefix_now=safe_state["safe_prefix_now"],
            skipped_rids=tuple(safe_state.get("skipped_rids", [])),
            skipped_reasons=tuple(safe_state.get("skipped_reasons", [])),
            breakdown_items=tuple(target_breakdown.items()),
        )

    def _apply_mutation(self, kind: str, payload) -> None:
        simulator = self._simulator
        if kind == "process_new_request":
            req, deltas_us, *rest = payload
            arrival_ts = rest[0] if rest else None
            simulator.process_new_request(req, deltas_us, arrival_timestamp=arrival_ts)
            return
        if kind == "note_retracted_reqs":
            reqs, deltas_us = payload
            for req in reqs:
                simulator.process_new_request(req, deltas_us)
            return
        if kind == "note_prefill_done":
            (reqs,) = payload
            for req in reqs:
                ut = simulator.users.get(req.uid)
                if ut is None:
                    continue
                tracked = ut.request_timelines.get(req.rid)
                if tracked is None:
                    continue
                # Commit a prefill event to history if not already present.
                if not any(isinstance(e, RequestPrefillEvent) for e in tracked.timeline.history):
                    iso_ts = (
                        tracked.arrival_timestamp
                        + isolated_prefill_time_estimation(
                            len(req.origin_input_ids), len(req.origin_input_ids), 1, simulator.fairinf_n
                        )
                    )
                    tracked.timeline.history.append(
                        RequestPrefillEvent(req_id=req.rid, duration=0.0, end_timestamp=iso_ts)
                    )
                s = ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid))
                s.prefill_done = True
            return
        if kind == "finished_decode":
            return
        if kind == "logical_decode_update":
            running_batch, decode_steps = payload if len(payload) == 2 else (payload[0], 1)
            if running_batch is not None:
                self._simulator.finished_decode(running_batch, decode_rounds=decode_steps)
            return
        if kind == "mark_request_finished":
            req, pass_id = payload
            # Mark is_complete on requests_real so rebuild_from_real_state can drop
            # this request from active_rids early on the next simulation pass.
            ut = simulator.users.get(req.uid)
            if ut is not None:
                s = ut.requests_real.get(req.rid)
                if s is not None:
                    s.is_complete = True
            simulator.mark_request_finished(req)
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
                try:
                    with torch.inference_mode():
                        mut_start = time.perf_counter()
                        kind, payload = self._mutation_queue.popleft()
                        self._apply_mutation(kind, payload)
                        self._applied_mutation_seq += 1
                        mut_elapsed_ms = (time.perf_counter() - mut_start) * 1000.0
                    self._log_worker_activity(
                        "apply_mutation",
                        kind=kind,
                        mutation_seq=self._applied_mutation_seq,
                        mutation_queue_len=len(self._mutation_queue),
                        task_queue_len=len(self._task_queue),
                        elapsed_ms=f"{mut_elapsed_ms:.2f}",
                    )
                except BaseException as exc:
                    self._thread_exception = exc
                    logger.exception("prepare worker mutation failed")
                    return
                continue
            if not self._task_queue:
                time.sleep(0.001)
                continue
            try:
                with torch.inference_mode():
                    task_start = time.perf_counter()
                    task_seq, task = self._task_queue.popleft()
                    (
                        waiting_queue,
                        running_batch,
                        frozen_cache_state,
                        frozen_inputs,
                        _requested_mutation_seq,
                    ) = task
                    self._log_worker_activity(
                        "start_task",
                        task_seq=task_seq,
                        mutation_seq=self._applied_mutation_seq,
                        waiting_len=len(waiting_queue),
                        running_len=len(running_batch.reqs) if running_batch is not None else 0,
                        mutation_queue_len=len(self._mutation_queue),
                        task_queue_len=len(self._task_queue),
                    )

                    breakdown: Dict[str, float] = {
                        "logical_event_update_ms": 0.0,
                        "rebuild_from_real_state_ms": 0.0,
                        "prepare_during_gpu_execution_total_ms": 0.0,
                    }
                    snapshot = self._build_prepare_snapshot(
                        waiting_queue,
                        running_batch,
                        task_seq=task_seq,
                        mutation_seq=self._applied_mutation_seq,
                        breakdown=breakdown,
                        frozen_cache_state=frozen_cache_state,
                        frozen_inputs=frozen_inputs,
                    )
                    task_elapsed_ms = (time.perf_counter() - task_start) * 1000.0
                    self._publish_snapshot(snapshot, elapsed_ms=task_elapsed_ms)
            except BaseException as exc:
                self._thread_exception = exc
                logger.exception("prepare worker task failed")
                return
