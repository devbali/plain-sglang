from __future__ import annotations

"""DocPolicy: deadline-ordered fairness scheduling policy."""

import logging
import os
import time
from copy import copy
from types import SimpleNamespace
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
    _PrepareReq,
)
from .time_estimation import (
    isolated_decode_time_estimation,  # imported so tests can patch this module
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

logger = logging.getLogger(__name__)
DOC_POLICY_TRACE_ENABLED = False

# When True: sort the safe prefill queue fair-clients-first.
PREFILL_PRIORITIZE_FAIR = True
# When True: only consider fair clients' decode deadlines when deciding whether to force a decode pass.
DECODE_PRIORITIZE_FAIR = True
# When True: only consider fair clients' decode deadlines when computing earliest_decode_start_deadline
# (i.e. unfair users' decode deadlines do not throttle prefills of fair users).
DECODE_DEADLINE_FAIR_ONLY = False


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
        self._earliest_decode_start_deadline: Optional[float] = None
        self._safe_prefix_now: Optional[float] = None
        self._debug_first_waiting_rid: Optional[str] = None
        self._debug_earliest_decode_rid: Optional[str] = None
        self._debug_earliest_decode_uid: Optional[str] = None
        self._debug_first_waiting_prompt_tokens: Optional[int] = None
        self._debug_first_candidate_prefill_ms: Optional[float] = None
        self._debug_first_candidate_residual_slack_ms: Optional[float] = None
        self._debug_known_fair_uids: Optional[frozenset] = None
        self._debug_earliest_uid_unevictable_kv: Optional[int] = None
        self._debug_fairinf_max_per_user: Optional[int] = None
        self._current_pass_id = 0
        self._last_prepare_breakdown_ms: Dict[str, float] = {}
        self._last_pass_state_source = "init"
        self._last_skipped_rids: Tuple[str, ...] = ()
        self._last_skipped_reasons: Tuple[str, ...] = ()
        self._prefill_no_retraction_token_cap: Optional[int] = None
        self._last_consumed_prepare_snapshot_seq = 0
        self._last_prepare_task_seq = 0
        self._last_force_decode_reason: str = ""
        self._force_prefill_override_rid: Optional[str] = None  # RID that triggered prefill_deadline_earlier_than_decode
        self._force_prefill_override_uid: Optional[str] = None
        self._debug_override_fate: str = ""  # what happened to the override request in process_waiting_queue_prefills
        self._pass_retraction_count: int = 0  # retractions this pass; reset after each log
        self._prepare_worker = _DocPolicyPrepareWorker(
            self,
            isolated_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
        )
        self._trace_csv_path = os.path.join(os.getcwd(), "doc_policy_trace.csv")

    @property
    def simulator(self) -> AlternateHistorySimulator:
        """Read-only access to the worker thread's simulator (for diagnostics)."""
        return self._prepare_worker.simulator

    # -------------------------------------------------------------------------
    # Tracing
    # -------------------------------------------------------------------------

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

            "earliest_rid",
            "earliest_completion",
            "earliest_deadline",
            "last_consumed_snapshot_seq",
            "current_pass_id",
            # start_of_pass decision fields
            "pass_state_source",
            "force_decode",
            "max_safe_prefill_tokens",
            "has_decode_deadline",
            "forced_prefill_count",
            "forced_prefill_rids",
            "safe_waiting_count",
            "safe_waiting_top3_rids",
            "safe_waiting_top3_deadlines",
            "snapshot_age_ms",
            # forced-prefill skip diagnostics
            "safe_skipped_rids",
            "safe_skipped_reasons",
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

    # -------------------------------------------------------------------------
    # Prepare-worker helpers
    # -------------------------------------------------------------------------

    def _enqueue_prepare_mutation(self, kind: str, payload) -> None:
        self._prepare_worker.enqueue_mutation(kind, payload)

    def _enqueue_prepare_task(self, task: tuple) -> int:
        return self._prepare_worker.enqueue_task(task)

    def _raise_prepare_thread_exception_if_any(self) -> None:
        self._prepare_worker.raise_exception_if_any()

    # -------------------------------------------------------------------------
    # Delta / fairness helpers
    # -------------------------------------------------------------------------

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

    def _freeze_prepare_cache_state(
        self,
        known_fair_uids: Optional[frozenset] = None,
    ) -> Optional[_FrozenPrepareCacheState]:
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
            known_fair_uids=known_fair_uids,
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
        if not self.delta_fairness_n:
            return False
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None or tree_cache.fairinf_max_per_user is None:
            return True
        # Use fairinf_max_per_user (total per-user budget), not the unevictable
        # reservation from calculate_delta_fair_reservation_size.
        return tree_cache.user_total_is_under_fair_share_reservation(
            user_id, this_user_sum
        )

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
        # cached_unevictable_tokens already includes decode output KV slots via
        # note_decode_kv_alloc → total_user_counters. Do not add uncached_running_tokens
        # (seq_len - prefix_indices) separately — that would double-count decode KV.
        protected_tokens = cached_unevictable_tokens + pending_prefill_tokens
        return (
            protected_tokens + req.extend_input_len
            <= frozen_cache_state.fairinf_max_per_user
        )

    def _force_prefill_within_user_headroom(
        self,
        req: Req,
        *,
        running_batch: Optional[ScheduleBatch],
        pending_prefill_tokens: int = 0,
    ) -> bool:
        tree_cache = self.tree_cache
        if tree_cache is None or tree_cache.fairinf_max_per_user is None:
            return True
        # Use only unevictable cached tokens (which already include decode KV via
        # note_decode_kv_alloc) — do not add uncached_running_tokens (double-count)
        # or decode_headroom (speculative, over-rejects users near their limit).
        cached_total = tree_cache.total_user_counters.get_tokens(req.uid)
        cached_evictable = tree_cache.evictable_total_user_counters.get_tokens(req.uid)
        cached_unevictable = cached_total - cached_evictable
        return (
            cached_unevictable + pending_prefill_tokens + req.extend_input_len
            <= tree_cache.fairinf_max_per_user
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
        return pooled_decode_time_estimation(
            sum(token_counts),
            max(token_counts),
            len(token_counts),
            max(int(self.delta_fairness_n or 1), 1),
        )

    # -------------------------------------------------------------------------
    # Safe-prefix state computation (used by both snapshot build and consume)
    # -------------------------------------------------------------------------

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
        fair_uids: Optional[set] = None,
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
                and (not DECODE_DEADLINE_FAIR_ONLY or fair_uids is None or candidate.req.uid in fair_uids)
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        if earliest_decode_candidate is None:
            # No decode deadline — force all fair waiting requests unconditionally.
            # Without a decode deadline there is no time constraint on prefills,
            # so every fair waiting request should be admitted (with retraction if
            # needed) to prevent starvation when the cache is full of unfair reqs.
            for req in waiting_queue:
                if fair_uids is None or req.uid in fair_uids:
                    forced_prefill_queue.append(req)
                    forced_prefill_rids.add(req.rid)
            max_safe_prefill_tokens = sum(len(req.origin_input_ids) for req in forced_prefill_queue) or None
            has_fair_waiting = bool(forced_prefill_queue) or has_fair_waiting
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
                "skipped_rids": [],
                "skipped_reasons": [],
            }

        has_decode_deadline = True
        earliest_decode_start_deadline = earliest_decode_candidate.start_deadline
        now = time.time()
        safe_prefix_now = now
        selected_batch: List[Req] = []
        selected_batch_max_tokens = 0
        selected_batch_count = 0
        safe_prompt_tokens = 0
        no_retraction_cap = (
            frozen_inputs.no_retraction_cap
            if frozen_inputs is not None
            else self._no_retraction_prefill_token_cap(running_batch)
        )
        pending_prefill_sum_by_user: Dict[str, int] = {}
        pending_prefill_len_by_user: Dict[str, int] = {}
        skipped_rids: List[str] = []
        skipped_reasons: List[str] = []
        for req in safe_waiting_queue:
            req_prefill_tokens = len(req.origin_input_ids)
            next_safe_prompt_tokens = safe_prompt_tokens + req_prefill_tokens
            beyond_no_retraction_cap = (
                no_retraction_cap is not None
                and next_safe_prompt_tokens > no_retraction_cap
            )
            is_fair = fair_uids is None or req.uid in fair_uids
            if not is_fair and beyond_no_retraction_cap:
                skipped_rids.append(req.rid)
                skipped_reasons.append("unfair_user")
                continue
            if is_fair and not self._force_prefill_within_user_headroom_from_frozen(
                req,
                running_batch=running_batch,
                pending_prefill_tokens=pending_prefill_sum_by_user.get(req.uid, 0),
                frozen_cache_state=frozen_cache_state,
                new_token_ratio=(
                    frozen_inputs.new_token_ratio if frozen_inputs is not None else 0.0
                ),
            ):
                skipped_rids.append(req.rid)
                skipped_reasons.append("headroom")
                continue
            candidate_max_tokens = max(selected_batch_max_tokens, req_prefill_tokens)
            pooled_prefill_s = pooled_prefill_time_estimation(
                next_safe_prompt_tokens,
                candidate_max_tokens,
                selected_batch_count + 1,
                (
                    frozen_inputs.fairinf_n
                    if frozen_inputs is not None
                    else max(int(self.delta_fairness_n or 1), 1)
                ),
            )
            if now + pooled_prefill_s <= earliest_decode_start_deadline:
                safe_prompt_tokens = next_safe_prompt_tokens
                selected_batch_max_tokens = candidate_max_tokens
                selected_batch_count += 1
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
            skipped_rids.append(req.rid)
            skipped_reasons.append("time")
            break

        max_safe_prefill_tokens = safe_prompt_tokens
        state = {
            "safe_waiting_queue": safe_waiting_queue,
            "safe_waiting_rids": safe_waiting_rids,
            "forced_prefill_queue": forced_prefill_queue,
            "forced_prefill_rids": forced_prefill_rids,
            "max_safe_prefill_tokens": max_safe_prefill_tokens,
            "has_fair_waiting": has_fair_waiting,
            "has_decode_deadline": has_decode_deadline,
            "earliest_decode_start_deadline": earliest_decode_start_deadline,
            "safe_prefix_now": safe_prefix_now,
            "skipped_rids": skipped_rids,
            "skipped_reasons": skipped_reasons,
        }

        print(f"DEBUG DOC POLICY COMPUTE_SAFE_PREFIX_STATE: {state}")
        return state
    # -------------------------------------------------------------------------
    # Prepare-thread snapshot builder (called from the worker thread)
    # -------------------------------------------------------------------------

    def consume_prepare_thread_wait_metrics(self) -> Dict[str, float]:
        return self._prepare_worker.consume_wait_metrics()

    # -------------------------------------------------------------------------
    # Snapshot consumption (main thread)
    # -------------------------------------------------------------------------

    def _consume_prepared_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
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

        waiting_by_rid = {req.rid: req for req in waiting_queue}
        running_by_rid = (
            {} if running_batch is None else {req.rid: req for req in running_batch.reqs}
        )
        live_req_by_rid = dict(waiting_by_rid)
        live_req_by_rid.update(running_by_rid)

        # Remap deadline_queue: replace _PrepareReq with live Req, drop any that left.
        remapped_deadline_queue = []
        for candidate in snapshot.deadline_queue:
            live_req = live_req_by_rid.get(candidate.req.rid)
            if live_req is None:
                continue
            remapped_candidate = copy(candidate)
            remapped_candidate.req = live_req
            remapped_deadline_queue.append(remapped_candidate)
        self._deadline_queue = tuple(remapped_deadline_queue)

        # Trust the snapshot's ordering and forced-prefill decision entirely.
        # Requests that arrived after the snapshot was built will be picked up
        # by the next pass — the worker will have seen them by then.
        self._waiting_prefill_start_deadline_by_rid = {
            rid: deadline
            for rid, deadline in snapshot.waiting_prefill_deadlines.items()
            if rid in waiting_by_rid
        }
        self._safe_waiting_queue = tuple(
            waiting_by_rid[req.rid]
            for req in snapshot.safe_waiting_queue
            if req.rid in waiting_by_rid
        )
        self._safe_waiting_rids = frozenset(req.rid for req in self._safe_waiting_queue)
        self._forced_prefill_queue = tuple(
            waiting_by_rid[req.rid]
            for req in snapshot.forced_prefill_queue
            if req.rid in waiting_by_rid
        )
        self._forced_prefill_rids = frozenset(req.rid for req in self._forced_prefill_queue)
        self._max_safe_prefill_tokens = snapshot.max_safe_prefill_tokens
        self._has_fair_waiting = snapshot.has_fair_waiting
        self._has_decode_deadline = snapshot.has_decode_deadline
        self._earliest_decode_start_deadline = snapshot.earliest_decode_start_deadline
        self._safe_prefix_now = snapshot.safe_prefix_now
        self._last_skipped_rids = snapshot.skipped_rids
        self._last_skipped_reasons = snapshot.skipped_reasons

        earliest_decode_candidate = min(
            (c for c in self._deadline_queue if c.event_type == "decode"),
            key=lambda c: c.start_deadline,
            default=None,
        )
        self._debug_earliest_decode_rid = (
            None if earliest_decode_candidate is None else earliest_decode_candidate.req.rid
        )
        self._debug_earliest_decode_uid = (
            None if earliest_decode_candidate is None else earliest_decode_candidate.req.uid
        )
        self._last_consumed_prepare_snapshot_seq = snapshot.task_seq
        self._trace(
            "consume_accept",
            snapshot_seq=snapshot.task_seq,
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

    # -------------------------------------------------------------------------
    # Main scheduling entry points
    # -------------------------------------------------------------------------

    def refresh_decode_hot_path_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        if self._consume_prepared_pass_state(
            waiting_queue,
            running_batch,
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

        # If no new task has been enqueued since the last consumed snapshot, enqueue one
        # now (with no mutations — just a fresh view of current state) so the background
        # thread can produce an up-to-date snapshot.  Skip this on fully idle passes
        # (nothing running and nothing waiting) since there is nothing to compute.
        if self._last_prepare_task_seq <= self._last_consumed_prepare_snapshot_seq:
            has_live = (running_batch is not None and len(running_batch.reqs) > 0) or len(waiting_queue) > 0
            if has_live:
                running_prepare = self._prepare_worker.make_prepare_batch(running_batch)
                waiting_prepare = [_PrepareReq.from_req(req) for req in waiting_queue]
                task_seq = self._enqueue_prepare_task(
                    (
                        waiting_prepare,
                        running_prepare,
                        self._freeze_prepare_cache_state(known_fair_uids=None),
                        self._freeze_prepare_inputs(new_token_ratio=new_token_ratio),
                        self._prepare_worker.mutation_seq,
                    )
                )
                self._last_prepare_task_seq = task_seq
                self._trace(
                    "enqueue_prepare",
                    pass_id=self._current_pass_id,
                    task_seq=task_seq,
                    event_type="start_of_pass_refresh",
                    waiting_len=len(waiting_prepare),
                    running_len=0 if running_prepare is None else len(running_prepare.reqs),
                    decode_steps=0,
                    current_pass_id=self._current_pass_id,
                )
        # Wait until the worker publishes a snapshot newer than the last one we consumed.
        if self._last_prepare_task_seq > self._last_consumed_prepare_snapshot_seq:
            self._prepare_worker.wait_for_snapshot(
                min_task_seq=self._last_consumed_prepare_snapshot_seq + 1,
            )
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "start_of_pass_consume_prepared"
        else:
            self._last_pass_state_source = "start_of_pass_no_prepared"

        after_simulator = time.perf_counter()
        self._current_pass_id += 1
        after_build = time.perf_counter()
        breakdown["doc_policy_after_super_ms"] = (after_simulator - pass_start) * 1000.0
        breakdown["doc_policy_start_of_pass_total_ms"] = (after_build - pass_start) * 1000.0
        self._last_pass_breakdown_ms = breakdown

        snapshot = self._prepare_worker.latest_snapshot()
        snap_age_ms = ""
        if snapshot is not None:
            snap_age_ms = f"{(time.time() - snapshot.safe_prefix_now) * 1000.0:.1f}" if snapshot.safe_prefix_now else ""
        top3 = list(self._safe_waiting_queue)[:3]
        self._trace(
            "start_of_pass",
            pass_id=self._current_pass_id,
            snapshot_seq=self._last_consumed_prepare_snapshot_seq,
            waiting_len=len(waiting_queue),
            running_len=0 if running_batch is None else len(running_batch.reqs),
            pass_state_source=self._last_pass_state_source,
            force_decode=self._has_decode_deadline and (self._max_safe_prefill_tokens or 0) <= 0,
            max_safe_prefill_tokens=self._max_safe_prefill_tokens,
            has_decode_deadline=self._has_decode_deadline,
            earliest_deadline=self._earliest_decode_start_deadline,
            earliest_rid=self._debug_earliest_decode_rid,
            forced_prefill_count=len(self._forced_prefill_rids),
            forced_prefill_rids="|".join(list(self._forced_prefill_rids)[:3]),
            safe_waiting_count=len(self._safe_waiting_queue),
            safe_waiting_top3_rids="|".join(r.rid for r in top3),
            safe_waiting_top3_deadlines="|".join(
                f"{self._waiting_prefill_start_deadline_by_rid.get(r.rid, float('inf')):.3f}"
                for r in top3
            ),
            snapshot_age_ms=snap_age_ms,
            safe_skipped_rids="|".join(self._last_skipped_rids[:5]),
            safe_skipped_reasons="|".join(self._last_skipped_reasons[:5]),
        )

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
            # Build _PrepareReq snapshots directly — no dict roundtrip, no torch tensors.
            running_prepare = self._prepare_worker.make_prepare_batch(running_batch)
            waiting_prepare = [_PrepareReq.from_req(req) for req in waiting_queue]
            if event_type == "decode" and running_batch is not None and decode_steps > 0:
                # Single mutation for the whole N-step decode epoch: advance finished_decode
                # N times then rebuild once, rather than N separate rebuild calls.
                running_prepare_for_mutation = self._prepare_worker.make_prepare_batch(running_batch)
                self._enqueue_prepare_mutation(
                    "logical_decode_update",
                    (running_prepare_for_mutation, decode_steps),
                )
            if event_type == "prefill" and scheduled_batch is not None and scheduled_batch.reqs:
                # Mark these requests as prefill-done in the simulator before the task runs,
                # so rebuild_from_real_state treats them as active (decode-eligible).
                scheduled_prepare_reqs = [_PrepareReq.from_req(req) for req in scheduled_batch.reqs]
                self._enqueue_prepare_mutation("note_prefill_done", (scheduled_prepare_reqs,))
                # Include the newly prefilled reqs in the running_prepare passed to the task,
                # so get_live_users sees them in running_batch and applies the prefill_done status.
                existing_reqs = running_prepare.reqs if running_prepare is not None else []
                running_prepare = SimpleNamespace(reqs=existing_reqs + scheduled_prepare_reqs)
            # Compute fair UIDs on the main thread using unevictable KV fairness check.
            # A user is fair for decode deadline purposes if their unevictable KV usage
            # is under their delta-fair reservation limit.
            if DECODE_PRIORITIZE_FAIR:
                tree_cache = getattr(self, "tree_cache", None)
                if tree_cache is not None and getattr(tree_cache, "fairinf_max_per_user", None) is not None:
                    all_live_uids_decode: set = set()
                    if running_batch is not None:
                        for req in running_batch.reqs:
                            all_live_uids_decode.add(req.uid)
                    for req in waiting_queue:
                        all_live_uids_decode.add(req.uid)
                    known_fair_uids = frozenset(
                        uid for uid in all_live_uids_decode
                        if tree_cache.user_unevictable_kv_is_under_fair_share_reservation(uid)
                    )
                    self._debug_known_fair_uids = known_fair_uids
                    self._debug_fairinf_max_per_user = getattr(tree_cache, "fairinf_max_per_user", None)
                    # Log unevictable KV for the current earliest decode uid
                    _earliest_uid = self._debug_earliest_decode_uid
                    if _earliest_uid:
                        _tc = tree_cache
                        _total = _tc.total_user_counters.get_tokens(_earliest_uid) if hasattr(_tc, "total_user_counters") else None
                        _evictable = _tc.evictable_total_user_counters.get_tokens(_earliest_uid) if hasattr(_tc, "evictable_total_user_counters") else None
                        if _total is not None and _evictable is not None:
                            self._debug_earliest_uid_unevictable_kv = _total - _evictable
                        else:
                            self._debug_earliest_uid_unevictable_kv = None
                    else:
                        self._debug_earliest_uid_unevictable_kv = None
                else:
                    known_fair_uids = None  # no limit configured, allow all
                    self._debug_known_fair_uids = None
                    self._debug_earliest_uid_unevictable_kv = None
                    self._debug_fairinf_max_per_user = None
            else:
                known_fair_uids = None
                self._debug_known_fair_uids = None
                self._debug_earliest_uid_unevictable_kv = None
                self._debug_fairinf_max_per_user = None
            task_seq = self._enqueue_prepare_task(
                (
                    waiting_prepare,
                    running_prepare,
                    self._freeze_prepare_cache_state(known_fair_uids=known_fair_uids),
                    self._freeze_prepare_inputs(new_token_ratio=new_token_ratio),
                    self._prepare_worker.mutation_seq,
                )
            )
            self._last_prepare_task_seq = task_seq
            self._trace(
                "enqueue_prepare",
                pass_id=self._current_pass_id,
                task_seq=task_seq,
                event_type=event_type,
                waiting_len=len(waiting_prepare),
                running_len=0 if running_prepare is None else len(running_prepare.reqs),
                decode_steps=decode_steps,
                current_pass_id=self._current_pass_id,
            )
            elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
            self._last_prepare_breakdown_ms = {
                "sync_live_user_tracking_ms": 0.0,
                "logical_event_update_ms": 0.0,
                "rebuild_from_real_state_ms": 0.0,
                "build_deadline_candidates_ms": 0.0,
                "prepare_during_gpu_execution_total_ms": elapsed_ms,
            }

    # -------------------------------------------------------------------------
    # Mutation-only lifecycle hooks
    # -------------------------------------------------------------------------

    def process_new_request(self, req: Req) -> None:
        arrival_ts = time.time()
        super().process_new_request(req)
        TIMELINE_WRITER.mark_queue_enter(req.rid, req.uid)
        self._enqueue_prepare_mutation(
            "process_new_request",
            (_PrepareReq.from_req(req), dict(self._deltas_us), arrival_ts),
        )

    def note_scheduled_prefill_batch(self, batch: ScheduleBatch) -> None:
        self._enqueue_prepare_mutation(
            "note_scheduled_prefill_batch",
            [_PrepareReq.from_req(req) for req in batch.reqs],
        )

    def note_retracted_reqs(self, reqs) -> None:
        self._pass_retraction_count += len(reqs)
        self._enqueue_prepare_mutation(
            "note_retracted_reqs",
            (
                [_PrepareReq.from_req(req) for req in reqs],
                dict(self._deltas_us),
            ),
        )

    def mark_request_finished(self, req: Req) -> None:
        super().mark_request_finished(req)
        TIMELINE_WRITER.mark_completed(req.rid, req.uid)
        self._enqueue_prepare_mutation(
            "mark_request_finished",
            (_PrepareReq.from_req(req), self._current_pass_id),
        )

    # -------------------------------------------------------------------------
    # Fairness policy interface
    # -------------------------------------------------------------------------

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
            self._last_force_decode_reason = ""
            self._force_prefill_override_rid = None
            self._force_prefill_override_uid = None
            return False, self._max_safe_prefill_tokens
        if not (self._has_decode_deadline and (self._max_safe_prefill_tokens or 0) <= 0):
            self._last_force_decode_reason = ""
            self._force_prefill_override_rid = None
            self._force_prefill_override_uid = None
            return False, self._max_safe_prefill_tokens

        # We would normally force decode. But if the earliest prefill deadline
        # is earlier than the earliest decode deadline, AND that request's user
        # is under their fair share, yield to prefill instead.
        if self._earliest_decode_start_deadline is not None and self._waiting_prefill_start_deadline_by_rid:
            # Find the waiting request with the earliest prefill deadline.
            earliest_prefill_rid = min(
                self._waiting_prefill_start_deadline_by_rid,
                key=self._waiting_prefill_start_deadline_by_rid.__getitem__,
            )
            earliest_prefill_deadline = self._waiting_prefill_start_deadline_by_rid[earliest_prefill_rid]
            if earliest_prefill_deadline < self._earliest_decode_start_deadline:
                # Find the request object and its user.
                earliest_prefill_req = None
                earliest_prefill_uid = None
                for req in self._safe_waiting_queue:
                    if req.rid == earliest_prefill_rid:
                        earliest_prefill_req = req
                        earliest_prefill_uid = req.uid
                        break
                if (
                    earliest_prefill_uid is not None
                    and self.user_is_fair_prefill(earliest_prefill_uid, running_batch=running_batch)
                    and not self._reject_based_on_computed_fair_limit(
                        earliest_prefill_uid,
                        len(earliest_prefill_req.origin_input_ids),
                    )
                ):
                        # Prefill deadline is more urgent and the user is under fair share —
                        # do not force decode; let the prefill proceed uncapped (None means
                        # "use normal system budget"). We must not return max_safe_prefill_tokens
                        # here because it is 0 and get_new_prefill_batch would treat that as
                        # "capped to zero" and return None immediately.
                        self._last_force_decode_reason = "prefill_deadline_earlier_than_decode"
                        self._force_prefill_override_rid = earliest_prefill_rid
                        self._force_prefill_override_uid = earliest_prefill_uid
                        self._debug_override_fate = ""
                        return False, None

        # If any waiting request can fit in the KV cache, yield to prefill rather than decode.
        # PREFILL_PRIORITIZE_FAIR controls whether unfair users count here too.
        for req in self._safe_waiting_queue:
            if PREFILL_PRIORITIZE_FAIR and not self.user_is_fair_prefill(req.uid, running_batch=running_batch):
                continue
            if not self._reject_based_on_computed_fair_limit(
                req.uid,
                len(req.origin_input_ids),
            ):
                self._last_force_decode_reason = ""
                self._force_prefill_override_rid = None
                self._force_prefill_override_uid = None
                return False, None

        self._last_force_decode_reason = "force_decode"
        self._force_prefill_override_rid = None
        self._force_prefill_override_uid = None
        return True, 0

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

        tree_cache = self.tree_cache
        all_evicted: List[Req] = []
        safe_prefill_cap = self._max_safe_prefill_tokens or 0
        used_safe_prefill_tokens = 0
        pending_prefill_by_user: Dict[str, int] = {
            uid: sum(tokens) for uid, tokens in token_counters_by_user.items()
        }
        extra_space = 0

        for req in self._forced_prefill_queue:
            if req.rid not in self._forced_prefill_rids:
                continue
            req_prefill_tokens = getattr(req, "extend_input_len", len(req.origin_input_ids))
            if (
                safe_prefill_cap > 0
                and used_safe_prefill_tokens + req_prefill_tokens > safe_prefill_cap
            ):
                break
            extra_sum = pending_prefill_by_user.get(req.uid, 0)
            if not self.user_is_fair_prefill(
                req.uid,
                running_batch=running_batch,
                this_user_len=len(token_counters_by_user.get(req.uid, [])),
                this_user_sum=extra_sum,
            ):
                continue
            if not self._force_prefill_within_user_headroom(
                req,
                running_batch=running_batch,
                pending_prefill_tokens=extra_sum,
            ):
                continue

            total_tokens = len(req.origin_input_ids) + min(
                req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
            )

            # Evict a decode slot if running batch is at capacity.
            if (
                max_running_requests is not None
                and running_batch.batch_size() + len(adder.can_run_list) >= max_running_requests
            ):
                slots_needed = (
                    running_batch.batch_size() + len(adder.can_run_list) - max_running_requests + 1
                )
                try:
                    evicted_for_slots, _ = running_batch.retract_decode_for_slots(slots_needed)
                except RuntimeError as exc:
                    if "Delta fairness retraction blocked" in str(exc):
                        evicted_for_slots = []
                    else:
                        raise
                if not evicted_for_slots:
                    continue
                waiting_queue.extend(evicted_for_slots)
                self.note_retracted_reqs(evicted_for_slots)
                all_evicted.extend(evicted_for_slots)

            # Evict KV tokens until we have enough space for this request.
            sz = token_to_kv_pool.available_size() + tree_cache.evictable_size()
            while sz < total_tokens:
                needed = max(0, total_tokens - sz)
                try:
                    evicted_for_tokens, _ = running_batch.retract_decode(needed)
                except RuntimeError as exc:
                    if "Delta fairness retraction blocked" in str(exc):
                        evicted_for_tokens = []
                    else:
                        raise
                if not evicted_for_tokens:
                    break
                waiting_queue.extend(evicted_for_tokens)
                self.note_retracted_reqs(evicted_for_tokens)
                all_evicted.extend(evicted_for_tokens)
                new_sz = token_to_kv_pool.available_size() + tree_cache.evictable_size()
                adder.expand_capacity(new_sz - sz)
                if max_input_size is not None:
                    adder.rem_input_tokens = min(
                        adder.rem_input_tokens,
                        max(0, max_input_size - adder.log_input_tokens),
                    )
                sz = new_sz

            if sz < total_tokens:
                # Could not free enough space; skip this request.
                continue

            # Init prefix/extend for this request.
            res = req.init_next_round_input(
                None if prefix_computed else tree_cache,
                fairness_policy=self,
                fair=True,
                extra_tokens=extra_sum,
            )
            if res == "rejected":
                continue

            # Evict more if the actual extend_input_len exceeds remaining input budget.
            if max_input_size is not None:
                remaining_input_budget = max(0, max_input_size - adder.log_input_tokens)
                while req.extend_input_len > remaining_input_budget:
                    try:
                        evicted_for_budget, _ = running_batch.retract_decode(
                            req.extend_input_len - remaining_input_budget
                        )
                    except RuntimeError as exc:
                        if "Delta fairness retraction blocked" in str(exc):
                            evicted_for_budget = []
                        else:
                            raise
                    if not evicted_for_budget:
                        break
                    waiting_queue.extend(evicted_for_budget)
                    self.note_retracted_reqs(evicted_for_budget)
                    all_evicted.extend(evicted_for_budget)
                    new_sz = token_to_kv_pool.available_size() + tree_cache.evictable_size()
                    adder.expand_capacity(new_sz - sz)
                    sz = new_sz
                    remaining_input_budget = max(0, max_input_size - adder.log_input_tokens)
                if req.extend_input_len > remaining_input_budget:
                    continue

            new_extra = extra_sum + req.extend_input_len
            token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)
            pending_prefill_by_user[req.uid] = new_extra
            self._ignore_global_prefill_budget = True
            try:
                add_res = adder.add_one_req(req, new_extra)
            finally:
                self._ignore_global_prefill_budget = False
            if add_res == "rejected":
                token_counters_by_user[req.uid].pop()
                pending_prefill_by_user[req.uid] = extra_sum
                continue

            used_safe_prefill_tokens += req_prefill_tokens
            extra_space += total_tokens

        return extra_space, all_evicted or None

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

        # Sort safe_waiting_queue with two-condition strict priority:
        #   Tier 0: fair user + prefill deadline before earliest decode deadline (condition 1)
        #   Tier 1: fair user, any deadline (condition 2)
        #   Tier 2: all others
        # Within each tier, sort by EDF.
        decode_deadline = self._earliest_decode_start_deadline
        fair_uid_cache: Dict[str, bool] = {}
        def _req_is_fair(req: Req) -> bool:
            uid = req.uid
            if uid not in fair_uid_cache:
                fair_uid_cache[uid] = self.user_is_fair_prefill(
                    uid,
                    running_batch=running_batch,
                    this_user_len=len(token_counters_by_user.get(uid, [])),
                    this_user_sum=pending_prefill_by_user.get(uid, 0),
                )
            return fair_uid_cache[uid]

        def _sort_key(r: Req) -> tuple:
            deadline = self._waiting_prefill_start_deadline_by_rid.get(r.rid, float("inf"))
            is_fair = _req_is_fair(r)
            cond1 = is_fair and decode_deadline is not None and deadline < decode_deadline
            tier = 0 if cond1 else (1 if is_fair else 2)
            return (tier, deadline)

        ordered_safe_waiting = sorted(self._safe_waiting_queue, key=_sort_key)

        # no_retraction_budget disabled — was incorrectly blocking unfair users
        # even when there was plenty of physical memory available.
        no_retraction_budget = float("inf")

        for prepared_req in ordered_safe_waiting:
            req = waiting_by_rid.get(prepared_req.rid)
            if req is None:
                if prepared_req.rid == self._force_prefill_override_rid:
                    self._debug_override_fate = "not_in_waiting"
                continue
            if max_input_size is not None and adder.log_input_tokens > max_input_size:
                if req.rid == self._force_prefill_override_rid:
                    self._debug_override_fate = "input_cap"
                break
            if max_input_size is not None:
                adder.rem_input_tokens = max_input_size - adder.log_input_tokens
            if req in adder.can_run_list:
                continue
            # Beyond the no-retraction budget, only fair users are admitted.
            if adder.log_input_tokens > no_retraction_budget and not _req_is_fair(req):
                continue

            extra_tokens = pending_prefill_by_user.get(req.uid, 0)
            is_override = req.rid == self._force_prefill_override_rid
            res = req.init_next_round_input(
                target_tree_cache,
                # Skip KV-limit rejection for the prefill-deadline override request;
                # it gets ignore_global_budget treatment in add_one_req below.
                fairness_policy=None if is_override else self,
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

            is_forced = req.rid in self._forced_prefill_rids or is_override
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
                if is_override:
                    self._debug_override_fate = "adder_rejected"
                continue
            if is_override:
                self._debug_override_fate = "admitted"

            token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)
            pending_prefill_by_user[req.uid] = extra_tokens + req.extend_input_len

            if (
                not add_res
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= effective_running_limit
            ):
                break

        # If the safe_waiting_queue loop admitted nothing (stale snapshot or all KV-rejected),
        # try from the live waiting queue sorted by prefill deadline.
        # Pre-split into fair-only and all lists to avoid expensive match_prefix
        # calls on requests that will just be rejected by fairness checks.
        if not adder.can_run_list:
            _deadline_key = lambda r: self._waiting_prefill_start_deadline_by_rid.get(r.rid, float("inf"))
            _fair_cache: Dict[str, bool] = {}
            live_fair: List[Req] = []
            live_all: List[Req] = []
            for req in waiting_queue:
                uid = req.uid
                if uid not in _fair_cache:
                    _fair_cache[uid] = self.user_is_fair_prefill(uid, running_batch=running_batch)
                if _fair_cache[uid]:
                    live_fair.append(req)
                live_all.append(req)
            live_fair.sort(key=_deadline_key)
            live_all.sort(key=_deadline_key)

            # Try fair users first; fall back to all users only if no fair
            # requests could be admitted (e.g. all fair users already running).
            candidate_lists = [live_fair]
            if not live_fair:
                candidate_lists = [live_all]

            for candidates in candidate_lists:
                for req in candidates:
                    if req in adder.can_run_list:
                        continue
                    if running_batch_size + len(adder.can_run_list) >= effective_running_limit:
                        break
                    if max_input_size is not None and adder.log_input_tokens > max_input_size:
                        break
                    if max_input_size is not None:
                        adder.rem_input_tokens = max_input_size - adder.log_input_tokens
                    extra_tokens = pending_prefill_by_user.get(req.uid, 0)
                    res = req.init_next_round_input(
                        target_tree_cache,
                        fairness_policy=self,
                        fair=_fair_cache.get(req.uid, False),
                        extra_tokens=extra_tokens,
                    )
                    if res == "rejected":
                        continue
                    add_res = adder.add_one_req(req, extra_tokens)
                    if add_res == "rejected":
                        continue
                    token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)
                    pending_prefill_by_user[req.uid] = extra_tokens + req.extend_input_len
                    if not add_res or adder.no_remaining_tokens():
                        break
                if adder.can_run_list:
                    break

    # -------------------------------------------------------------------------
    # Violation checking and GPU event hooks
    # -------------------------------------------------------------------------

    def _mark_violation_if_executed_after_deadline(
        self,
        req: Req,
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        simulator = self.simulator
        tracked = simulator.requests.get(req.rid)
        real_event = simulator.most_recent_event_real.get(req.rid)
        if tracked is None or real_event is None:
            return

        now_ts = time.time() if now is None else now
        matched_event = None
        for event in tracked.timeline.events_after(real_event):
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
            TIMELINE_WRITER.mark_prefill_done(req.rid, req.uid)
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="prefill",
                now=now,
            )

    def finished_decode(self, batch: ScheduleBatch, decode_rounds: int = 1) -> None:
        now = time.time()
        for req in batch.reqs:
            completion_number = len(getattr(req, "output_ids", []))
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="decode",
                completion_number=completion_number,
                now=now,
            )
        super().finished_decode(batch, decode_rounds=decode_rounds)

    # -------------------------------------------------------------------------
    # Async decode epoch (stub — not used in current design)
    # -------------------------------------------------------------------------

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
