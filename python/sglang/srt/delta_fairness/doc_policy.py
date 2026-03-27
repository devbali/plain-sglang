from __future__ import annotations

"""Design.md policy implementation."""

import logging
import time
from copy import copy
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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

    def _invalidate_prepared_state(self) -> None:
        self._simulator_rebuild_prepared = False
        self._prepared_deadline_queue = []
        self._prepared_waiting_prefill_start_deadline_by_rid = {}
        self._prepared_safe_waiting_queue = []
        self._prepared_safe_waiting_rids = set()
        self._prepared_forced_prefill_queue = []
        self._prepared_forced_prefill_rids = set()
        self._prepared_max_safe_prefill_tokens = None
        self._prepared_has_fair_waiting = False
        self._prepared_has_decode_deadline = False
        self._prepared_earliest_decode_start_deadline = None
        self._prepared_safe_prefix_now = None
        self._prepared_running_sig = None
        self._prepared_waiting_sig = None
        self._prepared_pass_state = None

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
        return super().user_is_fair_prefill(
            user_id,
            running_batch=None,
            this_user_len=this_user_len,
            this_user_sum=this_user_sum,
        )

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

        earliest_decode_deadline = min(
            (
                candidate.start_deadline
                for candidate in self._deadline_queue
                if candidate.event_type == "decode"
            ),
            default=None,
        )
        if earliest_decode_deadline is None:
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
        self._earliest_decode_start_deadline = earliest_decode_deadline

        now = time.time()
        self._safe_prefix_now = now
        candidate_batch: List[Req] = []
        safe_prompt_tokens = 0
        for req in self._safe_waiting_queue:
            candidate_batch.append(req)
            pooled_prefill_s = self._pooled_prefill_seconds(candidate_batch)
            if now + pooled_prefill_s <= earliest_decode_deadline:
                safe_prompt_tokens = sum(
                    len(batch_req.origin_input_ids) for batch_req in candidate_batch
                )
                self._forced_prefill_queue.append(req)
                self._forced_prefill_rids.add(req.rid)
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
        self._deadline_queue, self._waiting_prefill_start_deadline_by_rid = (
            self.simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: self.req_is_fair_prefill(
                    req, running_batch=rb
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
        )
        after_deadline_build = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["build_deadline_candidates_ms"] = (
                after_deadline_build - phase_start
            ) * 1000.0
        self._recompute_safe_prefix(
            waiting_queue,
            running_batch,
            timing_breakdown=timing_breakdown,
        )

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
        self._prepared_deadline_queue, self._prepared_waiting_prefill_start_deadline_by_rid = (
            target_simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: self.req_is_fair_prefill(
                    req, running_batch=rb
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
        )
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
        finished_rids = set(self._pending_finished_rids)
        scheduled_prefill_rids = set(self._pending_scheduled_prefill_reqs)
        self._deadline_queue = [
            candidate
            for candidate in self._deadline_queue
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
            new_candidates, new_waiting_deadlines = self.simulator.build_deadline_candidates(
                self._pending_new_requests,
                running_batch,
                req_is_fair_prefill=lambda req, rb: self.req_is_fair_prefill(
                    req, running_batch=rb
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
            rebuild_breakdown: Dict[str, float] = {}
            self.simulator.rebuild_all_tracked_requests(
                affected_user_ids,
                timing_breakdown=rebuild_breakdown,
            )
            deadline_start = time.perf_counter()
            deadline_queue, waiting_prefill_start_deadline_by_rid = (
                self.simulator.build_deadline_candidates(
                    predicted_waiting,
                    predicted_running,
                    req_is_fair_prefill=lambda req, rb: self.req_is_fair_prefill(
                        req, running_batch=rb
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
            )
            deadline_elapsed_ms = (time.perf_counter() - deadline_start) * 1000.0
            result = {
                "deadline_queue": deadline_queue,
                "waiting_prefill_start_deadline_by_rid": waiting_prefill_start_deadline_by_rid,
                "waiting_sig": self._queue_sig(predicted_waiting),
                "running_sig": self._running_sig(predicted_running),
                "breakdown": {
                    "sync_live_user_tracking_ms": 0.0,
                    "logical_event_update_ms": 0.0,
                    "rebuild_from_real_state_ms": rebuild_breakdown.get(
                        "rebuild_scheduler_loop_ms", 0.0
                    )
                    + rebuild_breakdown.get("rebuild_live_states_ms", 0.0)
                    + rebuild_breakdown.get("rebuild_state_setup_ms", 0.0),
                    "rebuild_live_states_ms": rebuild_breakdown.get(
                        "rebuild_live_states_ms", 0.0
                    ),
                    "rebuild_state_setup_ms": rebuild_breakdown.get(
                        "rebuild_state_setup_ms", 0.0
                    ),
                    "rebuild_scheduler_loop_ms": rebuild_breakdown.get(
                        "rebuild_scheduler_loop_ms", 0.0
                    ),
                    "rebuild_scheduler_step_count": rebuild_breakdown.get(
                        "rebuild_scheduler_step_count", 0
                    ),
                    "rebuild_prefill_step_count": rebuild_breakdown.get(
                        "rebuild_prefill_step_count", 0
                    ),
                    "rebuild_decode_step_count": rebuild_breakdown.get(
                        "rebuild_decode_step_count", 0
                    ),
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
        if not self._simulator_rebuild_prepared:
            return False
        if self._prepared_waiting_sig != self._queue_sig(waiting_queue):
            self._simulator_rebuild_prepared = False
            return False
        if self._prepared_running_sig != self._running_sig(running_batch):
            self._simulator_rebuild_prepared = False
            return False
        self._deadline_queue = list(self._prepared_deadline_queue)
        self._waiting_prefill_start_deadline_by_rid = dict(
            self._prepared_waiting_prefill_start_deadline_by_rid
        )
        self._safe_waiting_queue = list(self._prepared_safe_waiting_queue)
        self._safe_waiting_rids = set(self._prepared_safe_waiting_rids)
        self._forced_prefill_queue = list(self._prepared_forced_prefill_queue)
        self._forced_prefill_rids = set(self._prepared_forced_prefill_rids)
        self._max_safe_prefill_tokens = self._prepared_max_safe_prefill_tokens
        self._has_fair_waiting = self._prepared_has_fair_waiting
        self._has_decode_deadline = self._prepared_has_decode_deadline
        self._earliest_decode_start_deadline = (
            self._prepared_earliest_decode_start_deadline
        )
        self._safe_prefix_now = self._prepared_safe_prefix_now
        self._simulator_rebuild_prepared = False
        self._prepared_pass_state = None
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
        if (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
            and not self._has_pending_mutations()
        ):
            self._last_pass_state_source = "ensure_current_reuse"
            return
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "ensure_consume_prepared"
        elif self._has_pending_mutations():
            self._merge_pending_pass_state_mutations(waiting_queue, running_batch)
            self._last_pass_state_source = "ensure_merge_pending"
        else:
            self._forced_prefill_queue = []
            self._forced_prefill_rids = set()
            self._max_safe_prefill_tokens = 0
            self._has_fair_waiting = False
            self._has_decode_deadline = False
            self._earliest_decode_start_deadline = None
            self._safe_prefix_now = None
            self._last_pass_state_source = "ensure_no_prepared"
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig

    def start_of_pass(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        new_token_ratio: float = 0.0,
        max_running_requests: Optional[int] = None,
    ) -> None:
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
        have_current_state = bool(self._deadline_queue) or bool(
            self._waiting_prefill_start_deadline_by_rid
        )

        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "start_of_pass_consume_prepared"
        elif (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
            and not self._has_pending_mutations()
        ):
            self._last_pass_state_source = "start_of_pass_current_reuse"
        elif have_current_state:
            self._merge_pending_pass_state_mutations(waiting_queue, running_batch)
            self._last_pass_state_source = "start_of_pass_merge_pending"
        else:
            self.simulator.start_of_pass(
                running_batch,
                waiting_queue,
                deltas_in_microseconds=self._deltas_us,
                timing_breakdown=breakdown,
            )
            self._build_pass_state(
                waiting_queue,
                running_batch,
                self._deltas_us,
                timing_breakdown=breakdown,
            )
            self._last_pass_state_source = "start_of_pass_initial_build"

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
        for step in range(1, decode_steps + 1):
            for req in running_batch.reqs:
                if chosen is not None and req.rid not in chosen:
                    continue
                completion_number = len(req.output_ids) - decode_steps + step
                logical_ts = self._logical_next_event_timestamp(
                    req,
                    event_type="decode",
                    completion_number=completion_number,
                    simulator=simulator,
                )
                simulator.most_recent_event_real[req.rid] = RequestDecodeEvent(
                    req_id=req.rid,
                    end_timestamp=logical_ts,
                    completion_number=completion_number,
                )

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
    ) -> None:
        del selected_rids
        if not prepare_pass_state:
            self._invalidate_prepared_state()
            self._last_prepare_breakdown_ms = {
                "sync_live_user_tracking_ms": 0.0,
                "logical_event_update_ms": 0.0,
                "rebuild_from_real_state_ms": 0.0,
                "build_deadline_candidates_ms": 0.0,
                "prepare_during_gpu_execution_total_ms": 0.0,
            }
            return

        if event_type == "decode" and not waiting_queue:
            self._invalidate_prepared_state()
            self._last_prepare_breakdown_ms = {
                "sync_live_user_tracking_ms": 0.0,
                "logical_event_update_ms": 0.0,
                "rebuild_from_real_state_ms": 0.0,
                "build_deadline_candidates_ms": 0.0,
                "prepare_during_gpu_execution_total_ms": 0.0,
            }
            return

        phase_start = time.perf_counter()
        if event_type == "prefill" and scheduled_batch is not None:
            scheduled_rids = {req.rid for req in scheduled_batch.reqs}
            predicted_waiting = [
                req for req in waiting_queue if req.rid not in scheduled_rids
            ]
            predicted_running = self._predicted_running_batch(
                running_batch, scheduled_batch
            )
        else:
            predicted_waiting = list(waiting_queue)
            predicted_running = running_batch

        self._prepare_deadline_state(predicted_waiting, predicted_running)
        self._simulator_rebuild_prepared = True
        elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        self._last_prepare_breakdown_ms = {
            "sync_live_user_tracking_ms": 0.0,
            "logical_event_update_ms": 0.0,
            "rebuild_from_real_state_ms": 0.0,
            "build_deadline_candidates_ms": elapsed_ms,
            "prepare_during_gpu_execution_total_ms": elapsed_ms,
        }

    def process_new_request(self, req: Req) -> None:
        super().process_new_request(req)
        self.simulator.process_new_request(req, self._deltas_us)
        self._invalidate_prepared_state()
        self._pending_new_requests.append(req)

    def note_scheduled_prefill_batch(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            self._pending_scheduled_prefill_reqs[req.rid] = req
        self._invalidate_prepared_state()

    def note_retracted_reqs(self, reqs) -> None:
        for req in reqs:
            self._pending_new_requests.append(req)
        self._invalidate_prepared_state()

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
        return self.req_is_fair_prefill(
            req,
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
        self._ensure_current_pass_state(
            list(waiting_queue),
            running_batch,
            delta_fairness_deltas_microseconds,
        )
        if not self._forced_prefill_rids:
            return 0, None
        prioritized_waiting = [
            req for req in self._forced_prefill_queue if req.rid in self._forced_prefill_rids
        ]
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
        self._invalidate_prepared_state()

    def finished_decode(self, batch: ScheduleBatch) -> None:
        super().finished_decode(batch)
        now = time.time()
        for req in batch.reqs:
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="decode",
                completion_number=len(req.output_ids),
                now=now,
            )
        self.simulator.finished_decode(batch)
        for req in batch.reqs:
            self._pending_decoded_reqs[req.rid] = req
        self._invalidate_prepared_state()

    def mark_request_finished(self, req: Req) -> None:
        super().mark_request_finished(req)
        self.simulator.mark_request_finished(req)
        self._pending_finished_rids.add(req.rid)
        self._pending_decoded_reqs.pop(req.rid, None)
        self._pending_finished_prefill_reqs.pop(req.rid, None)
        self._pending_scheduled_prefill_reqs.pop(req.rid, None)
        self._invalidate_prepared_state()
