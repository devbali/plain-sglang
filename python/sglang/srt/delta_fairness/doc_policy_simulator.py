from __future__ import annotations

"""Independent isolated-setting simulator for the Design.md policy."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from sglang.global_config import global_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.policy_scheduler import CLIP_MAX_NEW_TOKENS
from sglang.srt.request_timeline import TIMELINE_WRITER

from .time_estimation import (
    isolated_decode_time_estimation,
    isolated_prefill_time_estimation,
)


def _iso_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class RequestEvent:
    req_id: str
    duration: float = 0.0
    end_timestamp: float = field(default_factory=time.time)

    def is_logically_after(self, other: "RequestEvent") -> bool:
        return False


class RequestStartEvent(RequestEvent):
    pass


class RequestPrefillEvent(RequestEvent):
    def is_logically_after(self, other: "RequestEvent") -> bool:
        return isinstance(other, RequestStartEvent)


@dataclass
class RequestDecodeEvent(RequestEvent):
    completion_number: int = 0

    def is_logically_after(self, other: "RequestEvent") -> bool:
        if getattr(other, "req_id", None) != self.req_id:
            return False
        if isinstance(other, RequestDecodeEvent):
            return self.completion_number > other.completion_number
        return isinstance(other, (RequestStartEvent, RequestPrefillEvent))


@dataclass
class UserEvent:
    duration: float = 0.0
    end_timestamp: float = field(default_factory=time.time)


class UserPrefillEvent(UserEvent):
    pass


class UserDecodeEvent(UserEvent):
    pass


@dataclass
class RequestTimeline:
    history: List[RequestEvent] = field(default_factory=list)
    anticipated_future_events: List[RequestEvent] = field(default_factory=list)

    def events_after(self, real_event: RequestEvent) -> List[RequestEvent]:
        events = list(self.history) + list(self.anticipated_future_events)
        if not events:
            return []

        match_idx = -1
        if isinstance(real_event, RequestStartEvent):
            for i, event in enumerate(events):
                if isinstance(event, RequestStartEvent):
                    match_idx = i
        elif isinstance(real_event, RequestPrefillEvent):
            for i, event in enumerate(events):
                if isinstance(event, RequestPrefillEvent):
                    match_idx = i
        elif isinstance(real_event, RequestDecodeEvent):
            for i, event in enumerate(events):
                if (
                    isinstance(event, RequestDecodeEvent)
                    and event.completion_number == real_event.completion_number
                ):
                    match_idx = i

        if match_idx < 0:
            return []
        return events[match_idx + 1 :]


@dataclass
class TrackedRequest:
    req: Req
    arrival_timestamp: float
    user_timeline: Optional["UserTimeline"] = None
    deltas_in_microseconds: Dict[str, int] = field(
        default_factory=lambda: {"prefill": 0, "first_decode": 0, "decode": 0}
    )
    alternate_history_timeline: RequestTimeline = field(default_factory=RequestTimeline)

    def most_recent_event(self) -> Optional[RequestEvent]:
        if not self.alternate_history_timeline.history:
            return None
        return self.alternate_history_timeline.history[-1]

    def earliest_events_after_real_time(
        self, real_event: RequestEvent
    ) -> Optional[List[RequestEvent]]:
        events = self.alternate_history_timeline.events_after(real_event)
        if not events:
            return None
        return [events[0]]


def _request_token_count(req: Req, simulated_decode_count: int) -> int:
    return len(req.origin_input_ids) + max(1, simulated_decode_count)


@dataclass
class _SimRequestState:
    tracked: TrackedRequest
    arrival_timestamp: float
    realized_prefill_done: bool = False
    realized_decode_count: int = 0
    prefill_done: bool = False
    simulated_decode_count: int = 0
    anticipated_recorded: bool = False

    def prefill_context_tokens(self) -> int:
        return len(self.tracked.req.origin_input_ids) + self.simulated_decode_count

    def remaining_max_new_tokens(self) -> int:
        sampling_params = getattr(self.tracked.req, "sampling_params", None)
        configured_max_new_tokens = getattr(
            sampling_params, "max_new_tokens", 0
        )
        max_new_tokens = min(
            configured_max_new_tokens,
            CLIP_MAX_NEW_TOKENS,
        )
        return max(0, max_new_tokens - self.simulated_decode_count)


@dataclass
class UserTimeline:
    uid: str
    max_kv_tokens: Optional[int] = None
    fairinf_n: int = 1
    history: List[UserEvent] = field(default_factory=list)
    anticipated_future_events: List[UserEvent] = field(default_factory=list)
    request_timelines: Dict[str, TrackedRequest] = field(default_factory=dict)
    finished_request_timelines: Dict[str, TrackedRequest] = field(default_factory=dict)
    min_new_token_ratio: float = 0.0

    def finished_request(self, req_id: str) -> None:
        tracked = self.request_timelines.pop(req_id, None)
        if tracked is not None:
            self.finished_request_timelines[req_id] = tracked

    def _clear_simulation(self) -> None:
        self.history = []
        self.anticipated_future_events = []
        for tracked in list(self.request_timelines.values()):
            tracked.alternate_history_timeline.history = [
                RequestStartEvent(
                    req_id=tracked.req.rid,
                    end_timestamp=tracked.arrival_timestamp,
                )
            ]
            tracked.alternate_history_timeline.anticipated_future_events = []

    def _target_state_for_request(
        self, tracked: TrackedRequest, real_event: Optional[RequestEvent]
    ) -> _SimRequestState:
        realized_prefill_done = False
        realized_decode_count = 0
        if isinstance(real_event, RequestPrefillEvent):
            realized_prefill_done = True
        elif isinstance(real_event, RequestDecodeEvent):
            realized_prefill_done = True
            realized_decode_count = real_event.completion_number
        return _SimRequestState(
            tracked=tracked,
            arrival_timestamp=tracked.arrival_timestamp,
            realized_prefill_done=realized_prefill_done,
            realized_decode_count=realized_decode_count,
        )

    def _live_states(
        self, req_id_real_statuses: Dict[str, RequestEvent]
    ) -> List[_SimRequestState]:
        states = []
        for tracked in list(self.request_timelines.values()):
            states.append(
                self._target_state_for_request(
                    tracked, req_id_real_statuses.get(tracked.req.rid)
                )
            )
        return sorted(
            states,
            key=lambda state: (state.arrival_timestamp, state.tracked.req.rid),
        )

    def _current_active_kv_tokens(self, active_states: List[_SimRequestState]) -> int:
        return sum(
            _request_token_count(state.tracked.req, state.simulated_decode_count)
            for state in active_states
        )

    def _remaining_decode_reservation_tokens(
        self, active_states: List[_SimRequestState]
    ) -> int:
        ratio = max(0.0, float(self.min_new_token_ratio))
        return int(
            sum(state.remaining_max_new_tokens() * ratio for state in active_states)
        )

    def _build_prefill_batch(
        self,
        waiting_states: List[_SimRequestState],
        active_states: List[_SimRequestState],
        sim_time: float,
    ) -> List[_SimRequestState]:
        ready = [
            state for state in waiting_states if state.arrival_timestamp <= sim_time
        ]
        if not ready:
            return []

        batch: List[_SimRequestState] = []
        current_kv = self._current_active_kv_tokens(active_states)
        remaining_budget = None
        if self.max_kv_tokens is not None:
            remaining_budget = (
                self.max_kv_tokens
                - current_kv
                - self._remaining_decode_reservation_tokens(active_states)
            )
        reserved_for_batch = 0
        for state in ready:
            prefill_tokens = state.prefill_context_tokens()
            total_tokens = prefill_tokens + state.remaining_max_new_tokens()
            projected_kv = current_kv + sum(
                batch_state.prefill_context_tokens() for batch_state in batch
            ) + prefill_tokens
            if self.max_kv_tokens is not None and projected_kv > self.max_kv_tokens:
                break
            if remaining_budget is not None and total_tokens > remaining_budget - reserved_for_batch:
                break
            batch.append(state)
            reserved_for_batch += total_tokens
        return batch

    def _isolated_retract_decode(
        self,
        waiting_states: List[_SimRequestState],
        active_states: List[_SimRequestState],
        *,
        extra_decode_tokens: int,
    ) -> bool:
        if self.max_kv_tokens is None:
            return True

        def active_kv() -> int:
            return self._current_active_kv_tokens(active_states)

        sorted_states = list(active_states)
        sorted_states.sort(
            key=lambda state: (
                len(state.tracked.req.output_ids),
                -len(state.tracked.req.origin_input_ids),
            ),
            reverse=True,
        )

        while sorted_states and active_kv() + len(active_states) + extra_decode_tokens > self.max_kv_tokens:
            if len(sorted_states) == 1 and active_kv() > 0:
                break
            state = sorted_states.pop()
            if state in active_states:
                active_states.remove(state)
                state.prefill_done = False
                state.anticipated_recorded = False
                state.tracked.alternate_history_timeline.anticipated_future_events = []
                waiting_states.insert(0, state)

        return active_kv() + len(active_states) + extra_decode_tokens <= self.max_kv_tokens

    def rebuild_from_real_state(
        self, req_id_real_statuses: Dict[str, RequestEvent]
    ) -> None:
        self._clear_simulation()

        live_states = self._live_states(req_id_real_statuses)
        if not live_states:
            return

        waiting_states = list(live_states)
        active_states: List[_SimRequestState] = []
        sim_time = min(state.arrival_timestamp for state in live_states)

        max_steps = sum(
            max(1, state.realized_decode_count + 2) for state in live_states
        ) + len(live_states) + 8

        for _ in range(max_steps):
            if all(state.anticipated_recorded for state in live_states):
                break

            batch = self._build_prefill_batch(waiting_states, active_states, sim_time)
            if batch:
                prompt_sizes = [state.prefill_context_tokens() for state in batch]
                duration = isolated_prefill_time_estimation(
                    sum(prompt_sizes),
                    max(prompt_sizes),
                    len(prompt_sizes),
                    self.fairinf_n,
                )
                sim_time += duration
                self.history.append(
                    UserPrefillEvent(duration=duration, end_timestamp=sim_time)
                )
                for state in batch:
                    state.prefill_done = True
                    event = RequestPrefillEvent(
                        req_id=state.tracked.req.rid,
                        duration=duration,
                        end_timestamp=sim_time,
                    )
                    has_prior_prefill = any(
                        isinstance(history_event, RequestPrefillEvent)
                        for history_event in state.tracked.alternate_history_timeline.history
                    )
                    if state.realized_prefill_done:
                        state.tracked.alternate_history_timeline.history.append(event)
                        if not has_prior_prefill:
                            TIMELINE_WRITER.mark_isolated_prefill_done(
                                state.tracked.req.rid,
                                state.tracked.req.uid,
                                timestamp_iso=_iso_ts(sim_time),
                            )
                    else:
                        state.tracked.alternate_history_timeline.anticipated_future_events = [
                            event
                        ]
                        state.anticipated_recorded = True
                    waiting_states.remove(state)
                    active_states.append(state)
                continue

            if active_states:
                if not self._isolated_retract_decode(
                    waiting_states,
                    active_states,
                    extra_decode_tokens=0,
                ):
                    break
                if not active_states:
                    continue
                token_counts = [
                    _request_token_count(state.tracked.req, state.simulated_decode_count)
                    for state in active_states
                ]
                duration = isolated_decode_time_estimation(
                    sum(token_counts),
                    max(token_counts),
                    len(token_counts),
                    self.fairinf_n,
                )
                next_arrival = min(
                    (
                        state.arrival_timestamp
                        for state in waiting_states
                        if state.arrival_timestamp > sim_time
                    ),
                    default=None,
                )
                if next_arrival is not None and sim_time + duration > next_arrival:
                    sim_time = next_arrival
                    continue

                milestone_rounds = [
                    (
                        state.realized_decode_count - state.simulated_decode_count
                        if state.simulated_decode_count < state.realized_decode_count
                        else 1
                    )
                    for state in active_states
                    if not state.anticipated_recorded
                ]
                if not milestone_rounds:
                    break

                rounds = max(1, min(milestone_rounds))
                if next_arrival is not None:
                    rounds_until_arrival = int((next_arrival - sim_time) // duration)
                    if rounds_until_arrival <= 0:
                        sim_time = next_arrival
                        continue
                    rounds = min(rounds, rounds_until_arrival)

                chunk_start = sim_time
                sim_time += rounds * duration
                self.history.append(
                    UserDecodeEvent(duration=rounds * duration, end_timestamp=sim_time)
                )
                for state in active_states:
                    prev_decode_count = state.simulated_decode_count
                    state.simulated_decode_count += rounds

                    if (
                        prev_decode_count < state.realized_decode_count
                        <= state.simulated_decode_count
                    ):
                        realized_round = state.realized_decode_count - prev_decode_count
                        realized_ts = chunk_start + realized_round * duration
                        realized_event = RequestDecodeEvent(
                            req_id=state.tracked.req.rid,
                            duration=duration,
                            end_timestamp=realized_ts,
                            completion_number=state.realized_decode_count,
                        )
                        state.tracked.alternate_history_timeline.history.append(
                            realized_event
                        )
                        TIMELINE_WRITER.mark_isolated_decode_done(
                            state.tracked.req.rid,
                            state.tracked.req.uid,
                            timestamp_iso=_iso_ts(realized_ts),
                            completion_number=state.realized_decode_count,
                        )

                    anticipated_completion = state.realized_decode_count + 1
                    if (
                        not state.anticipated_recorded
                        and prev_decode_count < anticipated_completion
                        <= state.simulated_decode_count
                    ):
                        anticipated_round = anticipated_completion - prev_decode_count
                        anticipated_ts = chunk_start + anticipated_round * duration
                        anticipated_event = RequestDecodeEvent(
                            req_id=state.tracked.req.rid,
                            duration=duration,
                            end_timestamp=anticipated_ts,
                            completion_number=anticipated_completion,
                        )
                        state.tracked.alternate_history_timeline.anticipated_future_events = [
                            anticipated_event
                        ]
                        state.anticipated_recorded = True
                continue

            next_arrival = min(
                (
                    state.arrival_timestamp
                    for state in waiting_states
                    if state.arrival_timestamp > sim_time
                ),
                default=None,
            )
            if next_arrival is None:
                break
            sim_time = next_arrival


@dataclass
class DeadlineCandidate:
    deadline: float
    start_deadline: float
    event_type: str
    req: Req
    event: RequestEvent


class AlternateHistorySimulator:
    def __init__(
        self,
        *,
        max_kv_tokens_per_user: Optional[int] = None,
        fairinf_n: int = 1,
        min_new_token_ratio: float = 0.0,
    ):
        self.max_kv_tokens_per_user = max_kv_tokens_per_user
        self.fairinf_n = max(int(fairinf_n), 1)
        self.min_new_token_ratio = max(0.0, float(min_new_token_ratio))
        self.users: Dict[str, UserTimeline] = {}
        self.requests: Dict[str, TrackedRequest] = {}
        self.most_recent_event_real: Dict[str, RequestEvent] = {}

    def _make_user_timeline(self, uid: str) -> UserTimeline:
        return UserTimeline(
            uid=uid,
            max_kv_tokens=self.max_kv_tokens_per_user,
            fairinf_n=self.fairinf_n,
            min_new_token_ratio=self.min_new_token_ratio,
        )

    def _ensure_tracked_request(
        self, req: Req, deltas_in_microseconds: Optional[Dict[str, int]]
    ) -> TrackedRequest:
        tracked = self.requests.get(req.rid)
        if tracked is None:
            arrival = self.most_recent_event_real.get(req.rid)
            tracked = TrackedRequest(
                req=req,
                arrival_timestamp=getattr(arrival, "end_timestamp", time.time()),
                deltas_in_microseconds=dict(
                    deltas_in_microseconds
                    or {"prefill": 0, "first_decode": 0, "decode": 0}
                ),
            )
            self.requests[req.rid] = tracked
        else:
            tracked.req = req
            if deltas_in_microseconds is not None:
                tracked.deltas_in_microseconds = dict(deltas_in_microseconds)
        return tracked

    def _track_request(
        self,
        req: Req,
        tracked: TrackedRequest,
        user_timeline: UserTimeline,
    ) -> None:
        tracked.user_timeline = user_timeline
        user_timeline.request_timelines[req.rid] = tracked

    def sync_fair_user_tracking(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        user_is_fair: Callable[[str, Optional[ScheduleBatch]], bool],
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        running_reqs = list(running_batch.reqs) if running_batch is not None else []
        live_by_user: Dict[str, List[Req]] = {}
        for req in waiting_queue:
            live_by_user.setdefault(req.uid, []).append(req)
        for req in running_reqs:
            live_by_user.setdefault(req.uid, []).append(req)

        fair_users = sorted(
            set(self.users)
            | {
                uid
                for uid in set(live_by_user)
                if user_is_fair(uid, running_batch)
            }
        )

        for uid in fair_users:
            user_timeline = self.users.get(uid)
            if user_timeline is None:
                user_timeline = self._make_user_timeline(uid)
                self.users[uid] = user_timeline

            for req in live_by_user.get(uid, []):
                tracked = self._ensure_tracked_request(req, deltas_in_microseconds)
                self._track_request(req, tracked, user_timeline)

        for uid in list(self.users.keys()):
            user_timeline = self.users[uid]
            if uid in fair_users and user_timeline.request_timelines:
                continue
            self.users.pop(uid, None)
            for rid, tracked in list(self.requests.items()):
                if tracked.req.uid == uid:
                    self.requests.pop(rid, None)
                    self.most_recent_event_real.pop(rid, None)

        return fair_users

    def start_of_pass(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        user_is_fair: Callable[[str, Optional[ScheduleBatch]], bool],
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        pass_start = time.perf_counter()
        fair_users = self.sync_fair_user_tracking(
            running_batch,
            waiting_queue,
            user_is_fair=user_is_fair,
            deltas_in_microseconds=deltas_in_microseconds,
        )
        after_sync = time.perf_counter()
        for uid in fair_users:
            user_timeline = self.users.get(uid)
            if user_timeline is not None:
                user_timeline.rebuild_from_real_state(self.most_recent_event_real)
        after_rebuild = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["sync_fair_user_tracking_ms"] = (
                after_sync - pass_start
            ) * 1000.0
            timing_breakdown["rebuild_from_real_state_ms"] = (
                after_rebuild - after_sync
            ) * 1000.0
            timing_breakdown["simulator_start_of_pass_ms"] = (
                after_rebuild - pass_start
            ) * 1000.0
        return fair_users

    def build_deadline_candidates(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        req_is_fair_prefill: Callable[[Req, Optional[ScheduleBatch]], bool],
        req_is_fair_decode: Callable[[Req, Optional[ScheduleBatch]], bool],
        event_delta_seconds: Callable[[TrackedRequest, RequestEvent], float],
        pooled_prefill_estimate_seconds: Callable[[Req], float],
        pooled_decode_estimate_seconds: Callable[[Req, Optional[ScheduleBatch]], float],
    ) -> Tuple[List[DeadlineCandidate], Dict[str, float]]:
        waiting_by_rid = {req.rid: req for req in waiting_queue}
        running_by_rid = {
            req.rid: req
            for req in (running_batch.reqs if running_batch is not None else [])
        }
        candidates: List[DeadlineCandidate] = []
        waiting_prefill_deadline_by_rid: Dict[str, float] = {}

        for rid, tracked in self.requests.items():
            real_event = self.most_recent_event_real.get(rid)
            if real_event is None:
                continue

            upcoming_events = tracked.earliest_events_after_real_time(real_event) or []
            req = tracked.req
            for event in upcoming_events:
                deadline = event.end_timestamp + event_delta_seconds(tracked, event)
                if isinstance(event, RequestPrefillEvent):
                    if rid not in waiting_by_rid:
                        continue
                    if not req_is_fair_prefill(req, running_batch):
                        continue
                    start_deadline = deadline - pooled_prefill_estimate_seconds(req)
                    candidates.append(
                        DeadlineCandidate(
                            deadline=deadline,
                            start_deadline=start_deadline,
                            event_type="prefill",
                            req=req,
                            event=event,
                        )
                    )
                    waiting_prefill_deadline_by_rid[rid] = start_deadline
                elif isinstance(event, RequestDecodeEvent):
                    if rid not in running_by_rid:
                        continue
                    if not req_is_fair_decode(req, running_batch):
                        continue
                    start_deadline = deadline - pooled_decode_estimate_seconds(
                        req, running_batch
                    )
                    candidates.append(
                        DeadlineCandidate(
                            deadline=deadline,
                            start_deadline=start_deadline,
                            event_type="decode",
                            req=req,
                            event=event,
                        )
                    )

        candidates.sort(
            key=lambda candidate: (
                candidate.start_deadline,
                0 if candidate.event_type == "decode" else 1,
                candidate.deadline,
            )
        )
        return candidates, waiting_prefill_deadline_by_rid

    def process_new_request(
        self, req: Req, deltas_in_microseconds: Optional[Dict[str, int]] = None
    ) -> None:
        now = time.time()
        self.most_recent_event_real[req.rid] = RequestStartEvent(
            req_id=req.rid,
            end_timestamp=now,
        )
        tracked = self.requests.get(req.rid)
        if tracked is None:
            tracked = TrackedRequest(
                req=req,
                arrival_timestamp=now,
                deltas_in_microseconds=dict(
                    deltas_in_microseconds
                    or {"prefill": 0, "first_decode": 0, "decode": 0}
                ),
            )
            self.requests[req.rid] = tracked
        else:
            tracked.arrival_timestamp = now
            tracked.req = req
            if deltas_in_microseconds is not None:
                tracked.deltas_in_microseconds = dict(deltas_in_microseconds)

    def finished_prefill(self, batch: ScheduleBatch) -> None:
        now = time.time()
        for req in batch.reqs:
            self.most_recent_event_real[req.rid] = RequestPrefillEvent(
                req_id=req.rid,
                end_timestamp=now,
            )

    def finished_decode(self, batch: ScheduleBatch) -> None:
        now = time.time()
        for req in batch.reqs:
            self.most_recent_event_real[req.rid] = RequestDecodeEvent(
                req_id=req.rid,
                end_timestamp=now,
                completion_number=len(req.output_ids),
            )

    def mark_request_finished(self, req: Req) -> None:
        user_timeline = self.users.get(req.uid)
        if user_timeline is not None:
            user_timeline.finished_request(req.rid)
        self.requests.pop(req.rid, None)
        self.most_recent_event_real.pop(req.rid, None)
