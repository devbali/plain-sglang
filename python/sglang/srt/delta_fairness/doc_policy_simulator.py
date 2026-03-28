from __future__ import annotations

"""Independent isolated-setting simulator for the Design.md policy."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

from sglang.global_config import global_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.policy_scheduler import CLIP_MAX_NEW_TOKENS
from sglang.srt.request_timeline import TIMELINE_WRITER

from .time_estimation import (
    isolated_decode_time_estimation,
    isolated_prefill_time_estimation,
)

RETRACTION_PENALTY_SECONDS = 0.030


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
    persistent_prefill_done: bool = False
    persistent_prefill_event: Optional[RequestPrefillEvent] = None
    persistent_decode_count: int = 0
    persistent_decode_events: Dict[int, RequestDecodeEvent] = field(default_factory=dict)
    live_in_running: bool = False
    restart_pending: bool = False
    latest_simulated_completion_timestamp: Optional[float] = None

    def most_recent_event(self) -> Optional[RequestEvent]:
        if self.persistent_decode_count > 0:
            return self.persistent_decode_events.get(self.persistent_decode_count)
        if self.persistent_prefill_event is not None:
            return self.persistent_prefill_event
        if not self.alternate_history_timeline.history:
            return None
        return self.alternate_history_timeline.history[-1]

    def reset_history_to_start(self) -> None:
        start_event = RequestStartEvent(
            req_id=self.req.rid,
            end_timestamp=self.arrival_timestamp,
        )
        self.alternate_history_timeline.history = [start_event]
        self.persistent_prefill_done = False
        self.persistent_prefill_event = None
        self.persistent_decode_count = 0
        self.persistent_decode_events = {}
        self.latest_simulated_completion_timestamp = None

    def append_persistent_event(self, event: RequestEvent) -> None:
        self.alternate_history_timeline.history.append(event)
        if isinstance(event, RequestPrefillEvent):
            self.persistent_prefill_done = True
            self.persistent_prefill_event = event
        elif isinstance(event, RequestDecodeEvent):
            self.persistent_prefill_done = True
            self.persistent_decode_count = max(
                self.persistent_decode_count, event.completion_number
            )
            self.persistent_decode_events[event.completion_number] = event

    def earliest_events_after_real_time(
        self, real_event: RequestEvent
    ) -> Optional[List[RequestEvent]]:
        def _not_before_real(event: RequestEvent) -> bool:
            return float(event.end_timestamp) >= float(real_event.end_timestamp)

        anticipated = self.alternate_history_timeline.anticipated_future_events
        if isinstance(real_event, RequestStartEvent):
            if self.persistent_prefill_event is not None:
                if _not_before_real(self.persistent_prefill_event):
                    return [self.persistent_prefill_event]
            filtered = [event for event in anticipated if _not_before_real(event)]
            return filtered or None
        if isinstance(real_event, RequestPrefillEvent):
            next_decode = self.persistent_decode_events.get(1)
            if next_decode is not None and _not_before_real(next_decode):
                return [next_decode]
            filtered = [event for event in anticipated if _not_before_real(event)]
            return filtered or None
        if isinstance(real_event, RequestDecodeEvent):
            next_decode = self.persistent_decode_events.get(
                real_event.completion_number + 1
            )
            if next_decode is not None and _not_before_real(next_decode):
                return [next_decode]
            filtered = [event for event in anticipated if _not_before_real(event)]
            return filtered or None
        filtered = [event for event in anticipated if _not_before_real(event)]
        return filtered or None

@dataclass
class _SimRequestState:
    tracked: TrackedRequest
    arrival_timestamp: float
    realized_prefill_done: bool = False
    realized_decode_count: int = 0
    prefill_done: bool = False
    simulated_decode_count: int = 0
    anticipated_recorded: bool = False
    prompt_tokens: int = 0
    configured_max_new_tokens: int = 0
    current_token_count: int = 0

    def __post_init__(self) -> None:
        self.prompt_tokens = len(self.tracked.req.origin_input_ids)
        sampling_params = getattr(self.tracked.req, "sampling_params", None)
        configured_max_new_tokens = getattr(
            sampling_params, "max_new_tokens", 0
        )
        self.configured_max_new_tokens = min(
            configured_max_new_tokens,
            CLIP_MAX_NEW_TOKENS,
        )
        self.current_token_count = self.prompt_tokens + max(
            1, self.simulated_decode_count
        )

    def prefill_context_tokens(self) -> int:
        return self.prompt_tokens + self.simulated_decode_count

    def remaining_max_new_tokens(self) -> int:
        return max(0, self.configured_max_new_tokens - self.simulated_decode_count)

    def set_simulated_decode_count(self, decode_count: int) -> None:
        self.simulated_decode_count = int(decode_count)
        self.current_token_count = self.prompt_tokens + max(
            1, self.simulated_decode_count
        )

    def advance_simulated_decode_count(self, rounds: int) -> None:
        if rounds <= 0:
            return
        self.simulated_decode_count += int(rounds)
        self.current_token_count += int(rounds)


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
    cached_live_rids: Tuple[str, ...] = field(default_factory=tuple)
    cached_history_end_timestamp: Optional[float] = None
    cached_state_by_rid: Dict[str, Tuple[bool, int]] = field(default_factory=dict)
    cached_anticipated_event_by_rid: Dict[str, RequestEvent] = field(default_factory=dict)

    def finished_request(self, req_id: str) -> None:
        tracked = self.request_timelines.pop(req_id, None)
        if tracked is not None:
            self.finished_request_timelines[req_id] = tracked

    def _persistent_request_state(self, tracked: TrackedRequest) -> Tuple[bool, int]:
        return tracked.persistent_prefill_done, tracked.persistent_decode_count

    def _merge_real_event_into_request_history(
        self, tracked: TrackedRequest, real_event: Optional[RequestEvent]
    ) -> None:
        timeline = tracked.alternate_history_timeline
        if not timeline.history:
            tracked.reset_history_to_start()
        if real_event is None:
            return

        if (
            isinstance(real_event, (RequestPrefillEvent, RequestDecodeEvent))
            and not tracked.persistent_prefill_done
        ):
            prefill_event = next(
                (
                    event
                    for event in timeline.anticipated_future_events
                    if isinstance(event, RequestPrefillEvent)
                ),
                None,
            )
            if prefill_event is None:
                start_ts = timeline.history[-1].end_timestamp
                context_tokens = len(tracked.req.origin_input_ids)
                prefill_duration = isolated_prefill_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                prefill_event = RequestPrefillEvent(
                    req_id=tracked.req.rid,
                    duration=prefill_duration,
                    end_timestamp=start_ts + prefill_duration,
                )
            tracked.append_persistent_event(prefill_event)
            timeline.anticipated_future_events = [
                event
                for event in timeline.anticipated_future_events
                if not isinstance(event, RequestPrefillEvent)
            ]

        if not isinstance(real_event, RequestDecodeEvent):
            return

        while tracked.persistent_decode_count < real_event.completion_number:
            next_completion = tracked.persistent_decode_count + 1
            decode_event = next(
                (
                    event
                    for event in timeline.anticipated_future_events
                    if isinstance(event, RequestDecodeEvent)
                    and event.completion_number == next_completion
                ),
                None,
            )
            if decode_event is None:
                context_tokens = len(tracked.req.origin_input_ids) + next_completion
                decode_duration = isolated_decode_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                decode_event = RequestDecodeEvent(
                    req_id=tracked.req.rid,
                    duration=decode_duration,
                    end_timestamp=tracked.most_recent_event().end_timestamp + decode_duration,
                    completion_number=next_completion,
                )
            tracked.append_persistent_event(decode_event)
            timeline.anticipated_future_events = [
                event
                for event in timeline.anticipated_future_events
                if not (
                    isinstance(event, RequestDecodeEvent)
                    and event.completion_number == next_completion
                )
            ]

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
        self._merge_real_event_into_request_history(tracked, real_event)
        prefill_done, simulated_decode_count = self._persistent_request_state(tracked)
        return _SimRequestState(
            tracked=tracked,
            arrival_timestamp=tracked.arrival_timestamp,
            realized_prefill_done=realized_prefill_done,
            realized_decode_count=realized_decode_count,
            prefill_done=prefill_done,
            simulated_decode_count=simulated_decode_count,
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
        return sum(state.current_token_count for state in active_states)

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
        future_history: List[UserEvent],
    ) -> List[_SimRequestState]:
        current_time = self._current_history_time(
            waiting_states, active_states, future_history
        )
        ready = [
            state for state in waiting_states if state.arrival_timestamp <= current_time
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
        batch_prefill_tokens = 0
        for state in ready:
            prefill_tokens = state.prefill_context_tokens()
            total_tokens = prefill_tokens + state.remaining_max_new_tokens()
            projected_kv = current_kv + batch_prefill_tokens + prefill_tokens
            if self.max_kv_tokens is not None and projected_kv > self.max_kv_tokens:
                break
            if remaining_budget is not None and total_tokens > remaining_budget - reserved_for_batch:
                break
            batch.append(state)
            reserved_for_batch += total_tokens
            batch_prefill_tokens += prefill_tokens
        return batch

    def _current_history_time(
        self,
        waiting_states: List[_SimRequestState],
        active_states: List[_SimRequestState],
        future_history: List[UserEvent],
    ) -> float:
        if future_history:
            return future_history[-1].end_timestamp
        if self.history:
            return self.history[-1].end_timestamp
        persistent_event_times = [
            state.tracked.most_recent_event().end_timestamp
            for state in waiting_states + active_states
            if state.tracked.most_recent_event() is not None
            and (
                state.tracked.persistent_prefill_done
                or state.tracked.persistent_decode_count > 0
            )
        ]
        if persistent_event_times:
            return max(persistent_event_times)
        return min(
            state.arrival_timestamp for state in waiting_states + active_states
        )

    def _isolated_retract_decode(
        self,
        waiting_states: List[_SimRequestState],
        active_states: List[_SimRequestState],
        *,
        extra_decode_tokens: int,
    ) -> Tuple[bool, bool]:
        if self.max_kv_tokens is None:
            return True, False
        current_kv = self._current_active_kv_tokens(active_states)
        current_required = current_kv + len(active_states) + extra_decode_tokens
        if current_required <= self.max_kv_tokens:
            return True, False

        sorted_states = list(active_states)
        sorted_states.sort(
            key=lambda state: (
                state.simulated_decode_count,
                -state.prompt_tokens,
            ),
            reverse=True,
        )
        retracted_any = False

        while sorted_states and current_required > self.max_kv_tokens:
            if len(sorted_states) == 1 and current_kv > 0:
                break
            state = sorted_states.pop()
            if state in active_states:
                active_states.remove(state)
                current_kv -= state.current_token_count
                state.prefill_done = False
                state.anticipated_recorded = False
                state.tracked.alternate_history_timeline.anticipated_future_events = []
                waiting_states.insert(0, state)
                retracted_any = True
                current_required = (
                    current_kv + len(active_states) + extra_decode_tokens
                )

        return current_required <= self.max_kv_tokens, retracted_any

    def _advance_scheduler_step(
        self,
        waiting_states: List[_SimRequestState],
        active_states: List[_SimRequestState],
        future_history: List[UserEvent],
    ) -> Optional[str]:
        current_time = self._current_history_time(
            waiting_states, active_states, future_history
        )
        next_arrival = min(
            (
                state.arrival_timestamp
                for state in waiting_states
                if state.arrival_timestamp > current_time
            ),
            default=None,
        )
        any_waiting_ready = any(
            state.arrival_timestamp <= current_time for state in waiting_states
        )
        batch = []
        if any_waiting_ready:
            batch = self._build_prefill_batch(
                waiting_states, active_states, future_history
            )
        if batch:
            prompt_sizes = [state.prefill_context_tokens() for state in batch]
            duration = isolated_prefill_time_estimation(
                sum(prompt_sizes),
                max(prompt_sizes),
                len(prompt_sizes),
                self.fairinf_n,
            )
            current_time += duration
            user_event = UserPrefillEvent(duration=duration, end_timestamp=current_time)
            persist_batch = all(state.realized_prefill_done for state in batch)
            if persist_batch:
                self.history.append(user_event)
            else:
                future_history.append(user_event)
            for state in batch:
                state.prefill_done = True
                event = RequestPrefillEvent(
                    req_id=state.tracked.req.rid,
                    duration=duration,
                    end_timestamp=current_time,
                )
                if persist_batch:
                    if not state.tracked.persistent_prefill_done:
                        state.tracked.append_persistent_event(event)
                        TIMELINE_WRITER.mark_isolated_prefill_done(
                            state.tracked.req.rid,
                            state.tracked.req.uid,
                            timestamp_iso=_iso_ts(current_time),
                        )
                elif not state.realized_prefill_done:
                    state.tracked.alternate_history_timeline.anticipated_future_events = [event]
                    state.anticipated_recorded = True
                active_states.append(state)
            del waiting_states[: len(batch)]
            return "prefill"

        if active_states:
            can_decode, retracted_any = self._isolated_retract_decode(
                waiting_states,
                active_states,
                extra_decode_tokens=0,
            )
            if not can_decode:
                return None
            if not active_states:
                return "retract"

            token_counts = [
                state.current_token_count
                for state in active_states
            ]
            duration = isolated_decode_time_estimation(
                sum(token_counts),
                max(token_counts),
                len(token_counts),
                self.fairinf_n,
            )
            if retracted_any:
                duration += RETRACTION_PENALTY_SECONDS
            if next_arrival is not None and current_time + duration > next_arrival:
                future_history.append(
                    UserDecodeEvent(duration=0.0, end_timestamp=next_arrival)
                )
                return "arrival"

            realized_gaps = [
                state.realized_decode_count - state.simulated_decode_count
                for state in active_states
            ]
            persist_rounds = min(realized_gaps) if realized_gaps and min(realized_gaps) > 0 else 0
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
                if next_arrival is not None:
                    future_history.append(
                        UserDecodeEvent(duration=0.0, end_timestamp=next_arrival)
                    )
                    return "arrival"
                return None

            rounds = 1
            persist_batch = persist_rounds > 0
            if persist_batch:
                rounds = min(rounds, persist_rounds)
            if next_arrival is not None:
                rounds_until_arrival = int((next_arrival - current_time) // duration)
                if rounds_until_arrival <= 0:
                    future_history.append(
                        UserDecodeEvent(duration=0.0, end_timestamp=next_arrival)
                    )
                    return "arrival"
                rounds = min(rounds, rounds_until_arrival)

            chunk_start = current_time
            current_time += rounds * duration
            user_event = UserDecodeEvent(
                duration=rounds * duration, end_timestamp=current_time
            )
            if persist_batch:
                self.history.append(user_event)
            else:
                future_history.append(user_event)
            for state in active_states:
                prev_decode_count = state.simulated_decode_count
                state.advance_simulated_decode_count(rounds)

                if persist_batch:
                    realized_upper = min(
                        state.realized_decode_count, state.simulated_decode_count
                    )
                    for completion_number in range(
                        prev_decode_count + 1, realized_upper + 1
                    ):
                        realized_round = completion_number - prev_decode_count
                        realized_ts = chunk_start + realized_round * duration
                        realized_event = RequestDecodeEvent(
                            req_id=state.tracked.req.rid,
                            duration=duration,
                            end_timestamp=realized_ts,
                            completion_number=completion_number,
                        )
                        state.tracked.append_persistent_event(realized_event)
                        TIMELINE_WRITER.mark_isolated_decode_done(
                            state.tracked.req.rid,
                            state.tracked.req.uid,
                            timestamp_iso=_iso_ts(realized_ts),
                            completion_number=completion_number,
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
            return "decode"

        if next_arrival is None:
            return None
        future_history.append(UserPrefillEvent(duration=0.0, end_timestamp=next_arrival))
        return "arrival"

    def rebuild_from_real_state(
        self,
        req_id_real_statuses: Dict[str, RequestEvent],
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        rebuild_start = time.perf_counter()
        live_states = self._live_states(req_id_real_statuses)
        after_live_states = time.perf_counter()
        if not live_states:
            self.anticipated_future_events = []
            self.cached_live_rids = ()
            self.cached_history_end_timestamp = None
            self.cached_state_by_rid = {}
            self.cached_anticipated_event_by_rid = {}
            if timing_breakdown is not None:
                timing_breakdown["rebuild_live_states_ms"] = (
                    after_live_states - rebuild_start
                ) * 1000.0
                timing_breakdown["rebuild_state_setup_ms"] = 0.0
                timing_breakdown["rebuild_scheduler_loop_ms"] = 0.0
            return

        live_rids = tuple(state.tracked.req.rid for state in live_states)
        reuse_cached_frontier = bool(self.cached_state_by_rid)
        future_history: List[UserEvent] = []
        if (
            reuse_cached_frontier
            and self.cached_history_end_timestamp is not None
            and (
                not self.history
                or self.cached_history_end_timestamp > self.history[-1].end_timestamp
            )
        ):
            future_history = [
                UserDecodeEvent(
                    duration=0.0,
                    end_timestamp=self.cached_history_end_timestamp,
                )
            ]
        self.anticipated_future_events = []
        for state in live_states:
            timeline = state.tracked.alternate_history_timeline
            real_event = req_id_real_statuses.get(state.tracked.req.rid)
            if not timeline.history:
                state.tracked.reset_history_to_start()
            cached_prefill_done = False
            cached_simulated_decode_count = 0
            cached_anticipated_event = None
            has_cached_state = False
            if reuse_cached_frontier and state.tracked.req.rid in self.cached_state_by_rid:
                has_cached_state = True
                cached_prefill_done, cached_simulated_decode_count = (
                    self.cached_state_by_rid.get(
                        state.tracked.req.rid,
                        (False, 0),
                    )
                )
                cached_anticipated_event = self.cached_anticipated_event_by_rid.get(
                    state.tracked.req.rid
                )
            if not state.realized_prefill_done:
                state.prefill_done = False
                state.set_simulated_decode_count(0)
            else:
                state.prefill_done = True
                seeded_decode_count = (
                    state.realized_decode_count
                    if not has_cached_state
                    else min(
                        state.simulated_decode_count,
                        state.realized_decode_count,
                    )
                )
                if has_cached_state and cached_prefill_done:
                    seeded_decode_count = max(
                        seeded_decode_count,
                        min(
                            cached_simulated_decode_count,
                            state.realized_decode_count + 1,
                        ),
                    )
                state.set_simulated_decode_count(seeded_decode_count)
            timeline.anticipated_future_events = []
            state.anticipated_recorded = False
            if (
                not state.tracked.live_in_running
                and isinstance(real_event, RequestStartEvent)
                and state.tracked.restart_pending
            ):
                req = state.tracked.req
                context_tokens = (
                    len(req.fill_ids)
                    if req.fill_ids is not None
                    else len(req.origin_input_ids) + len(req.output_ids)
                )
                prefill_duration = isolated_prefill_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                timeline.anticipated_future_events = [
                    RequestPrefillEvent(
                        req_id=req.rid,
                        duration=prefill_duration,
                        end_timestamp=state.arrival_timestamp + prefill_duration,
                    )
                ]
                state.anticipated_recorded = True
                state.tracked.restart_pending = False
            elif not state.tracked.live_in_running and state.realized_prefill_done:
                state.prefill_done = False
                state.set_simulated_decode_count(state.realized_decode_count)
                req = state.tracked.req
                context_tokens = len(req.origin_input_ids) + len(req.output_ids)
                prefill_duration = isolated_prefill_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                timeline.anticipated_future_events = [
                    RequestPrefillEvent(
                        req_id=req.rid,
                        duration=prefill_duration,
                        end_timestamp=time.time() + prefill_duration,
                    )
                ]
                state.anticipated_recorded = True
            elif cached_anticipated_event is not None:
                if (
                    isinstance(cached_anticipated_event, RequestPrefillEvent)
                    and not state.realized_prefill_done
                ):
                    timeline.anticipated_future_events = [cached_anticipated_event]
                    state.anticipated_recorded = True
                elif (
                    isinstance(cached_anticipated_event, RequestDecodeEvent)
                    and state.realized_prefill_done
                    and state.realized_decode_count
                    < cached_anticipated_event.completion_number
                    <= state.simulated_decode_count
                ):
                    timeline.anticipated_future_events = [cached_anticipated_event]
                    state.anticipated_recorded = True
        after_state_setup = time.perf_counter()

        waiting_states = [state for state in live_states if not state.prefill_done]
        active_states: List[_SimRequestState] = [
            state for state in live_states if state.prefill_done
        ]
        scheduler_step_count = 0
        prefill_step_count = 0
        decode_step_count = 0

        while True:
            if all(state.anticipated_recorded for state in live_states):
                break
            step_kind = self._advance_scheduler_step(
                waiting_states, active_states, future_history
            )
            if step_kind is None:
                break
            scheduler_step_count += 1
            if step_kind == "prefill":
                prefill_step_count += 1
            elif step_kind == "decode":
                decode_step_count += 1
        self.anticipated_future_events = future_history
        self.cached_live_rids = live_rids
        self.cached_history_end_timestamp = self._current_history_time(
            waiting_states, active_states, future_history
        )
        self.cached_state_by_rid = {
            state.tracked.req.rid: (state.prefill_done, state.simulated_decode_count)
            for state in live_states
        }
        self.cached_anticipated_event_by_rid = {
            state.tracked.req.rid: state.tracked.alternate_history_timeline.anticipated_future_events[0]
            for state in live_states
            if state.tracked.alternate_history_timeline.anticipated_future_events
        }
        after_scheduler_loop = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["rebuild_live_states_ms"] = (
                after_live_states - rebuild_start
            ) * 1000.0
            timing_breakdown["rebuild_state_setup_ms"] = (
                after_state_setup - after_live_states
            ) * 1000.0
            timing_breakdown["rebuild_scheduler_loop_ms"] = (
                after_scheduler_loop - after_state_setup
            ) * 1000.0
            timing_breakdown["rebuild_scheduler_step_count"] = scheduler_step_count
            timing_breakdown["rebuild_prefill_step_count"] = prefill_step_count
            timing_breakdown["rebuild_decode_step_count"] = decode_step_count


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
        enable_timeline_logging: bool = True,
    ):
        self.max_kv_tokens_per_user = max_kv_tokens_per_user
        self.fairinf_n = max(int(fairinf_n), 1)
        self.min_new_token_ratio = max(0.0, float(min_new_token_ratio))
        self.enable_timeline_logging = enable_timeline_logging
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

    def _seed_waiting_prefill_future_event(
        self,
        tracked: TrackedRequest,
        *,
        now: float,
    ) -> None:
        req = tracked.req
        context_tokens = (
            len(req.fill_ids)
            if req.fill_ids is not None
            else len(req.origin_input_ids) + len(req.output_ids)
        )
        prefill_duration = isolated_prefill_time_estimation(
            context_tokens,
            context_tokens,
            1,
            self.fairinf_n,
        )
        tracked.arrival_timestamp = now
        tracked.reset_history_to_start()
        tracked.alternate_history_timeline.history = [
            RequestStartEvent(req_id=req.rid, end_timestamp=now)
        ]
        tracked.alternate_history_timeline.anticipated_future_events = [
            RequestPrefillEvent(
                req_id=req.rid,
                duration=prefill_duration,
                end_timestamp=now + prefill_duration,
            )
        ]

    def _seed_waiting_reprefill_future_event(
        self,
        tracked: TrackedRequest,
        *,
        now: float,
    ) -> None:
        req = tracked.req
        context_tokens = len(req.origin_input_ids) + len(req.output_ids)
        prefill_duration = isolated_prefill_time_estimation(
            context_tokens,
            context_tokens,
            1,
            self.fairinf_n,
        )
        tracked.alternate_history_timeline.anticipated_future_events = [
            RequestPrefillEvent(
                req_id=req.rid,
                duration=prefill_duration,
                end_timestamp=now + prefill_duration,
            )
        ]

    def sync_live_user_tracking(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        running_reqs = list(running_batch.reqs) if running_batch is not None else []
        running_rids = {req.rid for req in running_reqs}
        live_by_user: Dict[str, List[Req]] = {}
        for req in waiting_queue:
            live_by_user.setdefault(req.uid, []).append(req)
        for req in running_reqs:
            live_by_user.setdefault(req.uid, []).append(req)

        live_user_ids = sorted(
            set(self.users)
            | set(live_by_user)
        )

        for uid in live_user_ids:
            user_timeline = self.users.get(uid)
            if user_timeline is None:
                user_timeline = self._make_user_timeline(uid)
                self.users[uid] = user_timeline

            for req in live_by_user.get(uid, []):
                tracked = self._ensure_tracked_request(req, deltas_in_microseconds)
                tracked.live_in_running = req.rid in running_rids
                self._track_request(req, tracked, user_timeline)

            live_rids = {req.rid for req in live_by_user.get(uid, [])}
            for rid in list(user_timeline.request_timelines.keys()):
                if rid in live_rids:
                    continue
                user_timeline.request_timelines.pop(rid, None)
                self.requests.pop(rid, None)
                self.most_recent_event_real.pop(rid, None)

        for uid in list(self.users.keys()):
            user_timeline = self.users[uid]
            if uid in live_user_ids and user_timeline.request_timelines:
                continue
            self.users.pop(uid, None)
            for rid, tracked in list(self.requests.items()):
                if tracked.req.uid == uid:
                    self.requests.pop(rid, None)
                    self.most_recent_event_real.pop(rid, None)

        return live_user_ids

    def start_of_pass(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        pass_start = time.perf_counter()
        live_user_ids = self.sync_live_user_tracking(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=deltas_in_microseconds,
        )
        after_sync = time.perf_counter()
        self.rebuild_all_tracked_requests(
            live_user_ids,
            timing_breakdown=timing_breakdown,
        )
        after_rebuild = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["sync_live_user_tracking_ms"] = (
                after_sync - pass_start
            ) * 1000.0
            timing_breakdown["rebuild_from_real_state_ms"] = (
                after_rebuild - after_sync
            ) * 1000.0
            timing_breakdown["simulator_start_of_pass_ms"] = (
                after_rebuild - pass_start
            ) * 1000.0
        return live_user_ids

    def rebuild_all_tracked_requests(
        self,
        user_ids: Optional[List[str]] = None,
        *,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        rebuild_live_states_ms = 0.0
        rebuild_state_setup_ms = 0.0
        rebuild_scheduler_loop_ms = 0.0
        rebuild_scheduler_step_count = 0
        rebuild_prefill_step_count = 0
        rebuild_decode_step_count = 0
        for uid in (user_ids if user_ids is not None else list(self.users.keys())):
            user_timeline = self.users.get(uid)
            if user_timeline is not None:
                rebuild_breakdown: Dict[str, float] = {}
                user_timeline.rebuild_from_real_state(
                    self.most_recent_event_real,
                    timing_breakdown=rebuild_breakdown,
                )
                rebuild_live_states_ms += rebuild_breakdown.get(
                    "rebuild_live_states_ms", 0.0
                )
                rebuild_state_setup_ms += rebuild_breakdown.get(
                    "rebuild_state_setup_ms", 0.0
                )
                rebuild_scheduler_loop_ms += rebuild_breakdown.get(
                    "rebuild_scheduler_loop_ms", 0.0
                )
                rebuild_scheduler_step_count += int(
                    rebuild_breakdown.get("rebuild_scheduler_step_count", 0)
                )
                rebuild_prefill_step_count += int(
                    rebuild_breakdown.get("rebuild_prefill_step_count", 0)
                )
                rebuild_decode_step_count += int(
                    rebuild_breakdown.get("rebuild_decode_step_count", 0)
                )
        if timing_breakdown is not None:
            timing_breakdown["rebuild_live_states_ms"] = rebuild_live_states_ms
            timing_breakdown["rebuild_state_setup_ms"] = rebuild_state_setup_ms
            timing_breakdown["rebuild_scheduler_loop_ms"] = rebuild_scheduler_loop_ms
            timing_breakdown["rebuild_scheduler_step_count"] = rebuild_scheduler_step_count
            timing_breakdown["rebuild_prefill_step_count"] = rebuild_prefill_step_count
            timing_breakdown["rebuild_decode_step_count"] = rebuild_decode_step_count

    def build_deadline_candidates(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        include_ordered_waiting_queue: bool = False,
        req_is_fair_prefill: Callable[[Req, Optional[ScheduleBatch]], bool],
        req_is_fair_decode: Callable[[Req, Optional[ScheduleBatch]], bool],
        event_delta_seconds: Callable[[TrackedRequest, RequestEvent], float],
        pooled_prefill_estimate_seconds: Callable[[Req], float],
        pooled_decode_estimate_seconds: Callable[[Req, Optional[ScheduleBatch]], float],
    ) -> Tuple[List[DeadlineCandidate], Dict[str, float]] | Tuple[
        List[DeadlineCandidate], Dict[str, float], Tuple[Req, ...]
    ]:
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
            if (
                not upcoming_events
                and rid in running_by_rid
                and isinstance(real_event, (RequestPrefillEvent, RequestDecodeEvent))
            ):
                next_completion_number = (
                    real_event.completion_number + 1
                    if isinstance(real_event, RequestDecodeEvent)
                    else max(1, len(req.output_ids) + 1)
                )
                context_tokens = len(req.origin_input_ids) + next_completion_number
                decode_duration = isolated_decode_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                upcoming_events = [
                    RequestDecodeEvent(
                        req_id=rid,
                        duration=decode_duration,
                        end_timestamp=float(real_event.end_timestamp) + decode_duration,
                        completion_number=next_completion_number,
                    )
                ]
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
        if not include_ordered_waiting_queue:
            return candidates, waiting_prefill_deadline_by_rid
        ordered_waiting_prefills = tuple(
            candidate.req for candidate in candidates if candidate.event_type == "prefill"
        )
        return candidates, waiting_prefill_deadline_by_rid, ordered_waiting_prefills

    def process_new_request(
        self, req: Req, deltas_in_microseconds: Optional[Dict[str, int]] = None
    ) -> None:
        now = time.time()
        if self.enable_timeline_logging:
            TIMELINE_WRITER.mark_isolated_start(
                req.rid,
                req.uid,
                timestamp_iso=_iso_ts(now),
            )
        previous_real_event = self.most_recent_event_real.get(req.rid)
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
            had_progress = (
                tracked.persistent_prefill_done
                or tracked.persistent_decode_count > 0
                or isinstance(previous_real_event, (RequestPrefillEvent, RequestDecodeEvent))
            )
            tracked.arrival_timestamp = now
            tracked.req = req
            tracked.restart_pending = had_progress
            if deltas_in_microseconds is not None:
                tracked.deltas_in_microseconds = dict(deltas_in_microseconds)
        self._seed_waiting_prefill_future_event(tracked, now=now)

    def finished_prefill(self, batch: ScheduleBatch) -> None:
        now = time.time()
        for req in batch.reqs:
            self.most_recent_event_real[req.rid] = RequestPrefillEvent(
                req_id=req.rid,
                end_timestamp=now,
            )
            tracked = self.requests.get(req.rid)
            if tracked is None:
                continue
            prefill_event = next(
                (
                    event
                    for event in tracked.alternate_history_timeline.anticipated_future_events
                    if isinstance(event, RequestPrefillEvent)
                ),
                None,
            )
            if prefill_event is None:
                prefill_duration = isolated_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    self.fairinf_n,
                )
                simulated_prefill_done_ts = tracked.arrival_timestamp + prefill_duration
            else:
                simulated_prefill_done_ts = prefill_event.end_timestamp
            if self.enable_timeline_logging:
                TIMELINE_WRITER.mark_isolated_prefill_done(
                    req.rid,
                    req.uid,
                    timestamp_iso=_iso_ts(simulated_prefill_done_ts),
                )
            tracked.latest_simulated_completion_timestamp = simulated_prefill_done_ts
            first_decode_completion = len(req.output_ids) + 1
            decode_context_tokens = len(req.origin_input_ids) + first_decode_completion
            decode_duration = isolated_decode_time_estimation(
                decode_context_tokens,
                decode_context_tokens,
                1,
                self.fairinf_n,
            )
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestDecodeEvent(
                    req_id=req.rid,
                    duration=decode_duration,
                    end_timestamp=simulated_prefill_done_ts + decode_duration,
                    completion_number=first_decode_completion,
                )
            ]

    def finished_decode(self, batch: ScheduleBatch, decode_rounds: int = 1) -> None:
        now = time.time()
        for req in batch.reqs:
            final_completion_number = len(req.output_ids)
            self.most_recent_event_real[req.rid] = RequestDecodeEvent(
                req_id=req.rid,
                end_timestamp=now,
                completion_number=final_completion_number,
            )
            tracked = self.requests.get(req.rid)
            if tracked is None:
                continue
            timeline = tracked.alternate_history_timeline
            if tracked.persistent_decode_count > 0:
                next_completion_number = tracked.persistent_decode_count + 1
                current_end_timestamp = tracked.persistent_decode_events[
                    tracked.persistent_decode_count
                ].end_timestamp
            elif tracked.persistent_prefill_event is not None:
                next_completion_number = 1
                current_end_timestamp = tracked.persistent_prefill_event.end_timestamp
            else:
                prefill_event = next(
                    (
                        event
                        for event in timeline.anticipated_future_events
                        if isinstance(event, RequestPrefillEvent)
                    ),
                    None,
                )
                if prefill_event is None:
                    prefill_duration = isolated_prefill_time_estimation(
                        len(req.origin_input_ids),
                        len(req.origin_input_ids),
                        1,
                        self.fairinf_n,
                    )
                    current_end_timestamp = tracked.arrival_timestamp + prefill_duration
                else:
                    current_end_timestamp = prefill_event.end_timestamp
                next_completion_number = 1

            for completion_number in range(
                next_completion_number, final_completion_number + 1
            ):
                context_tokens = len(req.origin_input_ids) + completion_number
                decode_duration = isolated_decode_time_estimation(
                    context_tokens,
                    context_tokens,
                    1,
                    self.fairinf_n,
                )
                current_end_timestamp += decode_duration
                if self.enable_timeline_logging:
                    TIMELINE_WRITER.mark_isolated_decode_done(
                        req.rid,
                        req.uid,
                        timestamp_iso=_iso_ts(current_end_timestamp),
                        completion_number=completion_number,
                    )
            tracked.latest_simulated_completion_timestamp = current_end_timestamp
            next_future_completion_number = final_completion_number + 1
            next_context_tokens = len(req.origin_input_ids) + next_future_completion_number
            next_decode_duration = isolated_decode_time_estimation(
                next_context_tokens,
                next_context_tokens,
                1,
                self.fairinf_n,
            )
            timeline.anticipated_future_events = [
                RequestDecodeEvent(
                    req_id=req.rid,
                    duration=next_decode_duration,
                    end_timestamp=current_end_timestamp + next_decode_duration,
                    completion_number=next_future_completion_number,
                )
            ]

    def mark_request_finished(self, req: Req) -> None:
        tracked = self.requests.get(req.rid)
        if tracked is not None and tracked.persistent_decode_count < len(req.output_ids):
            self.finished_decode(SimpleNamespace(reqs=[req]))
            tracked = self.requests.get(req.rid)
        if tracked is not None and tracked.latest_simulated_completion_timestamp is not None:
            if self.enable_timeline_logging:
                TIMELINE_WRITER.mark_isolated_completed(
                    req.rid,
                    req.uid,
                    timestamp_iso=_iso_ts(tracked.latest_simulated_completion_timestamp),
                )
        user_timeline = self.users.get(req.uid)
        if user_timeline is not None:
            user_timeline.finished_request(req.rid)
        self.requests.pop(req.rid, None)
        self.most_recent_event_real.pop(req.rid, None)
