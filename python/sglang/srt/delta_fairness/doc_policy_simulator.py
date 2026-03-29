from __future__ import annotations

"""Alternate History Simulator — per AlternateHistory.md design."""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.policy_scheduler import CLIP_MAX_NEW_TOKENS
from sglang.srt.request_timeline import TIMELINE_WRITER

from .time_estimation import (
    isolated_decode_time_estimation,
    isolated_prefill_time_estimation,
)

logger = logging.getLogger(__name__)

RETRACTION_PENALTY_SECONDS = 0.030


def _iso_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# RequestEvent hierarchy
# ---------------------------------------------------------------------------

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
        if isinstance(other, RequestDecodeEvent):
            return self.completion_number > other.completion_number
        return isinstance(other, (RequestStartEvent, RequestPrefillEvent))


# ---------------------------------------------------------------------------
# RequestTimeline
# ---------------------------------------------------------------------------

@dataclass
class RequestTimeline:
    """
    history: committed real events (Start, Prefill, DecodeEvent(1..N))
    next_anticipated_event: the single next predicted event
    """
    history: List[RequestEvent] = field(default_factory=list)
    next_anticipated_event: Optional[RequestEvent] = None

    def events_after(self, real_event: RequestEvent) -> List[RequestEvent]:
        """Events logically after real_event, from history + next_anticipated_event."""
        result: List[RequestEvent] = []

        if isinstance(real_event, RequestStartEvent):
            for e in self.history:
                if isinstance(e, (RequestPrefillEvent, RequestDecodeEvent)):
                    result.append(e)
            if self.next_anticipated_event is not None:
                if isinstance(self.next_anticipated_event, (RequestPrefillEvent, RequestDecodeEvent)):
                    if self.next_anticipated_event not in result:
                        result.append(self.next_anticipated_event)
        elif isinstance(real_event, RequestPrefillEvent):
            for e in self.history:
                if isinstance(e, RequestDecodeEvent):
                    result.append(e)
            if isinstance(self.next_anticipated_event, RequestDecodeEvent):
                if self.next_anticipated_event not in result:
                    result.append(self.next_anticipated_event)
        elif isinstance(real_event, RequestDecodeEvent):
            for e in self.history:
                if isinstance(e, RequestDecodeEvent) and e.completion_number > real_event.completion_number:
                    result.append(e)
            if (
                isinstance(self.next_anticipated_event, RequestDecodeEvent)
                and self.next_anticipated_event.completion_number > real_event.completion_number
                and self.next_anticipated_event not in result
            ):
                result.append(self.next_anticipated_event)

        return result


# ---------------------------------------------------------------------------
# RequestStatusReal
# ---------------------------------------------------------------------------

@dataclass
class RequestStatusReal:
    """Tracks what a request has done in the real world."""
    rid: str
    prefill_done: bool = False
    decode_count: int = 0
    is_complete: bool = False


# ---------------------------------------------------------------------------
# TrackedRequest
# ---------------------------------------------------------------------------

@dataclass
class TrackedRequest:
    req: Req
    arrival_timestamp: float
    deltas_in_microseconds: Dict[str, int] = field(
        default_factory=lambda: {"prefill": 0, "first_decode": 0, "decode": 0}
    )
    timeline: RequestTimeline = field(default_factory=RequestTimeline)
    latest_simulated_completion_timestamp: Optional[float] = None


# ---------------------------------------------------------------------------
# DeadlineCandidate
# ---------------------------------------------------------------------------

@dataclass
class DeadlineCandidate:
    deadline: float
    start_deadline: float
    event_type: str  # "prefill" or "decode"
    req: Req
    event: RequestEvent


# ---------------------------------------------------------------------------
# UserTimeline
# ---------------------------------------------------------------------------

@dataclass
class UserTimeline:
    uid: str
    max_kv_tokens: Optional[int] = None
    fairinf_n: int = 1
    min_new_token_ratio: float = 0.0
    request_timelines: Dict[str, TrackedRequest] = field(default_factory=dict)
    requests_real: Dict[str, RequestStatusReal] = field(default_factory=dict)
    finished_request_timelines: Dict[str, TrackedRequest] = field(default_factory=dict)

    def finished_request(self, rid: str) -> None:
        tracked = self.request_timelines.pop(rid, None)
        if tracked is not None:
            self.finished_request_timelines[rid] = tracked
        self.requests_real.pop(rid, None)

    # ------------------------------------------------------------------
    # Isolated scheduler primitives (operate on rid lists, not wrapper objects)
    # ------------------------------------------------------------------

    def _build_prefill_batch(
        self,
        waiting_rids: List[str],
        active_rids: List[str],
        current_time: float,
        sim_decode_count: Dict[str, int],
    ) -> List[str]:
        """Returns rids to prefill now."""
        ready = [
            rid for rid in waiting_rids
            if self.request_timelines[rid].timeline.history
            and self.request_timelines[rid].timeline.history[0].end_timestamp <= current_time
        ]
        if not ready:
            return []

        batch: List[str] = []
        active_kv = sum(
            len(self.request_timelines[rid].req.origin_input_ids) + max(1, sim_decode_count.get(rid, 0))
            for rid in active_rids
            if rid in self.request_timelines
        )
        batch_tokens = 0
        budget = self.max_kv_tokens
        for rid in ready:
            tracked = self.request_timelines[rid]
            req = tracked.req
            pt = len(req.origin_input_ids)
            sp = getattr(req, "sampling_params", None)
            max_new = min(int(getattr(sp, "max_new_tokens", 0) or 0), CLIP_MAX_NEW_TOKENS)
            if budget is not None and active_kv + batch_tokens + pt > budget:
                break
            batch.append(rid)
            batch_tokens += pt
        return batch

    def _isolated_retract(
        self,
        waiting_rids: List[str],
        active_rids: List[str],
        sim_decode_count: Dict[str, int],
    ) -> Optional[str]:
        """Evict the longest-running request back to waiting. Returns the evicted rid, or None."""
        if self.max_kv_tokens is None:
            return None
        active_kv = sum(
            len(self.request_timelines[rid].req.origin_input_ids) + max(1, sim_decode_count.get(rid, 0))
            for rid in active_rids
            if rid in self.request_timelines
        )
        if active_kv + len(active_rids) <= self.max_kv_tokens:
            return None
        # Evict the one with most decodes (longest running)
        if len(active_rids) <= 1:
            return None
        evict_rid = max(active_rids, key=lambda r: sim_decode_count.get(r, 0))
        active_rids.remove(evict_rid)
        waiting_rids.insert(0, evict_rid)
        sim_decode_count.pop(evict_rid, None)
        # Clear its anticipated event
        tracked = self.request_timelines.get(evict_rid)
        if tracked is not None:
            tracked.timeline.next_anticipated_event = None
        # Reset prefill_done so the isolated scheduler re-prefills this request
        s = self.requests_real.get(evict_rid)
        if s is not None:
            s.prefill_done = False
            s.decode_count = 0
        return evict_rid

    def _advance_scheduler_step(
        self,
        waiting_rids: List[str],
        active_rids: List[str],
        current_time: float,
        sim_decode_count: Dict[str, int],
        anticipated_recorded: set,
    ) -> Tuple[Optional[str], float]:
        """One step. Returns (step_kind, new_current_time). Never writes to timeline.history."""
        def _arrival_ts(rid: str) -> float:
            h = self.request_timelines[rid].timeline.history
            return h[0].end_timestamp if h else float("inf")

        next_arrival = min(
            (
                _arrival_ts(rid)
                for rid in waiting_rids
                if rid in self.request_timelines
                and _arrival_ts(rid) > current_time
            ),
            default=None,
        )
        any_ready = any(
            rid in self.request_timelines
            and _arrival_ts(rid) <= current_time
            for rid in waiting_rids
        )

        if any_ready:
            batch = self._build_prefill_batch(waiting_rids, active_rids, current_time, sim_decode_count)
            if batch:
                prompt_sizes = [len(self.request_timelines[rid].req.origin_input_ids) for rid in batch]
                duration = isolated_prefill_time_estimation(
                    sum(prompt_sizes), max(prompt_sizes), len(prompt_sizes), self.fairinf_n
                )
                current_time += duration
                for rid in batch:
                    waiting_rids.remove(rid)
                    active_rids.append(rid)
                    status = self.requests_real.get(rid)
                    if status is None or not status.prefill_done:
                        # This prefill is anticipated (not yet real)
                        self.request_timelines[rid].timeline.next_anticipated_event = RequestPrefillEvent(
                            req_id=rid, duration=duration, end_timestamp=current_time
                        )
                        anticipated_recorded.add(rid)
                return "prefill", current_time

        if active_rids:
            evicted_rid = self._isolated_retract(waiting_rids, active_rids, sim_decode_count)
            retracted = evicted_rid is not None
            if retracted:
                current_time += RETRACTION_PENALTY_SECONDS
                # Allow the evicted request to receive a fresh anticipated prefill event
                anticipated_recorded.discard(evicted_rid)

            if not active_rids:
                return "retract", current_time

            token_counts = [
                len(self.request_timelines[rid].req.origin_input_ids) + max(1, sim_decode_count.get(rid, 0))
                for rid in active_rids
                if rid in self.request_timelines
            ]
            if not token_counts:
                return "retract", current_time

            duration = isolated_decode_time_estimation(
                sum(token_counts), max(token_counts), len(token_counts), self.fairinf_n
            )

            if next_arrival is not None and current_time + duration > next_arrival:
                return "arrival", next_arrival

            current_time += duration
            for rid in active_rids:
                sim_decode_count[rid] = sim_decode_count.get(rid, 0) + 1
                if rid not in anticipated_recorded:
                    status = self.requests_real.get(rid)
                    real_dc = status.decode_count if status else 0
                    if sim_decode_count[rid] > real_dc:
                        tracked = self.request_timelines.get(rid)
                        if tracked is not None:
                            tracked.timeline.next_anticipated_event = RequestDecodeEvent(
                                req_id=rid, duration=duration,
                                end_timestamp=current_time,
                                completion_number=real_dc + 1,
                            )
                            anticipated_recorded.add(rid)
            return "decode", current_time

        if next_arrival is None:
            return None, current_time
        return "arrival", next_arrival

    def rebuild_from_real_state(
        self,
        _unused_real_statuses=None,
        until_timestamp: Optional[float] = None,
        timing_breakdown: Optional[Dict] = None,
    ) -> None:
        """Run the isolated scheduler forward to set next_anticipated_event for each live request."""
        if not self.request_timelines:
            return

        # Split into waiting (prefill not done) and active (prefill done) by real status
        waiting_rids: List[str] = []
        active_rids: List[str] = []
        sim_decode_count: Dict[str, int] = {}

        for rid, tracked in self.request_timelines.items():
            status = self.requests_real.get(rid)
            # Clear stale anticipated event
            tracked.timeline.next_anticipated_event = None
            if status is not None and status.prefill_done:
                active_rids.append(rid)
                sim_decode_count[rid] = status.decode_count
            else:
                waiting_rids.append(rid)

        # Sort by arrival timestamp
        waiting_rids.sort(key=lambda r: self.request_timelines[r].timeline.history[0].end_timestamp if self.request_timelines[r].timeline.history else 0.0)
        active_rids.sort(key=lambda r: self.request_timelines[r].arrival_timestamp)

        # Seed current_time from the last real committed event across active requests
        current_time = 0.0
        for rid in active_rids:
            tracked = self.request_timelines[rid]
            h = tracked.timeline.history
            if h:
                current_time = max(current_time, h[-1].end_timestamp)
        if not active_rids and waiting_rids:
            current_time = min(
                (
                    self.request_timelines[r].timeline.history[0].end_timestamp
                    for r in waiting_rids
                    if self.request_timelines[r].timeline.history
                ),
                default=0.0,
            )

        anticipated_recorded: set = set()
        all_rids = set(waiting_rids) | set(active_rids)

        for _ in range(200):
            if anticipated_recorded >= all_rids:
                break
            step_kind, current_time = self._advance_scheduler_step(
                waiting_rids, active_rids, current_time, sim_decode_count, anticipated_recorded
            )
            if step_kind is None:
                break
            if until_timestamp is not None and current_time >= until_timestamp:
                break

        # Any waiting request still without an anticipated event is queued (blocked by active decodes).
        # Use inf so it sorts behind requests whose isolated prefill slot is known.
        for rid in waiting_rids:
            if rid not in anticipated_recorded:
                tracked = self.request_timelines.get(rid)
                if tracked is None:
                    continue
                tracked.timeline.next_anticipated_event = RequestPrefillEvent(
                    req_id=tracked.req.rid, duration=0.0,
                    end_timestamp=float("inf"),
                )


# ---------------------------------------------------------------------------
# AlternateHistorySimulator
# ---------------------------------------------------------------------------

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

    def _ensure_tracked(self, req: Req, deltas: Optional[Dict[str, int]]) -> TrackedRequest:
        tracked = self.requests.get(req.rid)
        if tracked is None:
            arrival = self.most_recent_event_real.get(req.rid)
            tracked = TrackedRequest(
                req=req,
                arrival_timestamp=getattr(arrival, "end_timestamp", time.time()),
                deltas_in_microseconds=dict(deltas or {"prefill": 0, "first_decode": 0, "decode": 0}),
            )
            self.requests[req.rid] = tracked
        else:
            tracked.req = req
            if deltas is not None:
                tracked.deltas_in_microseconds = dict(deltas)
        return tracked

    def get_live_users(
        self,
        running_batch,
        waiting_queue: List[Req],
        *,
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        running_reqs = list(running_batch.reqs) if running_batch is not None else []
        waiting_rids: set = {req.rid for req in waiting_queue}
        live_by_user: Dict[str, List[Req]] = {}
        for req in waiting_queue:
            live_by_user.setdefault(req.uid, []).append(req)
        for req in running_reqs:
            live_by_user.setdefault(req.uid, []).append(req)

        live_user_ids = sorted(set(self.users) | set(live_by_user))

        for uid in live_user_ids:
            ut = self.users.get(uid)
            if ut is None:
                ut = self._make_user_timeline(uid)
                self.users[uid] = ut
            for req in live_by_user.get(uid, []):
                tracked = self._ensure_tracked(req, deltas_in_microseconds)
                ut.request_timelines[req.rid] = tracked
                self.requests[req.rid] = tracked
                # Sync requests_real from tracked timeline history.
                # Waiting requests are treated as not yet prefilled (retracted state).
                h = tracked.timeline.history
                if req.rid in waiting_rids:
                    s = ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid))
                    s.prefill_done = False
                    s.decode_count = 0
                    # Reset most_recent_event_real so events_after returns prefill events.
                    self.most_recent_event_real[req.rid] = RequestStartEvent(
                        req_id=req.rid, end_timestamp=tracked.arrival_timestamp
                    )
                    # Trim timeline history to just the start event so stale prefill/decode
                    # events don't appear in events_after for this retracted request.
                    start_events = [e for e in tracked.timeline.history if isinstance(e, RequestStartEvent)]
                    tracked.timeline.history = start_events or [
                        RequestStartEvent(req_id=req.rid, end_timestamp=tracked.arrival_timestamp)
                    ]
                else:
                    prefill_done = any(isinstance(e, RequestPrefillEvent) for e in h)
                    decode_count = max(
                        (e.completion_number for e in h if isinstance(e, RequestDecodeEvent)),
                        default=0,
                    )
                    s = ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid))
                    s.prefill_done = prefill_done
                    s.decode_count = decode_count
            live_rids = {req.rid for req in live_by_user.get(uid, [])}
            for rid in list(ut.request_timelines):
                if rid not in live_rids:
                    ut.request_timelines.pop(rid, None)
                    self.requests.pop(rid, None)

        live_user_set = set(live_user_ids)
        for uid in list(self.users):
            ut = self.users[uid]
            if uid in live_user_set and ut.request_timelines:
                continue
            self.users.pop(uid, None)
            # Use the user's own request_timelines dict rather than scanning all self.requests
            for rid in list(ut.request_timelines):
                self.requests.pop(rid, None)
            ut.request_timelines.clear()

        return live_user_ids

    def start_of_pass(
        self,
        running_batch,
        waiting_queue: List[Req],
        *,
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
    ) -> None:
        self.get_live_users(running_batch, waiting_queue, deltas_in_microseconds=deltas_in_microseconds)
        for ut in self.users.values():
            ut.rebuild_from_real_state()

    def process_new_request(
        self,
        req: Req,
        deltas_in_microseconds: Optional[Dict[str, int]] = None,
        *,
        arrival_timestamp: Optional[float] = None,
    ) -> None:
        now = arrival_timestamp if arrival_timestamp is not None else time.time()
        if self.enable_timeline_logging:
            TIMELINE_WRITER.mark_isolated_start(req.rid, req.uid, timestamp_iso=_iso_ts(now))

        self.most_recent_event_real[req.rid] = RequestStartEvent(req_id=req.rid, end_timestamp=now)

        tracked = self.requests.get(req.rid)
        if tracked is None:
            tracked = TrackedRequest(
                req=req,
                arrival_timestamp=now,
                deltas_in_microseconds=dict(deltas_in_microseconds or {"prefill": 0, "first_decode": 0, "decode": 0}),
            )
            self.requests[req.rid] = tracked
        else:
            tracked.req = req
            tracked.arrival_timestamp = now
            tracked.latest_simulated_completion_timestamp = None
            if deltas_in_microseconds is not None:
                tracked.deltas_in_microseconds = dict(deltas_in_microseconds)

        # Reset timeline: just the start event + anticipated prefill
        context_tokens = (
            len(req.fill_ids)
            if getattr(req, "fill_ids", None) is not None
            else len(req.origin_input_ids) + len(getattr(req, "output_ids", []))
        )
        dur = isolated_prefill_time_estimation(context_tokens, context_tokens, 1, self.fairinf_n)
        tracked.timeline = RequestTimeline(
            history=[RequestStartEvent(req_id=req.rid, end_timestamp=now)],
            next_anticipated_event=RequestPrefillEvent(req_id=req.rid, duration=dur, end_timestamp=now + dur),
        )

        ut = self.users.get(req.uid)
        if ut is not None:
            ut.requests_real[req.rid] = RequestStatusReal(rid=req.rid)
            ut.request_timelines[req.rid] = tracked

    def finished_prefill(self, batch) -> None:
        for req in batch.reqs:
            tracked = self.requests.get(req.rid)
            if tracked is None:
                continue

            ant = tracked.timeline.next_anticipated_event
            iso_ts = ant.end_timestamp if isinstance(ant, RequestPrefillEvent) else (
                tracked.arrival_timestamp + isolated_prefill_time_estimation(
                    len(req.origin_input_ids), len(req.origin_input_ids), 1, self.fairinf_n
                )
            )

            tracked.timeline.history.append(
                RequestPrefillEvent(req_id=req.rid, duration=0.0, end_timestamp=iso_ts)
            )
            tracked.latest_simulated_completion_timestamp = iso_ts
            self.most_recent_event_real[req.rid] = RequestPrefillEvent(req_id=req.rid, end_timestamp=iso_ts)

            if self.enable_timeline_logging:
                TIMELINE_WRITER.mark_isolated_prefill_done(req.rid, req.uid, timestamp_iso=_iso_ts(iso_ts))

            ut = self.users.get(req.uid)
            if ut is not None:
                ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid)).prefill_done = True

            # Seed anticipated first decode anchored to max(iso_ts, now) so it's not in the past
            first_n = len(getattr(req, "output_ids", [])) + 1
            ctx = len(req.origin_input_ids) + first_n
            dec_dur = isolated_decode_time_estimation(ctx, ctx, 1, self.fairinf_n)
            base = max(iso_ts, time.time())
            tracked.timeline.next_anticipated_event = RequestDecodeEvent(
                req_id=req.rid, duration=dec_dur,
                end_timestamp=base + dec_dur, completion_number=first_n,
            )

    def finished_decode(self, batch, decode_rounds: int = 1) -> None:
        for req in batch.reqs:
            tracked = self.requests.get(req.rid)
            if tracked is None:
                continue

            h = tracked.timeline.history
            if h and isinstance(h[-1], RequestDecodeEvent):
                last_n, base_ts = h[-1].completion_number, h[-1].end_timestamp
            elif h and isinstance(h[-1], RequestPrefillEvent):
                last_n, base_ts = 0, h[-1].end_timestamp
            else:
                dur = isolated_prefill_time_estimation(
                    len(req.origin_input_ids), len(req.origin_input_ids), 1, self.fairinf_n
                )
                base_ts = tracked.arrival_timestamp + dur
                last_n = 0
                h.append(RequestPrefillEvent(req_id=req.rid, duration=dur, end_timestamp=base_ts))
                self.most_recent_event_real[req.rid] = RequestPrefillEvent(req_id=req.rid, end_timestamp=base_ts)
                ut = self.users.get(req.uid)
                if ut is not None:
                    ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid)).prefill_done = True

            # Advance isolation timeline by decode_rounds steps from last_n.
            # Also respect len(output_ids) in case the caller has more accurate info
            # (e.g. direct calls where output_ids reflects real progress).
            new_final_n = max(last_n + decode_rounds, len(getattr(req, "output_ids", [])))
            cur_ts = base_ts
            for n in range(last_n + 1, new_final_n + 1):
                ctx = len(req.origin_input_ids) + n
                dur = isolated_decode_time_estimation(ctx, ctx, 1, self.fairinf_n)
                cur_ts += dur
                if self.enable_timeline_logging:
                    TIMELINE_WRITER.mark_isolated_decode_done(req.rid, req.uid, timestamp_iso=_iso_ts(cur_ts), completion_number=n)

            # Keep history compact: only the start event + the latest real event.
            start_events = [e for e in h if isinstance(e, RequestStartEvent)]
            if new_final_n > 0:
                last_real = RequestDecodeEvent(
                    req_id=req.rid, duration=0.0, end_timestamp=cur_ts, completion_number=new_final_n
                )
                tracked.timeline.history = start_events + [last_real]
            else:
                tracked.timeline.history = start_events

            tracked.latest_simulated_completion_timestamp = cur_ts
            self.most_recent_event_real[req.rid] = RequestDecodeEvent(
                req_id=req.rid, end_timestamp=cur_ts, completion_number=new_final_n
            )
            ut = self.users.get(req.uid)
            if ut is not None:
                s = ut.requests_real.setdefault(req.rid, RequestStatusReal(rid=req.rid, prefill_done=True))
                s.prefill_done = True
                s.decode_count = new_final_n

            next_n = new_final_n + 1
            ctx = len(req.origin_input_ids) + next_n
            dur = isolated_decode_time_estimation(ctx, ctx, 1, self.fairinf_n)
            base = max(cur_ts, time.time())
            tracked.timeline.next_anticipated_event = RequestDecodeEvent(
                req_id=req.rid, duration=dur,
                end_timestamp=base + dur, completion_number=next_n,
            )

    def mark_request_finished(self, req: Req) -> None:
        tracked = self.requests.get(req.rid)
        final_n = len(getattr(req, "output_ids", []))

        if tracked is not None and final_n > 0:
            h = tracked.timeline.history
            last_n = h[-1].completion_number if h and isinstance(h[-1], RequestDecodeEvent) else 0
            base_ts = h[-1].end_timestamp if h else tracked.arrival_timestamp
            cur_ts = base_ts
            dur = 0.0
            for n in range(last_n + 1, final_n + 1):
                ctx = len(req.origin_input_ids) + n
                dur = isolated_decode_time_estimation(ctx, ctx, 1, self.fairinf_n)
                cur_ts += dur
                if self.enable_timeline_logging:
                    TIMELINE_WRITER.mark_isolated_decode_done(req.rid, req.uid, timestamp_iso=_iso_ts(cur_ts), completion_number=n)
            if final_n > last_n:
                start_events = [e for e in h if isinstance(e, RequestStartEvent)]
                tracked.timeline.history = start_events + [
                    RequestDecodeEvent(req_id=req.rid, duration=dur, end_timestamp=cur_ts, completion_number=final_n)
                ]
            tracked.latest_simulated_completion_timestamp = cur_ts

        if tracked is not None and tracked.latest_simulated_completion_timestamp is not None:
            if self.enable_timeline_logging:
                TIMELINE_WRITER.mark_isolated_completed(
                    req.rid, req.uid, timestamp_iso=_iso_ts(tracked.latest_simulated_completion_timestamp)
                )

        ut = self.users.get(req.uid)
        if ut is not None:
            ut.finished_request(req.rid)

        self.requests.pop(req.rid, None)
        self.most_recent_event_real.pop(req.rid, None)

    def build_deadline_candidates(
        self,
        waiting_queue: List[Req],
        running_batch,
        *,
        include_ordered_waiting_queue: bool = False,
        req_is_fair_prefill: Callable[[Req, object], bool],
        req_is_fair_decode: Callable[[Req, object], bool],
        event_delta_seconds: Callable[[TrackedRequest, RequestEvent], float],
        pooled_prefill_estimate_seconds: Callable[[Req], float],
        pooled_decode_estimate_seconds: Callable[[Req, object], float],
    ):
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

            upcoming = tracked.timeline.events_after(real_event)

            if not upcoming and rid in running_by_rid and isinstance(
                real_event, (RequestPrefillEvent, RequestDecodeEvent)
            ):
                next_n = (
                    real_event.completion_number + 1
                    if isinstance(real_event, RequestDecodeEvent)
                    else max(1, len(tracked.req.output_ids) + 1)
                )
                ctx = len(tracked.req.origin_input_ids) + next_n
                dur = isolated_decode_time_estimation(ctx, ctx, 1, self.fairinf_n)
                upcoming = [RequestDecodeEvent(
                    req_id=rid, duration=dur,
                    end_timestamp=float(real_event.end_timestamp) + dur,
                    completion_number=next_n,
                )]

            req = tracked.req
            for event in upcoming:
                deadline = event.end_timestamp + event_delta_seconds(tracked, event)
                if isinstance(event, RequestPrefillEvent):
                    if rid not in waiting_by_rid:
                        continue
                    if not req_is_fair_prefill(req, running_batch):
                        waiting_prefill_deadline_by_rid[rid] = float("inf")
                        continue
                    start_dl = deadline - pooled_prefill_estimate_seconds(req)
                    candidates.append(DeadlineCandidate(deadline=deadline, start_deadline=start_dl, event_type="prefill", req=req, event=event))
                    waiting_prefill_deadline_by_rid[rid] = start_dl
                elif isinstance(event, RequestDecodeEvent):
                    if rid not in running_by_rid:
                        continue
                    if not req_is_fair_decode(req, running_batch):
                        continue
                    start_dl = deadline - pooled_decode_estimate_seconds(req, running_batch)
                    candidates.append(DeadlineCandidate(deadline=deadline, start_deadline=start_dl, event_type="decode", req=req, event=event))

        candidates.sort(key=lambda c: (c.start_deadline, 0 if c.event_type == "decode" else 1, c.deadline, self.requests[c.req.rid].arrival_timestamp))

        if not include_ordered_waiting_queue:
            return candidates, waiting_prefill_deadline_by_rid
        return candidates, waiting_prefill_deadline_by_rid, tuple(c.req for c in candidates if c.event_type == "prefill")
