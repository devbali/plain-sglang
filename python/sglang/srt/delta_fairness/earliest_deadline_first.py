from __future__ import annotations

"""Delta fairness variant with forced-prefill disabled."""

from typing import Dict, List, Optional, Sequence, Tuple

from .delta_fairness_policy import DeltaFairnessPolicy
from .time_estimation import (
    isolated_decode_time_estimation,
    isolated_prefill_time_estimation,
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder

import time
from sglang.srt.request_timeline import TIMELINE_WRITER

class Event  ():
    def __init__ (self, duration=None, end_timestamp=None):
        self.timestamp = end_timestamp if end_timestamp is not None else time.time()
        self.duration = duration
        self.end_timestamp = end_timestamp if end_timestamp is not None else time.time()

class UserEvent (Event):
    pass

class UserDecodeEvent (UserEvent):
    pass

class UserPrefillEvent (UserEvent):
    pass

class UserDoNothingEvent (UserEvent):
    pass

class RequestEvent (Event):
    def __init__ (self, req_id, duration=None, end_timestamp=None):
        self.req_id = req_id
        super().__init__(duration=duration, end_timestamp=end_timestamp)
    
    def is_logically_after (self, e: RequestEvent):
        pass

class RequestDecodeEvent (RequestEvent):
    def __init__ (self, completion_number, *args, **kwargs):
        self.completion_number = completion_number
        super().__init__(*args, **kwargs)
    
    def is_logically_after (self, e: RequestEvent):
        assert e.req_id == self.req_id
        if isinstance[e, RequestDecodeEvent]:
            return self.completion_number > e.completion_number
        return True

class RequestPrefillEvent (RequestEvent):
    def is_logically_after (self, e: RequestEvent):
        assert e.req_id == self.req_id
        if isinstance[e, RequestStartEvent]:
            return True
        return False

class RequestStartEvent (RequestEvent):
    def is_logically_after (self, e: RequestEvent):
        return False

ISOLATED_DECODE_TIME_ESTIMATION = isolated_decode_time_estimation
ISOLATED_PREFILL_TIME_ESTIMATION = isolated_prefill_time_estimation
POOLED_DECODE_TIME_ESTIMATION = pooled_decode_time_estimation
POOLED_PREFILL_TIME_ESTIMATION = pooled_prefill_time_estimation

class RequestTimeline ():
    def __init__ (self):
        self.history: List[RequestEvent] = []

        # should be just one decode in the current policy
        self.anticipated_future_events: List[RequestEvent] = []
    
    def prune_till_after_event (self, e: RequestEvent) -> Optional[RequestEvent]:
        # go through our history, and remove any event that is not after e
        #  but don't empty out history, keep the most recent event
        # return the list of events that is after e, if such exists

        if len(self.history) == 0:
            return None

        target_i = None
        for i, h in enumerate(self.history):
            if not h.is_logically_after(e):
                continue
            
            # found something that is after e
            target_i = i
            break
        
        if target_i is None:
            self.history = self.history[-1]
            return None

        self.history = self.history[target_i:]
        return self.history
                

USER_CONFIG = {
    "max_prefill_tokens": 4096,
    "max_running_batch": 256,
}

class UserTimeline ():
    def __init__ (self, uid):
        self.history: List[UserEvent] = []
        
        # for the current policy, anticipated future should be a single decode atmost
        self.anticipated_future_events: List[UserEvent] = []

        self.uid = uid
        self.active_request_timelines : dict[str, "TrackedRequest"] = {} # req_id -> TrackedRequest
        self.waiting_request_timelines: dict[str, "TrackedRequest"] = {}
        self.finished_request_timelines: dict[str, "TrackedRequest"] = {}
    
    def add_request (self, req_id, tracked_request):
        self.waiting_request_timelines[req_id] = tracked_request
        tracked_request.add_event(RequestStartEvent(req_id, 0, time.time()))
    
    def finished_request (self, req_id):
        assert req_id in self.active_request_timelines, "Trying to finish request that is not active in user timeline"
        tracked_request = self.active_request_timelines.pop(req_id)
        self.finished_request_timelines[req_id] = tracked_request
    
    def nullify_anticipated_events (self):
        self.anticipated_future_events = []
        
        for tracked_req in self.active_request_timelines.values():
            tracked_req.alternate_history_timeline.anticipated_future_events = []
    
    def schedule_prefill (self, start_time = None) -> Optional[float]:
        # list of waiting request sorted by time of entry
        reqs = list(sorted(self.waiting_request_timelines.values(), key=lambda tracked_req: tracked_req.most_recent_event().end_timestamp))
        
        if start_time is None and len(reqs) > 0:
            start_time = reqs[0].most_recent_event().end_timestamp
        
        if start_time is None:
            return None # no waiting requests to schedule, and nothing has been done so far

        remaining_slots = USER_CONFIG["max_running_batch"] - len(self.active_request_timelines)
        if remaining_slots <= 0:
            return start_time

        batch = []
        batch_size = 0
        get_prompt_tokens = lambda tracked_req: len(tracked_req.req.origin_input_ids)
        for tracked_req in reqs:
            if tracked_req.most_recent_event.end_timestamp > start_time:
                break # all subsequent requests will also be after start_time, since they are sorted by time of entry
            
            assert get_prompt_tokens(tracked_req) <= USER_CONFIG["max_prefill_tokens"], "Request has more prompt tokens than the user max prefill tokens, cannot schedule"
            
            if batch_size + get_prompt_tokens(tracked_req) > USER_CONFIG["max_prefill_tokens"]:
                continue

            if len(batch) >= remaining_slots:
                break
            
            batch.append(tracked_req)
            batch_size += get_prompt_tokens(tracked_req)
        if not batch:
            return start_time

        batch_duration = ISOLATED_PREFILL_TIME_ESTIMATION(batch_size, max(get_prompt_tokens(tracked_req) for tracked_req in batch), len(batch))
        batch_end_time = start_time + batch_duration

        # add prefill event in each request
        # move it from waiting to active queue in user timeline
        for tracked_req in batch:
            tracked_req.alternate_history_timeline.history.append(RequestPrefillEvent(tracked_req.req.rid, batch_duration, batch_end_time))
            # move request from waiting to active in user timeline
            self.active_request_timelines[tracked_req.req.rid] = tracked_req
            del self.waiting_request_timelines[tracked_req.req.rid]
        
        self.history.append(UserPrefillEvent(batch_duration, batch_end_time))
    
    def schedule_decode (self, start_time, req_id_real_statuses, anticipated=False) -> Optional[float]:
        # first check to see if any request would go into anticipation if we schedule a decode here
        # if so, simply return none and only anticipate the decode
        
        get_num_tokens = lambda req: len(req.fill_ids)
        batch = [] # (rid, completion #, num_tokens)

        req_id_decodes: dict[str, int] = {} # req id to number of completion tokens done in real
        for rid, event in req_id_real_statuses.values():
            if isinstance[event, RequestDecodeEvent]:
                req_id_decodes[rid] = event.completion_number
            else:
                req_id_decodes[rid] = 0

        assert len(self.active_request_timelines) <= USER_CONFIG["max_running_batch"], "Active running request more than max running batch for isolated setting, unfair user but still doing alternate history"
        
        for rid, tracked_req in self.active_request_timelines.items():
            most_recent : RequestDecodeEvent = tracked_req.most_recent_event()

            if rid not in req_id_decodes or req_id_decodes[rid] == 0:
                batch.append((rid, 1, get_num_tokens(tracked_req.req)))
                continue
            
            assert isinstance[most_recent, RequestDecodeEvent]
            
            if most_recent.completion_number > req_id_decodes[rid]:
                # we have already done n + 1, we must only anticipate from now on
                anticipated = True

            batch.append((rid, most_recent.completion_number + 1, get_num_tokens(tracked_req.req)))

        batch_duration = ISOLATED_DECODE_TIME_ESTIMATION(
            sum([b[2] for b in batch]), 
            max([b[2] for b in batch]),
            len(batch)
        )
        batch_end_time = start_time + batch_duration

        if not anticipated:
            # this decode has every request confirmed to have this completion token
            # mark this in every request's timeline and this user's timeline
            for rid, completion_number, num_tokens in batch:
                tracked_req = self.active_request_timelines[rid]
                tracked_req.alternate_history_timeline.history.append(
                    RequestDecodeEvent(completion_number, rid, duration=batch_duration, end_timestamp=batch_end_time))

            self.history.append(UserDecodeEvent(duration=batch_duration, end_timestamp=batch_end_time))
            
            # if we just had a fresh decode, the anticipated is nullified. let's do the anticipated one again
            self.nullify_anticipated_events()
            self.schedule_decode(batch_end_time, req_id_decodes, anticipated=True)
            return batch_end_time
        
        elif len(self.anticipated_future_events) == 0:
            for rid, completion_number, num_tokens in batch:
                tracked_req = self.active_request_timelines[rid]
                tracked_req.alternate_history_timeline.anticipated_future_events.append(
                    RequestDecodeEvent(completion_number, rid, duration=batch_duration, end_timestamp=batch_end_time))

            self.anticipated_future_events.append(UserDecodeEvent(duration=batch_duration, end_timestamp=batch_end_time))

        return None

    def complete_upto_time (self, timestamp, req_id_real_statuses):
        # prefill prioritizing alternate history
        # first, complete all the prefills from the requests in the waiting queue
        # then do decode batches
        # if any of these can not start before timestamp, abort
        # only schedule decodes upto when the req_id_decodes last, or 1 (first decode is assumed to happen),
        #   rest should be anticipated, abort

        if len(self.history) == 0:
            # no events in history, start from the earliest waiting request
            start_time = self.schedule_prefill()
        
        else:
            last_event = self.history[-1]
            start_time = last_event.end_timestamp
        
        while start_time is not None and start_time < timestamp:
            if (
                len(self.waiting_request_timelines) > 0
                and len(self.active_request_timelines) < USER_CONFIG["max_running_batch"]
            ):
                # schedule a prefill batch
                start_time = self.schedule_prefill(start_time)
            elif len(self.active_request_timelines) > 0:
                # schedule a decode batch
                # if schedule decode "anticipated" a batch, did not do it for real,
                #  the time will return as None, which will exit the loop
                start_time = self.schedule_decode(start_time, req_id_real_statuses)

class TrackedRequest ():
    def __init__ (self, req: Req, user_timeline: UserTimeline, deltas_in_microseconds=None):
        self.req = req
        self.alternate_history_timeline = RequestTimeline()
        self.deltas_in_microseconds = {
            "first_decode": 0,
            "decode": 0,
            "prefill": 0
        }  if deltas_in_microseconds is None else deltas_in_microseconds
        self.user_timeline = user_timeline

    def most_recent_event (self):
        if len(self.alternate_history_timeline.history) == 0:
            return None
        return self.alternate_history_timeline.history[-1]

    def earliest_events_after_real_time (self, real_e):
        earliest_events = self.alternate_history_timeline.prune_till_after_event(real_e)

        if earliest_events != None:
            return earliest_events

        if len(self.alternate_history_timeline.anticipated_future_events) > 0:
            e = self.alternate_history_timeline.anticipated_future_events[-1]
            if e.is_logically_after(real_e):
                return [e]

class EventQueue ():
    def __init__ (self):
        self.users: dict[str, UserTimeline] = {} # user_id -> UserTimeline
        self.requests: dict[str, TrackedRequest] = {} # req_id -> TrackedRequest object
        self.most_recent_event_real : dict[str, RequestEvent] = {} # req_id -> event encompassing logically what's the most recent

    def start_of_pass (self, fair_users):
        # first, remove all users that are not fair anymore
        for uid in list(self.users.keys()):
            if uid not in fair_users:
                del self.users[uid]
            
            for rid, tracked_req in list(self.requests.items()):
                if tracked_req.req.uid == uid:
                    del self.requests[rid]
        
        current_time = time.time()
        # next, schedule the alternate histories upto now
        for user_timeline in self.users.values():
            user_timeline.complete_upto_time(current_time, self.most_recent_event_real)

    def finished_decode (self, batch: ScheduleBatch):
        for req in batch.reqs:
            completion_tokens_done = len(req.output_ids)
            self.most_recent_event_real[req.rid] = RequestDecodeEvent(completion_tokens_done)
    
    def finished_prefill (self, batch):
        # ignoring chunking of prefills for now
        for req in batch.reqs:
            self.most_recent_event_real[req.rid] = RequestPrefillEvent()

    def process_new_request (self, req: "Req", deltas=None):
        if req.rid not in self.requests:
            self.requests[req.rid] = TrackedRequest(req, deltas)
        self.most_recent_event_real[req.rid] = RequestStartEvent()
    
    def mark_request_finished (self, req: "Req"):
        if req.uid in self.users:
            self.users[req.uid].finished_request(req.rid)
        if req.rid in self.requests:
            del self.requests[req.rid]
        if req.rid in self.most_recent_event_real:
            del self.most_recent_event_real[req.rid]

class EarliestDeltaFirst (DeltaFairnessPolicy):
    """Delta fairness policy that maintains a single queue of all upcoming fair events, and chooses the earliest deadline first.
    Maintains an alternate history of each user's isolated decodes and computes forced decodes"""

    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_queue = EventQueue()
    
    def sorted_waiting_queue(self, waiting_queue: List["Req"]):
        return waiting_queue
    
    # def fairinf_prioritize_force_prefill():
    #     return False

    
    def finished_decode (self, batch: "ScheduleBatch"):
        super().finished_decode(batch)
        self.event_queue.finished_decode(batch)

    def finished_prefill (self, batch: "ScheduleBatch"):
        super().finished_prefill(batch)
        self.event_queue.finished_prefill(batch)
    
    def mark_request_finished (self, req):
        self.event_queue.mark_request_finished(req)

    def start_of_pass (self, running_batch, waiting_queue):
        reqs = [*running_batch.reqs, *waiting_queue]
        users = [r.uid for r in reqs]
        fair_users = [self.user_is_fair_prefill(user, running_batch=running_batch) for user in users]
        self.event_queue.start_of_pass(fair_users)
    
    def process_new_request (self, req: "Req"):
        self.event_queue.process_new_request(req, None)
    
    
    # todo implement more
    def __init__(self, *args, **kwargs):
        self._edf_pooled_quanta_us = int(
            kwargs.pop(
                "delta_fairness_pooled_quanta_us",
                kwargs.pop("delta_fairness_quanta_us", 0),
            )
            or 0
        )
        self._edf_exclusive_quanta_us = int(
            kwargs.pop("delta_fairness_exclusive_quanta_us", 0) or 0
        )
        runtime_max_prefill_tokens = kwargs.pop("max_prefill_tokens", None)
        runtime_max_running_batch = kwargs.get("max_running_requests", None)
        super().__init__(*args, **kwargs)
        self.event_queue = EventQueue()
        self._edf_deadline_queue = []
        self._edf_earliest = None
        self._edf_waiting_prefill_deadline_by_rid: Dict[str, float] = {}
        self._edf_deltas_us = {"prefill": 0, "first_decode": 0, "decode": 0}
        self._fairinf_log_path = "fairinf_log.csv"
        self._fairinf_pass_id = 0
        self._fairinf_log_sample_rate = 0.01
        self._fairinf_log_this_pass = False
        if runtime_max_prefill_tokens is not None:
            USER_CONFIG["max_prefill_tokens"] = int(runtime_max_prefill_tokens)
        if runtime_max_running_batch is not None:
            per_user_running_batch = max(1, int(runtime_max_running_batch) // max(int(self.delta_fairness_n or 1), 1))
            USER_CONFIG["max_running_batch"] = per_user_running_batch
        self._patch_timeline_helpers()

    def fairinf_prioritize_force_prefill(self):
        # Compute global earliest event in prefill hook first, then decode hook consumes it.
        return True

    def _format_deadline_event(self, row) -> str:
        if row is None:
            return ""
        deadline, event_type, req = row
        return f"{req.rid}:{event_type}:{deadline:.6f}"

    def _format_deadline_queue(self) -> str:
        if not self._edf_deadline_queue:
            return ""
        return "|".join(self._format_deadline_event(row) for row in self._edf_deadline_queue)

    def _earliest_within_quanta(self) -> bool:
        if self._edf_earliest is None:
            return False
        if self._edf_pooled_quanta_us <= 0:
            return True
        deadline, _, _ = self._edf_earliest
        return (deadline - time.time()) <= (self._edf_pooled_quanta_us / 1_000_000.0)

    def _within_exclusive_quanta(self, deadline: float) -> bool:
        return (deadline - time.time()) <= (self._edf_exclusive_quanta_us / 1_000_000.0)

    def _write_fairinf_log(self, phase: str, action: str, note: str):
        if not self._fairinf_log_this_pass:
            return
        import csv
        import os

        row = [
            f"{time.time():.6f}",
            str(self._fairinf_pass_id),
            phase,
            action,
            self._format_deadline_event(self._edf_earliest),
            note,
            self._format_deadline_queue(),
        ]

        write_header = not os.path.exists(self._fairinf_log_path) or os.path.getsize(
            self._fairinf_log_path
        ) == 0
        with open(self._fairinf_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "timestamp",
                        "pass_id",
                        "phase",
                        "action",
                        "closest_event",
                        "note",
                        "edf_events_sorted",
                    ]
                )
            writer.writerow(row)

    def _format_fairness_snapshot(
        self,
        running_batch: Optional["ScheduleBatch"],
        waiting_queue: List["Req"],
    ) -> str:
        user_ids = sorted(
            {
                str(req.uid)
                for req in (running_batch.reqs if running_batch is not None else [])
            }
            | {str(req.uid) for req in waiting_queue}
        )
        parts = []
        for uid in user_ids:
            state = self.debug_user_fairness_state(uid, running_batch=running_batch)
            parts.append(
                (
                    f"{uid}:fair={int(bool(state['is_fair']))}:"
                    f"reason={state['reason']}:"
                    f"unevictable={state['unevictable_used']}/{state['unevictable_limit']}:"
                    f"running={state['running_count']}/{state['running_limit']}"
                )
            )
        return "|".join(parts)

    def _patch_timeline_helpers(self):
        # Patch buggy helper methods in this module at runtime so EDF can rely on
        # earliest_events_after_real_time without touching code above the TODO line.
        if getattr(RequestTimeline, "_edf_patch_applied", False):
            return

        def _prefill_after(self_e: "RequestPrefillEvent", other_e: "RequestEvent"):
            return isinstance(other_e, RequestStartEvent)

        def _decode_after(self_e: "RequestDecodeEvent", other_e: "RequestEvent"):
            if getattr(other_e, "req_id", None) != self_e.req_id:
                return False
            if isinstance(other_e, RequestDecodeEvent):
                return self_e.completion_number > other_e.completion_number
            return isinstance(other_e, (RequestStartEvent, RequestPrefillEvent))

        def _prune_after(self_tl: "RequestTimeline", real_e: "RequestEvent"):
            history = self_tl.history
            if not isinstance(history, list):
                history = [history] if history is not None else []
                self_tl.history = history
            if len(history) == 0:
                return None

            target_i = None
            for i, h in enumerate(history):
                try:
                    if h.is_logically_after(real_e):
                        target_i = i
                        break
                except Exception:
                    continue

            if target_i is None:
                self_tl.history = [history[-1]]
                return None

            self_tl.history = history[target_i:]
            return self_tl.history

        def _schedule_prefill(self_ut: "UserTimeline", start_time=None):
            reqs = sorted(
                self_ut.waiting_request_timelines.values(),
                key=lambda tracked_req: tracked_req.most_recent_event().end_timestamp,
            )
            if start_time is None and reqs:
                start_time = reqs[0].most_recent_event().end_timestamp
            if start_time is None:
                return None

            remaining_slots = USER_CONFIG["max_running_batch"] - len(
                self_ut.active_request_timelines
            )
            if remaining_slots <= 0:
                return start_time

            batch, batch_size, next_start_time = [], 0, None
            for tracked_req in reqs:
                tracked_start_time = tracked_req.most_recent_event().end_timestamp
                if tracked_start_time > start_time:
                    next_start_time = tracked_start_time
                    break
                prompt_tokens = len(tracked_req.req.origin_input_ids)
                assert (
                    prompt_tokens <= USER_CONFIG["max_prefill_tokens"]
                ), "Request has more prompt tokens than the user max prefill tokens, cannot schedule"
                if batch_size + prompt_tokens > USER_CONFIG["max_prefill_tokens"]:
                    continue
                if len(batch) >= remaining_slots:
                    break
                batch.append(tracked_req)
                batch_size += prompt_tokens

            if not batch:
                return start_time if remaining_slots <= 0 else next_start_time

            batch_duration = ISOLATED_PREFILL_TIME_ESTIMATION(
                batch_size,
                max(len(tracked_req.req.origin_input_ids) for tracked_req in batch),
                len(batch),
            )
            batch_end_time = start_time + batch_duration
            for tracked_req in batch:
                tracked_req.alternate_history_timeline.history.append(
                    RequestPrefillEvent(tracked_req.req.rid, batch_duration, batch_end_time)
                )
                self_ut.active_request_timelines[tracked_req.req.rid] = tracked_req
                del self_ut.waiting_request_timelines[tracked_req.req.rid]
            self_ut.history.append(UserPrefillEvent(batch_duration, batch_end_time))
            return batch_end_time

        def _schedule_decode(
            self_ut: "UserTimeline", start_time, req_id_real_statuses, anticipated=False
        ):
            req_id_decodes = {
                rid: (event.completion_number if isinstance(event, RequestDecodeEvent) else 0)
                for rid, event in req_id_real_statuses.items()
            }
            assert (
                len(self_ut.active_request_timelines) <= USER_CONFIG["max_running_batch"]
            ), "Active running request more than max running batch for isolated setting, unfair user but still doing alternate history"

            batch = []
            for rid, tracked_req in self_ut.active_request_timelines.items():
                most_recent = tracked_req.most_recent_event()
                real_decode = req_id_decodes.get(rid, 0)
                req_token_count = len(
                    tracked_req.req.fill_ids
                    if tracked_req.req.fill_ids is not None
                    else (tracked_req.req.origin_input_ids + tracked_req.req.output_ids)
                )
                if real_decode == 0:
                    batch.append((rid, 1, req_token_count))
                    continue
                if not isinstance(most_recent, RequestDecodeEvent):
                    batch.append((rid, real_decode + 1, req_token_count))
                    continue
                anticipated = anticipated or most_recent.completion_number > real_decode
                batch.append((rid, most_recent.completion_number + 1, req_token_count))

            if not batch:
                return None

            batch_duration = ISOLATED_DECODE_TIME_ESTIMATION(
                sum(num_tokens for _, _, num_tokens in batch),
                max(num_tokens for _, _, num_tokens in batch),
                len(batch),
            )
            batch_end_time = start_time + batch_duration
            target_attr = "anticipated_future_events" if anticipated else "history"
            for rid, completion_number, _ in batch:
                tracked_req = self_ut.active_request_timelines[rid]
                getattr(tracked_req.alternate_history_timeline, target_attr).append(
                    RequestDecodeEvent(
                        completion_number,
                        rid,
                        duration=batch_duration,
                        end_timestamp=batch_end_time,
                    )
                )

            if anticipated:
                if not self_ut.anticipated_future_events:
                    self_ut.anticipated_future_events.append(
                        UserDecodeEvent(duration=batch_duration, end_timestamp=batch_end_time)
                    )
                return None

            self_ut.history.append(
                UserDecodeEvent(duration=batch_duration, end_timestamp=batch_end_time)
            )
            self_ut.nullify_anticipated_events()
            self_ut.schedule_decode(batch_end_time, req_id_real_statuses, anticipated=True)
            return batch_end_time

        def _next_prefill_end_time(self_ut: "UserTimeline", start_time=None):
            reqs = sorted(
                self_ut.waiting_request_timelines.values(),
                key=lambda tracked_req: tracked_req.most_recent_event().end_timestamp,
            )
            if start_time is None and reqs:
                start_time = reqs[0].most_recent_event().end_timestamp
            if start_time is None:
                return None

            remaining_slots = USER_CONFIG["max_running_batch"] - len(
                self_ut.active_request_timelines
            )
            if remaining_slots <= 0:
                return None

            batch = []
            batch_size = 0
            for tracked_req in reqs:
                tracked_start_time = tracked_req.most_recent_event().end_timestamp
                if tracked_start_time > start_time:
                    break
                prompt_tokens = len(tracked_req.req.origin_input_ids)
                if prompt_tokens > USER_CONFIG["max_prefill_tokens"]:
                    continue
                if batch_size + prompt_tokens > USER_CONFIG["max_prefill_tokens"]:
                    continue
                if len(batch) >= remaining_slots:
                    break
                batch.append(tracked_req)
                batch_size += prompt_tokens

            if not batch:
                return None

            batch_duration = ISOLATED_PREFILL_TIME_ESTIMATION(
                batch_size,
                max(len(tracked_req.req.origin_input_ids) for tracked_req in batch),
                len(batch),
            )
            return start_time + batch_duration

        def _next_decode_end_time(self_ut: "UserTimeline", start_time, req_id_real_statuses):
            if len(self_ut.active_request_timelines) == 0:
                return None

            req_id_decodes = {
                rid: (event.completion_number if isinstance(event, RequestDecodeEvent) else 0)
                for rid, event in req_id_real_statuses.items()
            }
            batch = []
            for rid, tracked_req in self_ut.active_request_timelines.items():
                req_token_count = len(
                    tracked_req.req.fill_ids
                    if tracked_req.req.fill_ids is not None
                    else (tracked_req.req.origin_input_ids + tracked_req.req.output_ids)
                )
                real_decode = req_id_decodes.get(rid, 0)
                if real_decode == 0:
                    batch.append((rid, 1, req_token_count))
                    continue

                most_recent = tracked_req.most_recent_event()
                if isinstance(most_recent, RequestDecodeEvent):
                    completion_number = max(
                        getattr(most_recent, "completion_number", 0) + 1,
                        real_decode + 1,
                    )
                else:
                    completion_number = real_decode + 1
                batch.append((rid, completion_number, req_token_count))

            if not batch:
                return None

            batch_duration = ISOLATED_DECODE_TIME_ESTIMATION(
                sum(num_tokens for _, _, num_tokens in batch),
                max(num_tokens for _, _, num_tokens in batch),
                len(batch),
            )
            return start_time + batch_duration

        def _complete_upto_time(self_ut: "UserTimeline", timestamp, req_id_real_statuses):
            if len(self_ut.history) == 0:
                waiting_reqs = sorted(
                    self_ut.waiting_request_timelines.values(),
                    key=lambda tracked_req: tracked_req.most_recent_event().end_timestamp,
                )
                start_time = (
                    waiting_reqs[0].most_recent_event().end_timestamp if waiting_reqs else None
                )
            else:
                start_time = self_ut.history[-1].end_timestamp

            while start_time is not None and start_time < timestamp:
                next_prefill_end = _next_prefill_end_time(self_ut, start_time)
                next_decode_end = _next_decode_end_time(self_ut, start_time, req_id_real_statuses)

                if next_prefill_end is None and next_decode_end is None:
                    break
                if next_decode_end is None:
                    start_time = self_ut.schedule_prefill(start_time)
                    continue
                if next_prefill_end is None:
                    start_time = self_ut.schedule_decode(start_time, req_id_real_statuses)
                    continue

                if next_decode_end <= next_prefill_end:
                    start_time = self_ut.schedule_decode(start_time, req_id_real_statuses)
                else:
                    start_time = self_ut.schedule_prefill(start_time)

        RequestPrefillEvent.is_logically_after = _prefill_after
        RequestDecodeEvent.is_logically_after = _decode_after
        RequestTimeline.prune_till_after_event = _prune_after
        UserTimeline.schedule_prefill = _schedule_prefill
        UserTimeline.schedule_decode = _schedule_decode
        UserTimeline.complete_upto_time = _complete_upto_time
        RequestTimeline._edf_patch_applied = True

    def _read_deltas(self, delta_fairness_deltas_microseconds: Optional[Dict[str, int]]):
        deltas = delta_fairness_deltas_microseconds or {}
        self._edf_deltas_us = {
            "prefill": deltas.get("prefill", deltas.get("prefill_running_batch", 0)),
            "first_decode": deltas.get(
                "first_decode",
                deltas.get("first_decode_running_batch", deltas.get("decode_running_batch", 0)),
            ),
            "decode": deltas.get("decode", deltas.get("decode_running_batch", 0)),
        }

    def _event_delta_seconds(self, tracked_req: "TrackedRequest", e: "RequestEvent") -> float:
        req_deltas = tracked_req.deltas_in_microseconds
        if isinstance(e, RequestPrefillEvent):
            delta_us = req_deltas.get("prefill", self._edf_deltas_us["prefill"])
        elif isinstance(e, RequestDecodeEvent):
            if getattr(e, "completion_number", 0) <= 1:
                delta_us = req_deltas.get("first_decode", self._edf_deltas_us["first_decode"])
            else:
                delta_us = req_deltas.get("decode", self._edf_deltas_us["decode"])
        else:
            delta_us = 0
        return float(delta_us) / 1_000_000.0

    def _build_deadline_queue(
        self,
        waiting_queue: List["Req"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]],
        *,
        log_calc: bool = True,
        sync_live_state: bool = False,
    ):
        self._read_deltas(delta_fairness_deltas_microseconds)
        if sync_live_state:
            self._sync_fair_user_tracking(running_batch, waiting_queue)
        waiting_by_rid = {r.rid: r for r in waiting_queue}
        running_by_rid = {r.rid: r for r in (running_batch.reqs if running_batch else [])}

        queue = []
        waiting_prefill_deadline_by_rid: Dict[str, float] = {}
        stats = {
            "tracked": 0,
            "missing_real": 0,
            "timeline_exc": 0,
            "upcoming_empty": 0,
            "prefill_filtered_not_waiting": 0,
            "prefill_filtered_unfair": 0,
            "decode_filtered_not_running": 0,
            "decode_filtered_unfair": 0,
            "fallback_decode_added": 0,
        }

        for rid, tracked_req in self.event_queue.requests.items():
            stats["tracked"] += 1
            real_e = self.event_queue.most_recent_event_real.get(rid)
            if real_e is None:
                stats["missing_real"] += 1
                continue

            if not isinstance(tracked_req.alternate_history_timeline.history, list):
                tracked_req.alternate_history_timeline.history = [
                    tracked_req.alternate_history_timeline.history
                ]

            try:
                upcoming_events = tracked_req.earliest_events_after_real_time(real_e) or []
            except Exception:
                # Keep EDF robust even if alternate timeline logic throws.
                stats["timeline_exc"] += 1
                continue

            if len(upcoming_events) == 0:
                stats["upcoming_empty"] += 1

            req = tracked_req.req
            added_decode_for_req = False
            for e in upcoming_events:
                if isinstance(e, RequestPrefillEvent):
                    if rid not in waiting_by_rid:
                        stats["prefill_filtered_not_waiting"] += 1
                        continue
                    if not self.req_is_fair_prefill(req, running_batch=running_batch):
                        stats["prefill_filtered_unfair"] += 1
                        continue
                    deadline = e.end_timestamp + self._event_delta_seconds(tracked_req, e)
                    queue.append((deadline, "prefill", req))
                    prev = waiting_prefill_deadline_by_rid.get(rid)
                    if prev is None or deadline < prev:
                        waiting_prefill_deadline_by_rid[rid] = deadline
                elif isinstance(e, RequestDecodeEvent):
                    if rid not in running_by_rid:
                        stats["decode_filtered_not_running"] += 1
                        continue
                    if not self.req_is_fair_decode(req, running_batch=running_batch):
                        stats["decode_filtered_unfair"] += 1
                        continue
                    deadline = e.end_timestamp + self._event_delta_seconds(tracked_req, e)
                    queue.append((deadline, "decode", req))
                    added_decode_for_req = True

            # Fallback: if a fair running request somehow has no timeline-produced upcoming
            # decode, synthesize the anticipated next decode so EDF always sees a candidate.
            if (
                rid in running_by_rid
                and not added_decode_for_req
                and self.req_is_fair_decode(req, running_batch=running_batch)
            ):
                if isinstance(real_e, RequestDecodeEvent):
                    completion_number = getattr(real_e, "completion_number", len(req.output_ids))
                    synth_event = RequestDecodeEvent(
                        completion_number + 1,
                        req.rid,
                        0,
                        max(time.time(), getattr(real_e, "end_timestamp", time.time())),
                    )
                else:
                    synth_event = RequestDecodeEvent(
                        1,
                        req.rid,
                        0,
                        max(time.time(), getattr(real_e, "end_timestamp", time.time())),
                    )
                deadline = synth_event.end_timestamp + self._event_delta_seconds(
                    tracked_req, synth_event
                )
                queue.append((deadline, "decode", req))
                stats["fallback_decode_added"] += 1

        queue.sort(key=lambda x: (x[0], 0 if x[1] == "decode" else 1))
        self._edf_deadline_queue = queue
        self._edf_earliest = queue[0] if queue else None
        self._edf_waiting_prefill_deadline_by_rid = waiting_prefill_deadline_by_rid
        if log_calc:
            self._write_fairinf_log(
                "deadline_calc",
                "built_deadlines",
                (
                    f"waiting={len(waiting_queue)},running={len(running_by_rid)},"
                    f"candidates={len(queue)},tracked={stats['tracked']},"
                    f"missing_real={stats['missing_real']},timeline_exc={stats['timeline_exc']},"
                    f"upcoming_empty={stats['upcoming_empty']},"
                    f"decode_not_running={stats['decode_filtered_not_running']},"
                    f"decode_unfair={stats['decode_filtered_unfair']},"
                    f"fallback_decode_added={stats['fallback_decode_added']}"
                ),
            )

    def fairinf_force_prefill_any_waiting(
        self,
        waiting_queue: List["Req"],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional["ScheduleBatch"] = None,
    ) -> bool:
        import random

        self._fairinf_pass_id += 1
        self._fairinf_log_this_pass = (
            random.random() < self._fairinf_log_sample_rate
        )
        if not self.delta_fairness_n:
            self._edf_deadline_queue = []
            self._edf_earliest = None
            self._edf_waiting_prefill_deadline_by_rid = {}
            self._write_fairinf_log(
                "start_of_pass",
                "pass_begin",
                f"waiting={len(waiting_queue)},running={len(running_batch.reqs) if running_batch else 0},tracked_reqs={len(self.event_queue.requests)}",
            )
            self._write_fairinf_log(
                "deadline_calc",
                "skipped",
                "fairness disabled",
            )
            return False

        self._build_deadline_queue(
            waiting_queue,
            running_batch,
            delta_fairness_deltas_microseconds,
            log_calc=False,
            sync_live_state=True,
        )
        self._write_fairinf_log(
            "start_of_pass",
            "pass_begin",
            f"waiting={len(waiting_queue)},running={len(running_batch.reqs) if running_batch else 0},tracked_reqs={len(self.event_queue.requests)}",
        )
        self._write_fairinf_log(
            "deadline_calc",
            "built_deadlines",
            f"waiting={len(waiting_queue)},running={len(running_batch.reqs) if running_batch else 0},candidates={len(self._edf_deadline_queue)}",
        )
        if not self._earliest_within_quanta():
            self._write_fairinf_log(
                "force_decision",
                "no_force_prefill",
                f"nearest deadline not within pooled_quanta_us={self._edf_pooled_quanta_us}",
            )
            return False

        force_prefill = self._edf_earliest is not None and self._edf_earliest[1] == "prefill"
        self._write_fairinf_log(
            "force_decision",
            "force_prefill" if force_prefill else "no_force_prefill",
            "closest event drives force prefill decision",
        )
        return force_prefill

    def fairinf_force_decode(
        self,
        running_batch: Optional["ScheduleBatch"],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        decode_time_us: int = 20000,
    ) -> Tuple[bool, Optional[int]]:
        if not self.delta_fairness_n or running_batch is None:
            self._write_fairinf_log(
                "force_decision",
                "no_force_decode",
                "fairness disabled or no running batch",
            )
            return False, None
        if not self._earliest_within_quanta():
            self._write_fairinf_log(
                "force_decision",
                "no_force_decode",
                f"nearest deadline not within pooled_quanta_us={self._edf_pooled_quanta_us}",
            )
            return False, None
        if self._edf_earliest is not None and self._edf_earliest[1] == "decode":
            # Decode has the earliest global deadline; do decode batch.
            self._write_fairinf_log(
                "force_decision",
                "force_decode",
                "closest event is decode",
            )
            return True, 0
        self._write_fairinf_log(
            "force_decision",
            "no_force_decode",
            "closest event is not decode",
        )
        return False, None

    def fairinf_overdue_decode_subset_rids(
        self,
        running_batch: Optional["ScheduleBatch"],
    ) -> Optional[set[str]]:
        if (
            running_batch is None
            or self._edf_earliest is None
            or self._edf_earliest[1] != "decode"
        ):
            return None

        earliest_deadline, _, _ = self._edf_earliest
        if not self._within_exclusive_quanta(earliest_deadline):
            return None

        overdue_decode_uids = {
            req.uid
            for deadline, event_type, req in self._edf_deadline_queue
            if event_type == "decode" and self._within_exclusive_quanta(deadline)
        }
        if not overdue_decode_uids:
            return None

        selected_rids = {
            req.rid
            for req in running_batch.reqs
            if req.uid in overdue_decode_uids
        }
        return selected_rids or None

    def sorted_waiting_queue(self, waiting_queue: List["Req"]):
        # For forced prefill, prioritize waiting requests with earliest prefill deadlines.
        indexed = list(enumerate(waiting_queue))
        indexed.sort(
            key=lambda x: (
                0 if x[1].rid in self._edf_waiting_prefill_deadline_by_rid else 1,
                self._edf_waiting_prefill_deadline_by_rid.get(x[1].rid, float("inf")),
                x[0],
            )
        )
        overdue = [
            req
            for _, req in indexed
            if self._within_exclusive_quanta(
                self._edf_waiting_prefill_deadline_by_rid.get(req.rid, float("inf"))
            )
        ]
        if overdue:
            return overdue
        return [x[1] for x in indexed]

    def _ensure_tracked_request(self, req: "Req") -> "TrackedRequest":
        tracked = self.event_queue.requests.get(req.rid)
        if tracked is None:
            tracked = TrackedRequest(req, None, self._edf_deltas_us.copy())
            self.event_queue.requests[req.rid] = tracked
        else:
            tracked.req = req
            if tracked.deltas_in_microseconds is None:
                tracked.deltas_in_microseconds = self._edf_deltas_us.copy()
        if not isinstance(tracked.alternate_history_timeline.history, list):
            tracked.alternate_history_timeline.history = (
                [tracked.alternate_history_timeline.history]
                if tracked.alternate_history_timeline.history is not None
                else []
            )
        return tracked

    def _seed_request_tracking(
        self,
        tracked: "TrackedRequest",
        user_timeline: "UserTimeline",
        now: float,
        *,
        waiting: bool,
    ):
        tracked.user_timeline = user_timeline
        real_e = self.event_queue.most_recent_event_real.get(tracked.req.rid)
        event_ts = getattr(real_e, "end_timestamp", now)
        if waiting:
            tracked.alternate_history_timeline.history = [
                RequestStartEvent(tracked.req.rid, 0, event_ts)
            ]
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestPrefillEvent(tracked.req.rid, 0, event_ts)
            ]
            return
        completion_number = 0
        if isinstance(real_e, RequestDecodeEvent):
            completion_number = getattr(real_e, "completion_number", 0) or 0
        else:
            completion_number = len(tracked.req.output_ids)
        if completion_number > 0:
            tracked.alternate_history_timeline.history = [
                RequestDecodeEvent(completion_number, tracked.req.rid, 0, event_ts)
            ]
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestDecodeEvent(completion_number + 1, tracked.req.rid, 0, event_ts)
            ]
        else:
            tracked.alternate_history_timeline.history = [
                RequestPrefillEvent(tracked.req.rid, 0, event_ts)
            ]
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestDecodeEvent(1, tracked.req.rid, 0, event_ts)
            ]

    def _set_live_request_state(
        self,
        req: "Req",
        tracked: "TrackedRequest",
        user_timeline: "UserTimeline",
        now: float,
        *,
        waiting: bool,
    ):
        target = (
            user_timeline.waiting_request_timelines
            if waiting
            else user_timeline.active_request_timelines
        )
        other = (
            user_timeline.active_request_timelines
            if waiting
            else user_timeline.waiting_request_timelines
        )
        should_seed = tracked.user_timeline is not user_timeline or req.rid not in target
        if should_seed:
            self._seed_request_tracking(tracked, user_timeline, now, waiting=waiting)
        other.pop(req.rid, None)
        target[req.rid] = tracked
        real_e = self.event_queue.most_recent_event_real.get(req.rid)
        if waiting:
            # Preserve the original waiting-start timestamp while a request remains queued.
            # Resetting it every sync pass keeps pushing the prefill deadline forward and
            # prevents fair waiting requests from ever becoming force-prefill candidates.
            if not isinstance(real_e, RequestStartEvent):
                self.event_queue.most_recent_event_real[req.rid] = RequestStartEvent(req.rid, 0, now)
        elif isinstance(real_e, RequestDecodeEvent):
            self.event_queue.most_recent_event_real[req.rid] = RequestDecodeEvent(
                getattr(real_e, "completion_number", len(req.output_ids)),
                req.rid,
                0,
                now,
            )
        elif len(req.output_ids) > 0:
            self.event_queue.most_recent_event_real[req.rid] = RequestDecodeEvent(
                len(req.output_ids), req.rid, 0, now
            )
        else:
            self.event_queue.most_recent_event_real[req.rid] = RequestPrefillEvent(
                req.rid, 0, now
            )

    def _sync_fair_user_tracking(self, running_batch, waiting_queue):
        now = time.time()
        running_reqs = list(running_batch.reqs) if running_batch is not None else []
        waiting_by_user: Dict[str, List["Req"]] = {}
        running_by_user: Dict[str, List["Req"]] = {}
        for req in waiting_queue:
            waiting_by_user.setdefault(req.uid, []).append(req)
        for req in running_reqs:
            running_by_user.setdefault(req.uid, []).append(req)

        fair_users = {
            uid
            for uid in set(waiting_by_user) | set(running_by_user)
            if self.user_is_fair_prefill(uid, running_batch=running_batch)
        }

        for uid in fair_users:
            user_timeline = self.event_queue.users.get(uid)
            if user_timeline is None:
                user_timeline = UserTimeline(uid)
                self.event_queue.users[uid] = user_timeline

            live_waiting = {req.rid: req for req in waiting_by_user.get(uid, [])}
            live_running = {req.rid: req for req in running_by_user.get(uid, [])}
            live_rids = set(live_waiting) | set(live_running)

            for rid in list(user_timeline.waiting_request_timelines.keys()):
                if rid not in live_rids:
                    user_timeline.waiting_request_timelines.pop(rid, None)
            for rid in list(user_timeline.active_request_timelines.keys()):
                if rid not in live_rids:
                    user_timeline.active_request_timelines.pop(rid, None)
            for rid, tracked in list(self.event_queue.requests.items()):
                if tracked.req.uid == uid and rid not in live_rids:
                    self.event_queue.requests.pop(rid, None)
                    self.event_queue.most_recent_event_real.pop(rid, None)

            for req in live_waiting.values():
                tracked = self._ensure_tracked_request(req)
                self._set_live_request_state(
                    req,
                    tracked,
                    user_timeline,
                    now,
                    waiting=True,
                )

            for req in live_running.values():
                tracked = self._ensure_tracked_request(req)
                self._set_live_request_state(
                    req,
                    tracked,
                    user_timeline,
                    now,
                    waiting=False,
                )

        return fair_users

    def start_of_pass(
        self,
        running_batch,
        waiting_queue,
        *,
        new_token_ratio: float = 0.0,
        max_running_requests=None,
    ):
        # Keep event queue internals updated without relying on the buggy pre-TODO version.
        self._fairinf_pass_id += 1
        fair_users = self._sync_fair_user_tracking(running_batch, waiting_queue)
        for uid in list(self.event_queue.users.keys()):
            if uid in fair_users:
                continue
            self.event_queue.users.pop(uid, None)
            for rid, tracked_req in list(self.event_queue.requests.items()):
                if tracked_req.req.uid == uid:
                    self.event_queue.requests.pop(rid, None)
                    self.event_queue.most_recent_event_real.pop(rid, None)

        current_time = time.time()
        for uid in fair_users:
            user_timeline = self.event_queue.users.get(uid)
            if user_timeline is not None:
                user_timeline.complete_upto_time(current_time, self.event_queue.most_recent_event_real)
        self._build_deadline_queue(waiting_queue, running_batch, None)
        self._write_fairinf_log(
            "start_of_pass",
            "pass_begin",
            (
                f"fair_users={len(fair_users)},tracked_reqs={len(self.event_queue.requests)},"
                f"fairness_snapshot={self._format_fairness_snapshot(running_batch, waiting_queue)}"
            ),
        )

    def process_new_request (self, req: "Req"):
        if req.rid not in self.event_queue.requests:
            tracked = TrackedRequest(req, None, self._edf_deltas_us.copy())
            now = time.time()
            tracked.alternate_history_timeline.history = [RequestStartEvent(req.rid, 0, now)]
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestPrefillEvent(req.rid, 0, now)
            ]
            self.event_queue.requests[req.rid] = tracked
        self.event_queue.most_recent_event_real[req.rid] = RequestStartEvent(
            req.rid, 0, time.time()
        )

    def _mark_violation_if_executed_after_deadline(
        self,
        req: "Req",
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        tracked = self.event_queue.requests.get(req.rid)
        real_e = self.event_queue.most_recent_event_real.get(req.rid)
        if tracked is None or real_e is None:
            return

        now_ts = time.time() if now is None else now
        try:
            upcoming_events = tracked.earliest_events_after_real_time(real_e) or []
        except Exception:
            return

        matched_event = None
        for event in upcoming_events:
            if event_type == "prefill" and isinstance(event, RequestPrefillEvent):
                matched_event = event
                break
            if event_type == "decode" and isinstance(event, RequestDecodeEvent):
                if completion_number is None or event.completion_number == completion_number:
                    matched_event = event
                    break

        if matched_event is None:
            return

        deadline = matched_event.end_timestamp + self._event_delta_seconds(tracked, matched_event)
        if now_ts > deadline:
            TIMELINE_WRITER.mark_delta_violation(req.rid, req.uid, event_type=event_type)

    def finished_prefill (self, batch: "ScheduleBatch"):
        super().finished_prefill(batch)
        for req in batch.reqs:
            now = time.time()
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="prefill",
                now=now,
            )
            tracked = self.event_queue.requests.get(req.rid)
            if tracked is not None:
                tracked.alternate_history_timeline.history.append(
                    RequestPrefillEvent(req.rid, 0, now)
                )
                tracked.alternate_history_timeline.anticipated_future_events = [
                    RequestDecodeEvent(1, req.rid, 0, now)
                ]
            self.event_queue.most_recent_event_real[req.rid] = RequestPrefillEvent(req.rid, 0, now)

    def finished_decode (self, batch: "ScheduleBatch"):
        super().finished_decode(batch)
        for req in batch.reqs:
            now = time.time()
            completion_number = len(req.output_ids)
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="decode",
                completion_number=completion_number,
                now=now,
            )
            tracked = self.event_queue.requests.get(req.rid)
            if tracked is not None:
                tracked.alternate_history_timeline.history.append(
                    RequestDecodeEvent(completion_number, req.rid, 0, now)
                )
                tracked.alternate_history_timeline.anticipated_future_events = [
                    RequestDecodeEvent(completion_number + 1, req.rid, 0, now)
                ]
            self.event_queue.most_recent_event_real[req.rid] = RequestDecodeEvent(
                completion_number, req.rid, 0, now
            )

    def mark_request_finished (self, req):
        super().mark_request_finished(req)
        self.event_queue.requests.pop(req.rid, None)
        self.event_queue.most_recent_event_real.pop(req.rid, None)
