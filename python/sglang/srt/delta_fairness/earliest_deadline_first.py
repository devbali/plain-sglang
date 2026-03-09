from __future__ import annotations

"""Delta fairness variant with forced-prefill disabled."""

from typing import Dict, List, Optional, Sequence, Tuple

from .delta_fairness_policy import DeltaFairnessPolicy

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder

import time

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

# for isolated decode time estimation, use the underestimate for latency
ISOLATED_DECODE_TIME_ESTIMATION = lambda total_batch_sum, max_token_size, batch_length: min(5e-3,
    1.03882419e-02 + 6.81862494e-08*total_batch_sum + 2.62872519e-07*max_token_size + 5.65921863e-05*batch_length
)

ISOLATED_PREFILL_TIME_ESTIMATION = lambda total_batch_sum, max_token_size, batch_length: min(5e-3,
    -9.50861833e-02 + 6.60140608e-05*total_batch_sum + 8.86754786e-06*max_token_size + -1.97662090e-04*batch_length
)

POOLED_DECODE_TIME_ESTIMATION = lambda total_batch_sum, max_token_size, batch_length: min(5e-3,
    8.20769189e-03 + 3.68620965e-08*total_batch_sum + 2.47297800e-07*max_token_size + 2.39200099e-05*batch_length
)

POOLED_PREFILL_TIME_ESTIMATION = lambda total_batch_sum, max_token_size, batch_length: min(5e-3,
    2.26878890e-02 + 2.58293523e-05*total_batch_sum + 5.72380928e-06*max_token_size + -6.10541804e-05*batch_length
)

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
    "max_running_batch": 56
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
        
        batch = []
        batch_size = 0
        get_prompt_tokens = lambda tracked_req: len(tracked_req.req.origin_input_ids)
        for tracked_req in reqs:
            if tracked_req.most_recent_event.end_timestamp > start_time:
                break # all subsequent requests will also be after start_time, since they are sorted by time of entry
            
            assert get_prompt_tokens(tracked_req) <= USER_CONFIG["max_prefill_tokens"], "Request has more prompt tokens than the user max prefill tokens, cannot schedule"
            
            if batch_size + get_prompt_tokens(tracked_req) > USER_CONFIG["max_prefill_tokens"]:
                continue
            
            batch.append(tracked_req)
            batch_size += get_prompt_tokens(tracked_req)
        
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
            if len(self.waiting_request_timelines) > 0:
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
        self.event_queue.finished_decode(batch)

    def finished_prefill (self, batch: "ScheduleBatch"):
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

        RequestPrefillEvent.is_logically_after = _prefill_after
        RequestDecodeEvent.is_logically_after = _decode_after
        RequestTimeline.prune_till_after_event = _prune_after
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
    ):
        self._read_deltas(delta_fairness_deltas_microseconds)
        waiting_by_rid = {r.rid: r for r in waiting_queue}
        running_by_rid = {r.rid: r for r in (running_batch.reqs if running_batch else [])}

        queue = []
        waiting_prefill_deadline_by_rid: Dict[str, float] = {}

        for rid, tracked_req in self.event_queue.requests.items():
            real_e = self.event_queue.most_recent_event_real.get(rid)
            if real_e is None:
                continue

            if not isinstance(tracked_req.alternate_history_timeline.history, list):
                tracked_req.alternate_history_timeline.history = [
                    tracked_req.alternate_history_timeline.history
                ]

            try:
                upcoming_events = tracked_req.earliest_events_after_real_time(real_e) or []
            except Exception:
                # Keep EDF robust even if alternate timeline logic throws.
                continue

            req = tracked_req.req
            for e in upcoming_events:
                if isinstance(e, RequestPrefillEvent):
                    if rid not in waiting_by_rid:
                        continue
                    if not self.req_is_fair_prefill(req, running_batch=running_batch):
                        continue
                    deadline = e.end_timestamp + self._event_delta_seconds(tracked_req, e)
                    queue.append((deadline, "prefill", req))
                    prev = waiting_prefill_deadline_by_rid.get(rid)
                    if prev is None or deadline < prev:
                        waiting_prefill_deadline_by_rid[rid] = deadline
                elif isinstance(e, RequestDecodeEvent):
                    if rid not in running_by_rid:
                        continue
                    if not self.req_is_fair_decode(req, running_batch=running_batch):
                        continue
                    deadline = e.end_timestamp + self._event_delta_seconds(tracked_req, e)
                    queue.append((deadline, "decode", req))

        queue.sort(key=lambda x: (x[0], 0 if x[1] == "decode" else 1))
        self._edf_deadline_queue = queue
        self._edf_earliest = queue[0] if queue else None
        self._edf_waiting_prefill_deadline_by_rid = waiting_prefill_deadline_by_rid
        if log_calc:
            self._write_fairinf_log(
                "deadline_calc",
                "built_deadlines",
                f"waiting={len(waiting_queue)},running={len(running_by_rid)},candidates={len(queue)}",
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
        return [x[1] for x in indexed]

    def start_of_pass(self, running_batch, waiting_queue):
        # Keep event queue internals updated without relying on the buggy pre-TODO version.
        self._fairinf_pass_id += 1
        reqs = list(waiting_queue)
        if running_batch is not None:
            reqs.extend(running_batch.reqs)
        fair_users = {
            r.uid for r in reqs if self.user_is_fair_prefill(r.uid, running_batch=running_batch)
        }
        self.event_queue.start_of_pass(fair_users)
        self._build_deadline_queue(waiting_queue, running_batch, None)
        self._write_fairinf_log(
            "start_of_pass",
            "pass_begin",
            f"fair_users={len(fair_users)},tracked_reqs={len(self.event_queue.requests)}",
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

    def finished_prefill (self, batch: "ScheduleBatch"):
        for req in batch.reqs:
            now = time.time()
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
        for req in batch.reqs:
            now = time.time()
            completion_number = len(req.output_ids)
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
        self.event_queue.requests.pop(req.rid, None)
        self.event_queue.most_recent_event_real.pop(req.rid, None)
