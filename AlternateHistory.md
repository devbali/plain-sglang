## The Alternate History Simulator

### `RequestEvent` and subclasses

`RequestEvent` is the base event type stored in request timelines. Three concrete types:

- **`RequestStartEvent`** — marks when a request arrived. Anchored to `arrival_timestamp`.
- **`RequestPrefillEvent`** — marks when a request's prefill would complete in isolation. Carries `end_timestamp` (isolated clock) and `duration`.
- **`RequestDecodeEvent`** — marks when decode token `completion_number` would complete in isolation. Carries `end_timestamp`, `duration`, and `completion_number`.

Each event has `is_logically_after(other)` which defines a total order: Start < Prefill < Decode(1) < Decode(2) < ...

### `RequestTimeline`

Stores the isolated history for a single request.

```
history:                   [StartEvent, PrefillEvent, DecodeEvent(1), DecodeEvent(2), ...]
anticipated_future_events: [DecodeEvent(next)]  # at most one anticipated event
```

- **`history`** — events that have been "committed", resolved. This goes upto the last time this timeline was "resolved" and should have events scheduled till that point in time
- **`next_anticipated_event`** — This is the next event anticipated by the request's isolated simulator, but it is not the next after the history, it is the next after the logical status of the request in the real world. The timing is approximate for decode events. take the last decode in history, and add the difference in number of decodes multiplied by the amount of time a decode batch would take if scheduled now, using the isolated decode time estimator.

**`events_after(real_event)`** — returns events in `history + anticipated_future_events` that come after the given real event. Used to find upcoming deadline candidates.

### `RequestStatusReal`
This is a dataclass that tracks what the request has done in real life. It has request id, whether prefill is done, and how many decodes have been completed, and whether the request itself is complete.
This is used by the real scheduler to mark status, in a dictionary.

### `UserTimeline`

Per-user isolated scheduler. Owns all `TrackedRequest`s for one user and runs an isolated scheduling simulation over them.

**Fields:**
- `uid` — user ID
- `max_kv_tokens` — KV memory budget per user in isolation (maps to GPU pool size)
- `fairinf_n` — number of concurrent users (used for time estimation)
- `history` — committed `UserEvent` list (prefill/decode events whose isolated time is in the past)
- `anticipated_next_event` — predicted future user events (should be ahead of real progress)
- `request_timelines` — dict `rid → TrackedRequest` for live requests
- `requests_real` -- dict `rid ->` RequestStatusReal` for live request statuses in real
- `finished_request_timelines` — dict `rid → TrackedRequest` for completed requests
- `finishing_requests` — list of `_FinishingRequest` ghosts
- `min_new_token_ratio` — decode reservation ratio (affects KV memory accounting)



**Queued requests**: If the isolated scheduler cannot schedule a waiting request's prefill within the simulation loop (e.g., blocked by the KV budget of active decodes), the anticipated prefill event should use `end_timestamp = float("inf")`. This signals that the request is queued but has no known prefill slot yet. It will still appear as a prefill candidate in `build_deadline_candidates`, but its start_deadline will be `inf`, placing it last in deadline ordering.

**`_advance_scheduler_step(waiting_states, active_states, future_history)`** — one step of the isolated scheduler:
- If there are ready-to-prefill requests, run `_build_prefill_batch` and simulate the prefill.
- Otherwise if there are active/finishing requests, simulate one decode round (possibly retracted if memory is tight).
- If a request is complete (you can see from the requests_real dictionary), in the requests dictionary, and you have scheduled the last decode, you can stop tracking it and remove it from the requests dictionary**`rebuild_from_real_state(optional timestamp float)`** — the core simulation method. Runs the isolated scheduler forward from the committed real state to produce `anticipated_future_events` for each live request.
Before you do this, you can split the dictionary into requests you do not have in the running batch already (the ones you need to prefill maybe) and the ones you have in the running batch of the isolated scheduler
Steps:
1. Get request statuses, requests_real is a dictionary that maps req_id and RequestStatusReal, these are the real statuses
2. Run the scheduler loop: repeatedly call `_advance_scheduler_step` until we reach the current time or the optional timestamp float if given
3. Make sure every request we care about has an anticipated deadline for a logically future event from the current status in the real world, as given real state will show
This means that the `anticipated_next_event` should be set here. If the request in real has not done a prefill, this should be a prefill. If it has done a prefill, it should be the first decode. If it has done decodes, it should be the next decode. If it is complete, it should be empty.
- If a new arrival is coming before the next step completes, stop at the arrival point.

**`_build_prefill_batch(waiting_states, active_states, future_history)`** — builds an isolated prefill batch respecting `max_kv_tokens`. Requests that exceed the KV budget are excluded.

**`_isolated_retract_decode(waiting_states, active_states, extra_decode_tokens)`** — if the KV budget is exceeded (e.g. too many active requests), evict the longest-running request back to waiting, adding a `RETRACTION_PENALTY_SECONDS` to that step.


### `DeadlineCandidate`

The output of the deadline computation. One candidate per upcoming event per live request.

**Fields:** `deadline`, `start_deadline`, `event_type` ("prefill" or "decode"), `req`, `event`

- `deadline` = `event.end_timestamp + delta_seconds` (isolated deadline + fairness delta tolerance)
- `start_deadline` = `deadline - pooled_prefill/decode_estimate` (when we must start the GPU operation)

### `AlternateHistorySimulator`

The top-level simulator that owns all `UserTimeline`s.

**Fields:**
- `users: Dict[str, UserTimeline]`
- `requests: Dict[str, TrackedRequest]`
- `most_recent_event_real: Dict[str, RequestEvent]` — the "ground truth" of where each request is in isolated time. Updated by `process_new_request`, `finished_prefill`, and `_apply_logical_decode_updates`.

**`get_live_users(running_batch, waiting_queue, deltas_in_microseconds)`** — synchronizes the simulator with the current real batch state. Creates `TrackedRequest`s for new requests, updates existing ones, and removes requests that are no longer live.

**`process_new_request(req, deltas_in_microseconds, arrival_timestamp)`** — called when a new request enters the waiting queue. Records `RequestStartEvent` in `most_recent_event_real` with the provided arrival timestamp (captured on the main thread, not inside the worker).

**`finished_prefill(batch)`** — called when the GPU completes a prefill pass. Updates the request status in the relevant user's requests_real so that the next rebuild_from_real_state of that user anticipates a decode now.

**`build_deadline_candidates(waiting_queue, running_batch, ...)`** — scans all tracked requests and builds `DeadlineCandidate` objects from their upcoming events. Returns the candidate list, a dict of `waiting_prefill_deadline_by_rid`, and optionally an `ordered_waiting_queue` (the waiting requests sorted by their prefill start_deadline).
