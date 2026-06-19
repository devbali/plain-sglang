# Scheduling Policy Hook API Specification

Extracted from FairInference paper §5.1 (Algorithm 2: τ-Token Fair Scheduler)
and cross-referenced with the actual hook call sites in
`python/sglang/srt/managers/scheduler.py`.

Every scheduling hook below has **three policy rows**:
- **NoOp** (the stub — pass-through, no fairness)
- **Static Partition** (hard per-user memory partitions)
- **τ-Fair** (deadline-driven fairness with reservations, the paper's contribution)

---

## Hook 1: `on_new_request(req) → None`

| | |
|---|---|
| **Call site** | `scheduler.py:2228` — `_add_request_to_queue()` |
| **Algorithm 2 step** | 2a — "Update client resource usages" |
| **Returns** | Nothing (None) |

### What it sees
- `req.uid: str` — client identifier (may be missing/empty for untagged requests)
- `req.rid: str` — request identifier
- `req.arrival_time` — wall-clock arrival timestamp
- `req.sampling_params.max_new_tokens` — planned decode length
- `len(req.origin_input_ids)` — prompt token count
- `req.deadline` — per-token deadline (if already computed; usually set elsewhere)

### NoOp behavior
```
Nothing. Fully pass-through.
```

### Static Partition behavior
```
Record the request arrival time per user. Optionally, if a user's current
usage already exceeds their static partition (max_per_user), the request
can be pre-emptively marked — but admission is really decided later in
can_admit_request(), not here.
```

### τ-Fair behavior

**What this hook must do:**

1. **Record arrival time** for the user:
   ```
   self._user_arrival_time[uid] = now()
   ```

2. **If uid is new,** initialize per-user EDF state:
   ```
   if uid not in self._user_edf_state:
       self._user_edf_state[uid] = {
           "prefill_count": 0,
           "decode_steps": 0,
           "retracted_count": 0,
       }
   ```

3. **Compute initial deadline for the request** (delegated to the simulator /
   performance model, see `deadline_model.md`):
   ```
   req.deadline = self._compute_deadline(req, is_first=True)  # will be refined
   ```

4. **Track total pending tokens per user** for admission decisions:
   ```
   self._user_pending_tokens[uid] = (
       self._user_pending_tokens.get(uid, 0)
       + len(req.origin_input_ids)
       + req.sampling_params.max_new_tokens
   )
   ```

**Thread safety:** Single-threaded (called from scheduler main loop only).

**Key invariant:** After this hook returns, the scheduler expects that any policy-level
bookkeeping for this request has been initialized.

---

## Hook 2: `on_prefill_vs_decode_decision(waiting_queue, running_batch, new_prefill_batch) → Optional[str]`

| | |
|---|---|
| **Call site** | `scheduler.py:2558` — `get_next_batch_to_run()` |
| **Algorithm 2 step** | 2d.i — "If decode deadlines are violated from admitting r, skip r" |
| **Returns** | `None` → default (prefill if available, else decode) |
|   | `"prefill"` → force prefill |
|   | `"decode"` → force decode |

### What it sees
- `waiting_queue: List[Req]` — pending prefill requests (pre-sorted by upstream `calc_priority`)
- `running_batch: ScheduleBatch` — currently running decode batch
  - `running_batch.reqs` — list of running decode requests with `.uid`, `.deadline`
  - `running_batch.batch_is_full` — whether at max capacity
- `new_prefill_batch: Optional[ScheduleBatch]` — prefill batch the scheduler has formed, or `None`

### NoOp behavior
```
return None   # prefill-first always
```

### Static Partition behavior
```
# (From StaticPartitionSchedulingPolicy.on_prefill_vs_decode_decision)
If max_per_user > 0 (initialized):
    For each req in running_batch.reqs:
        uid = req.uid
        if uid and cache_policy.get_user_total_tokens(uid) > max_per_user:
            return "decode"   # force decode to free over-quota user's space
return None
```

### τ-Fair behavior

**What this hook must do:**

This is the most critical τ-fair hook. It implements the decode-deadline-
protection check from Algorithm 2, step 2d.i.

**Step-by-step logic:**

```
1. If running_batch is empty:
       return None  # nothing to protect, let default prefill run

2. If new_prefill_batch is None:
       return None  # no prefill work to consider; default → decode

3. QUANTUM CHECK: For each running decode request:
   a. Get its next decode deadline d_k
   b. Estimate when the next decode step will complete:
         now + Δdecode_MT(running_batch_size)
   c. If now + Δdecode_MT > d_k for any well-behaved client:
         return "decode"   # deadlines at risk — skip prefill

4. HEADROOM CHECK: Compute headroom for this pass:
       headroom = self._headroom_remaining()   # from simulator / performance model
       prefill_cost = Δprefill(new_prefill_batch_tokens, ...)
   If prefill_cost > headroom:
       return "decode"   # prefill would eat into decode deadlines

5. FAIR-SHARE CHECK: Check if any running well-behaved client is being starved:
       for each running req with usage ≤ fair_share:
           if req has been in running_batch for > τ without free decode:
               return "decode"

6. Otherwise:
       return None  # safe to run prefill (or return "prefill" to force)
```

**How to estimate Δdecode_MT:** Use the performance model (`performance_model.md`):
```
Δdecode_MT = cd + αd * S(running_batch) + βd * M(running_batch) + γd * len(running_batch.reqs)
```

**What determines whether a client is "well-behaved":**
Check the cache policy: `cache_policy.get_user_total_tokens(uid) ≤ fair_share`.

**Decision priority (from paper):**
1. Overdue decode deadlines → always return `"decode"`
2. Prefill would consume > available headroom → `"decode"`
3. Well-behaved clients being starved → `"decode"`
4. Otherwise → `None` (allow prefill-first)

**Thread safety:** Single-threaded (scheduler main loop).

**Key invariant:** This hook must NOT return `"prefill"` when any running decode's
deadline would be violated by the admission. This is the primary safety guarantee
of τ-fairness.

---

## Hook 3: `on_schedule_prefill(waiting_queue, running_batch, prefill_adder) → Optional[List[Req]]`

| | |
|---|---|
| **Call site** | `scheduler.py:2656` — `_get_new_batch_prefill_raw()` |
| **Algorithm 2 step** | 2d.ii–2d.iv — "Earliest deadline first; fair clients first; over-quota only with space" |
| **Returns** | `None` → use default queue (scheduler's `calc_priority` result) |
|   | `List[Req]` → override queue with this filtered/reordered list |

### What it sees
- `waiting_queue: List[Req]` — prefill requests, pre-sorted by scheduler's priority policy
- `running_batch: ScheduleBatch` — active decode batch
- `prefill_adder: PrefillAdder` — resource manager with:
  - `prefill_adder.rem_total()` — remaining token budget
  - `prefill_adder.rem_req()` — remaining request slots
  - `prefill_adder.can_run_prefill` — whether prefill is feasible
  - `prefill_adder.preempt_to_schedule(req, ...)` — preempt running requests to make room

### NoOp behavior
```
return None   # use default queue from calc_priority()
```

### Static Partition behavior
```
# (From StaticPartitionSchedulingPolicy.on_schedule_prefill)
If max_per_user > 0 (initialized):
    Sort waiting_queue by: (available_quota = max_per_user - user_total_tokens)
    → clients with most available quota first
    → over-quota clients go to the back
return sorted_queue
```

### τ-Fair behavior

**What this hook must do:**

This is the main fairness-aware queue ordering hook. Algorithm 2, steps 2d.ii–2d.iv.

**Step-by-step logic:**

```
1. If cache_policy is not initialized yet (max_per_user == 0):
       return None  # defer to default

2. COMPUTE EDF + FAIR-SHARE KEY for each request in waiting_queue:

   def ordering_key(req):
       uid = req.uid or ""
       deadline = req.deadline or float('inf')

       # Primary sort: deadline tier
       if deadline <= now():                return (0, ...)  # overdue
       if deadline <= now() + τ:            return (1, ...)  # at-risk
       else:                                 return (2, ...)  # safe

       # Secondary sort: fair-share status
       total = cache_policy.get_user_total_tokens(uid)
       fair_share = cache_policy.fair_share_per_user  # M / |C|
       shortfall = max(0, fair_share - total)  # positive → under quota
       excess = max(0, total - fair_share)      # positive → over quota

       # Under-quota users prioritized; among over-quota, smaller excess first
       return (deadline_tier, -shortfall, excess)

3. REORDER: Apply EDF + fair-share key ordering:
       sorted_queue = sorted(waiting_queue, key=ordering_key)

4. FILTER (optional, aggressive τ-fair): Remove requests where admission
   would violate decode deadlines:
       filtered = []
       for req in sorted_queue:
           if self._would_violate_decode_deadline(req, running_batch, prefill_adder):
               continue  # skip this request, retry later
           filtered.append(req)

5. Return filtered (or sorted_queue if filtering is handled by can_admit_request below)

return sorted_queue
```

**The key ordering (from paper §5.1):**
1. Earliest deadline first (EDF) — requests closest to missing SLOs go first
2. Among same-deadline-tier: well-behaved clients (usage ≤ fair share) first
3. Among over-quota clients: those with smallest excess first

**Filtering note:** The τ-fair paper filters in step 2d.i (before admission), but
in our hook architecture, filtering out requests that would violate decode deadlines
can happen either here (return a filtered list) or in `can_admit_request()` (per-request
check in the add loop). The spec recommends: reorder here, filter in `can_admit_request`.

**Thread safety:** Single-threaded (scheduler main loop).

**Key invariant:** The returned queue must be at least as restrictive as the default
queue — it can reorder and filter, but cannot add requests that weren't already
in the waiting_queue.

---

## Hook 4: `on_prefill_decision(batch) → None`

| | |
|---|---|
| **Call site** | `scheduler.py:2577` — `get_next_batch_to_run()`, just before GPU dispatch |
| **Algorithm 2 step** | 2e — "Execute prefill batch" |
| **Returns** | Nothing (None) |

### What it sees
- `batch: ScheduleBatch` — the selected prefill batch
  - `batch.reqs` — list of requests admitted for prefill

### NoOp behavior
```
Nothing. Fully pass-through.
```

### Static Partition behavior
```
# Record which requests were admitted
for req in batch.reqs:
    # No specific action needed; tracking is handled by cache hooks (on_insert, etc.)
    pass
```

### τ-Fair behavior

**What this hook must do:**

1. **Record admitted prefills for headroom accounting:**
   ```
   for req in batch.reqs:
       uid = req.uid
       # Record that this request consumed prefill resources
       self._user_prefill_count[uid] = self._user_prefill_count.get(uid, 0) + 1
       # Update last prefill time
       self._user_last_prefill_time[uid] = now()
   ```

2. **Charge the headroom budget** with the estimated prefill cost:
   ```
   prefill_tokens = sum(len(r.origin_input_ids) for r in batch.reqs)
   prefill_cost = Δprefill_model(prefill_tokens, batch_size=len(batch.reqs))
   self._headroom_consumed += prefill_cost
   ```

3. **Log for observability:** which users were admitted, their token counts,
   and whether they were under or over quota at admission time.

**Thread safety:** Single-threaded (scheduler main loop).

**Key invariant:** After this hook returns, the prefill batch is immutable and
will be dispatched to GPU. The hook must NOT modify `batch.reqs` at this point.

---

## Hook 5: `on_decode_decision(batch) → None`

| | |
|---|---|
| **Call site** | `scheduler.py:2587` — `get_next_batch_to_run()`, just before GPU dispatch |
| **Algorithm 2 step** | 2f.ii — "Execute decode for all requests in Rdecode" |
| **Returns** | Nothing (None) |

### What it sees
- `batch: ScheduleBatch` — the running decode batch
  - `batch.reqs` — list of active decode requests with `.uid`, `.deadline`

### NoOp behavior
```
Nothing. Fully pass-through.
```

### Static Partition behavior
```
# Record which users are decoding, but no specific action needed.
pass
```

### τ-Fair behavior

**What this hook must do:**

1. **Advance per-user decode step counters:**
   ```
   for req in batch.reqs:
       uid = req.uid
       self._user_decode_steps[uid] = self._user_decode_steps.get(uid, 0) + 1
   ```

2. **Check for deadline misses** (observability, not enforcement):
   ```
   now_time = now()
   for req in batch.reqs:
       if req.deadline is not None and now_time > req.deadline:
           self._deadline_miss_count[uid] = self._deadline_miss_count.get(uid, 0) + 1
           logger.warning(f"Deadline miss: uid={uid}, rid={req.rid}, "
                           f"deadline={req.deadline}, now={now_time}")
   ```

3. **Release headroom budget** (decode completes faster in MT than ISO):
   ```
   headroom_gained = Δdecode_ISO - Δdecode_MT  # from performance model
   self._headroom_consumed = max(0, self._headroom_consumed - headroom_gained)
   ```

4. **Update per-user decode time tracking** for future deadline refinements:
   ```
   for req in batch.reqs:
       uid = req.uid
       self._user_last_decode_time[uid] = now()
   ```

**Thread safety:** Single-threaded (scheduler main loop).

**Key invariant:** This hook observes the decode step after the scheduler has
committed to running it. It cannot change what runs; only record what happened.

---

## Hook 6: `on_end_of_scheduler_pass(batch) → None`

| | |
|---|---|
| **Call site** | `scheduler.py:2603` — `get_next_batch_to_run()`, final step |
| **Algorithm 2 step** | End of while-loop iteration; implicit |
| **Returns** | Nothing (None) |

### What it sees
- `batch: Optional[ScheduleBatch]` — the batch that will run, or `None` if idle

### NoOp behavior
```
Nothing. Fully pass-through.
```

### Static Partition behavior
```
# No specific end-of-pass action.
pass
```

### τ-Fair behavior

**What this hook must do:**

1. **Housekeeping of the headroom budget:**
   ```
   if batch is None:
       # Idle pass — accumulate headroom
       self._headroom_consumed = max(0, self._headroom_consumed - Δdecode_ISO)
   ```

2. **Prune completed requests from EDF tracking:**
   ```
   completed_rids = {r.rid for r in batch.reqs if r.finished()} if batch else set()
   for rid in completed_rids:
       self._edf_state.pop(rid, None)
   ```

3. **Export metrics** (if metrics collector is attached):
   ```
   self._export_pass_metrics(batch)  # per-pass headroom, deadline misses, etc.
   ```

4. **Log the pass outcome** for debugging and policy observability.

5. **Reconcile simulator state** with real scheduler state (synchronize the
   isolated-execution simulator with the current time and real request state).

**Thread safety:** Single-threaded (scheduler main loop).

**Key invariant:** This is the hook for any end-of-iteration bookkeeping. It must
be non-blocking and fast — it runs on the critical scheduling path.

---

## Summary: Hook Call Order in One Scheduler Pass

```
get_next_batch_to_run():
   1. on_new_request(req)         ← called earlier, when requests arrive
   2. Form candidate prefill batch
   3. on_prefill_vs_decode_decision()  → returns "prefill" / "decode" / None
   4a. IF prefill:
         on_schedule_prefill()    → returns reordered/filtered queue
         ... add requests to batch ...
         on_prefill_decision()    → records admitted requests
   4b. IF decode:
         on_decode_decision()     → records decode step
   5. on_end_of_scheduler_pass()  → housekeeping
```

## Summary: Policy Behavior Matrix

| Hook | NoOp | Static Partition | τ-Fair |
|------|------|-----------------|--------|
| `on_new_request` | — | Record arrival | Compute initial deadline, init EDF state |
| `on_prefill_vs_decode_decision` | `None` | `"decode"` if any over-quota user running | `"decode"` if decode deadlines at risk; else `None` |
| `on_schedule_prefill` | `None` | Sort by (quota - usage) desc | Sort by (deadline tier, fair-share shortfall, excess) |
| `on_prefill_decision` | — | — | Charge headroom budget, record admitted users |
| `on_decode_decision` | — | — | Advance decode counters, check for deadline misses, release headroom |
| `on_end_of_scheduler_pass` | — | — | Prune completed state, reconcile simulator, export metrics |
