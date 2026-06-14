# τ-Token Fair Scheduler — Full Specification

Extracted from FairInference paper §5.1 (Algorithm 2) and §5.3 (deadline/tolerance model).

---

## 1. Algorithm (Algorithm 2 in paper)

```
Require:
    Pending prefill priority queue Qprefill       [sorted by deadline, EDF]
    Running decode set Rdecode                    [sorted by deadline, EDF]
    τ-token fair KV cache                          [provides reservations ρ_i, usages u_i]
    Fair shares f_i = M / |C|
    Per-token deadlines d(r)_k for all requests r, tokens k
    Memory budget M, current KV usage C

While Qprefill ≠ ∅ or Rdecode ≠ ∅:
    a. Update client resource usages             → on_new_request, query cache hooks
    b. Sort Qprefill and Rdecode by deadlines    → on_schedule_prefill (EDF)
    c. Bprefill = ∅
    d. For each r in Qprefill (EDF order):
        i.   If admitting r violates ANY running decode deadline → skip r
        ii.  Else if client is fair (u_i ≤ f_i) → admit r into Bprefill
        iii. Else if client NOT fair but C − M > 0 (space available) → admit r
        iv.  Else → leave r in Qprefill, retry later
    e. If Bprefill ≠ ∅:
        i.   Execute prefill batch
        ii.  Move admitted requests to Rdecode
    f. Else if Rdecode ≠ ∅:
        i.   Preempted requests removed from Rdecode → re-enqueued to Qprefill
        ii.  Execute decode for all in Rdecode
        iii. Remove completed requests from Rdecode
```

---

## 2. Hook-to-Algorithm Mapping

| Algorithm Step | Hook | What It Must Do |
|----------------|------|-----------------|
| 2a — Update usages | `on_new_request(req)` | Initialize EDF state, compute initial deadline, track pending tokens per user |
| 2b, 2d.ii-iv — EDF ordering + fair admission | `on_schedule_prefill(wq, rb, adder)` → `List[Req]` | Return waiting_queue sorted by (deadline tier, fair-share shortfall, excess) |
| 2d.i — Decode deadline gate | `on_prefill_vs_decode_decision(wq, rb, b)` → `Optional[str]` | Return `"decode"` if prefill would blow through decode deadlines or headroom |
| 2e.i — Prefill dispatched | `on_prefill_decision(batch)` | Charge headroom budget, record admitted users |
| 2f.ii — Decode dispatched | `on_decode_decision(batch)` | Advance step counters, check deadline misses, release headroom |
| Implicit — End of pass | `on_end_of_scheduler_pass(batch)` | Prune completed, reconcile simulator, export metrics |

---

## 3. Headroom Model — The Core τ-Fair Concept

### 3.1 Definition

From the paper (§5.3.1, §4.1):

```
headroom = Δdecode_ISO − Δdecode_MT

Where:
  Δdecode_ISO = time between decode batches in isolated execution (perf model)
  Δdecode_MT  = time between decode batches in multi-tenant execution (measured)
```

**If headroom > 0:** decode steps complete FASTER in multi-tenant than in isolation.
This creates slack time that the scheduler can use to run prefills without
violating decode deadlines.

**If headroom ≤ 0:** no slack — decode is already at or past isolated speed.
Cannot admit any new prefills without violating deadlines.

### 3.2 How to compute headroom

```python
def compute_headroom(self, running_batch, scheduler):
    # Isolated decode time (from perf model)
    S = sum(len(r.origin_input_ids) + r.sampling_params.max_new_tokens
            for r in running_batch.reqs)
    M_token = max(len(r.origin_input_ids) for r in running_batch.reqs)
    N = len(running_batch.reqs)

    # Scaled to |C| clients (each client gets 1/|C| of resources in iso)
    # In iso, the batch would have only this client's requests
    # So we compute: what would decode time be with only 1/|C| of the batch?
    delta_decode_iso = cd + αd*(S/|C|) + βd*(M_token/|C|) + γd*(N/|C|)

    # Multi-tenant decode time (measured per pass)
    delta_decode_mt = self._last_observed_decode_time  # tracked in on_decode_decision

    return max(0.0, delta_decode_iso - delta_decode_mt)
```

### 3.3 Headroom Accumulation

Headroom accumulates over consecutive decode steps:

```
H_accumulated += headroom_per_step × consecutive_decode_steps

When a prefill is admitted:
  prefill_cost = Δprefill(new_batch)
  if prefill_cost ≤ H_accumulated:
      H_accumulated -= prefill_cost
      admit prefill  # safe — deadlines still protected
  else:
      skip prefill   # would violate deadlines
```

**This is the paper's key insight (Figure 4):** if decodes run faster in
multi-tenant than in isolation, you can safely "spend" the saved time on
prefills without risking token deadlines.

---

## 4. Decode Deadline Protection — Step-by-Step

This is Algorithm 2, step 2d.i, implemented in `on_prefill_vs_decode_decision`.

### 4.1 When to Force Decode

```python
def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
    # 1. Nothing to protect
    if running_batch.is_empty():
        return None
    if new_prefill_batch is None:
        return None

    # 2. Check for imminent decode deadlines
    now_time = time.time()
    for req in running_batch.reqs:
        if req.deadline is None:
            continue
        # Deadline within one decode step
        if req.deadline <= now_time + self.estimated_decode_step_time:
            return "decode"

    # 3. Check headroom — can we afford this prefill?
    prefill_cost = self._estimate_prefill_cost(new_prefill_batch)
    if prefill_cost > self._accumulated_headroom():
        # Prefill would eat into protected decode time
        return "decode"

    # 4. Check well-behaved client starvation
    for req in running_batch.reqs:
        uid = req.uid
        if uid and self._is_well_behaved(uid):
            # Has this client been waiting > τ for progress?
            last_decode = self._user_last_decode_time.get(uid, 0)
            if now_time - last_decode > self.tau_seconds:
                return "decode"  # let the well-behaved client make progress

    # Safe to prefill
    return None
```

### 4.2 Three Levels of Urgency

| Level | Condition | Action |
|-------|-----------|--------|
| CRITICAL | Any decode deadline is within Δdecode_MT | `"decode"` — immediate |
| WARNING | Prefill cost > accumulated headroom | `"decode"` — skip prefill this pass |
| STARVATION | Well-behaved client hasn't decoded in > τ | `"decode"` — let them progress |
| SAFE | None of the above | `None` — prefill-first is fine |

---

## 5. EDF + Fair-Share Queue Ordering

This is Algorithm 2, steps 2b + 2d.ii-iv, implemented in `on_schedule_prefill`.

### 5.1 Ordering Key

```python
def compute_edf_fairshare_key(req, cache_policy, tau):
    uid = req.uid or ""
    deadline = req.deadline or float('inf')
    now = time.time()

    # --- Deadline tier (primary sort) ---
    if deadline <= now:
        tier = 0          # OVERDUE — highest priority
    elif deadline <= now + tau:
        tier = 1          # WITHIN τ — elevated priority
    else:
        tier = 2          # SAFE — normal priority

    # --- Fair-share status (secondary sort) ---
    total = cache_policy.get_user_total_tokens(uid)
    fair_share = cache_policy.fair_share_per_user

    if total <= fair_share:
        shortfall = fair_share - total   # positive: under quota → prioritize
        excess = 0
    else:
        shortfall = 0
        excess = total - fair_share       # positive: over quota → deprioritize

    # Lexicographic: (deadline_tier, -shortfall, excess)
    #   → lower tier first (overdue before safe)
    #   → larger shortfall first (more under-quota first)
    #   → smaller excess first (less over-quota first)
    return (tier, -shortfall, excess)
```

### 5.2 Rationale

From the paper §5.1:
- "The scheduler prioritizes clients closest to their deadlines and those
  using below their fair share of GPU resources"
- "The 'earliest-deadline-first' policy is known to be optimal for meeting
  deadlines in real-time systems"
- "Prioritizing well-behaved clients ensures they never get starved by
  high-demand clients"

---

## 6. Fair-Share-Aware Admission — Per-Request Decision

This is the green-highlighted parts of Algorithm 2, step 2d.

### 6.1 Per-Request Logic (in the add loop, guarded by `can_admit_request`)

```
For each req r in Qprefill (EDF order):
    uid = r.uid; n = needed_tokens(r)

    // Step 2d.i: DECODE DEADLINE CHECK
    If admitting r would cause any running decode to miss its deadline:
        skip r → stays in Qprefill

    // Step 2d.ii: FAIR CLIENT ADMISSION (τ-fair eviction protects)
    Elif u_i ≤ f_i (well-behaved):
        ADMIT r   # cache_policy.can_admit_request returns True

    // Step 2d.iii: OVER-QUOTA WITH SLACK
    Elif u_i > f_i BUT slack = M − Σ u_j > n:
        ADMIT r   # there's genuinely unused capacity

    // Step 2d.iv: OVER-QUOTA, NO SLACK
    Else:
        skip r → stays in Qprefill, retry later
```

### 6.2 Interaction with Cache Policy

The admission check is a **collaboration** between:

| Check | Done By | Hook |
|-------|---------|------|
| Memory budget | Scheduler (`_check_can_add_request`) | PrefillAdder |
| Per-user quota | Cache policy (`can_admit_request`) | Cache hooks |
| Decode deadline | Scheduling policy (`on_prefill_vs_decode_decision`) | Scheduling hooks |
| Queue ordering | Scheduling policy (`on_schedule_prefill`) | Scheduling hooks |

The scheduler calls `can_admit_request()` in the add loop at line 2773. This
is a per-request admission gate. The scheduling hooks set up the ordering
and the prefill-vs-decode decision; the cache hook enforces the per-user limit.

---

## 7. Simulator Integration — Computing Per-Token Deadlines

### 7.1 What the Simulator Does

From the paper §5.5:
- Runs on a separate worker thread
- Receives: request arrival timestamps, completion tokens per request, finished status
- Synchronized to current time at each scheduler pass
- Models each client with 1/N of system GPU resources
- Produces `T_ISO_k` for each token k (when it would complete in isolation)
- Deadlines: `d_k = T_ISO_k + τ`

### 7.2 Simulator Inputs (from scheduling hooks)

| Input | Source | Hook |
|-------|--------|------|
| Request arrival | `Req` enqueued | `on_new_request` |
| Completion token count | `req.sampling_params.max_new_tokens` | Read in `on_new_request` |
| Finished status | `req.finished()` | Checked in `on_end_of_scheduler_pass` |
| Current time | `time.time()` | Poll at each pass |

### 7.3 Simulator Output (to scheduling hooks)

| Output | Consumed By |
|--------|-------------|
| `T_ISO_0` (first token time) | `on_new_request` → sets `req.deadline` |
| `T_ISO_k` (subsequent tokens) | `on_decode_decision` → updates `req.deadline` |
| Overdue events | `on_prefill_vs_decode_decision` → force decode |
| Expected future decodes | `on_schedule_prefill` → EDF ordering key |

### 7.4 Simplified Simulator (without separate thread)

A practical simplification uses the performance model directly:

```python
def _compute_isolated_first_token_time(self, req):
    """Estimate T_ISO_0 for this request if running alone."""
    prompt_tokens = len(req.origin_input_ids)
    # Model: how long prefill takes for 1 request in isolation
    # With 1/N of total GPU compute (modeled by scaling factors up)
    delta_prefill_iso = cp + αp*prompt_tokens*|C| + βp*prompt_tokens*|C| + γp*|C|
    return time.time() + delta_prefill_iso

def _compute_isolated_decode_time(self, batch):
    """Estimate time for one decode step in isolation."""
    S = sum(r.completion_tokens_generated for r in batch.reqs)
    N = len(batch.reqs)
    # Scale by |C| for isolated (1 client runs on 1/|C| resources → |C|× slower)
    return cd + αd*S*|C| + βd*S*|C| + γd*N*|C|
```

---

## 8. Preemption with Deadline Preservation

Algorithm 2, step 2f.i: "Preempted requests removed from Rdecode → Qprefill"

When a decode request must be retracted to make room for prefill:

1. **Compute retraction priority** via `cache_policy.get_retract_priority()`
2. **Retract highest-priority requests** (over-quota first)
3. **Preserve deadlines:** re-enqueued requests keep their original deadlines
4. **Re-EDF:** after re-enqueue, `on_schedule_prefill` re-sorts by deadline

This ensures preempted requests don't lose their place in the fairness
ordering — they're re-enqueued with their original deadline and compete
fairly with newly arriving requests.

---

## 9. Configuration Parameters

| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| `tau` / `--tau-ms` | τ | 1000 ms | Maximum tolerable token delay in multi-tenant |
| `tau_cache_ratio` | — | 0.8 (80%) | Fraction of τ given to KV cache restoration |
| `tau_prefill_ratio` | — | 0.2 (20%) | Fraction of τ given to prefill scheduling |
| `tau_decode_ms` | τ_decode | 80 ms | Max tolerable per-decode-batch delay |
| `--num-clients` | \|C\| | 4 | Number of distinct clients/users |
| Perf model coeffs | αp, βp, γp, αd, βd, γd | (model/hw specific) | Fitted offline |

---

## 10. Interaction Flow Between Scheduling and Cache Policies

```
SCHEDULER PASS:
    │
    ├─ on_new_request(req)                    ← scheduler: new request arrives
    │    └─ compute initial deadline
    │    └─ init per-user EDF state
    │
    ├─ on_schedule_prefill(wq, rb, adder)     ← scheduler: form prefill batch
    │    └─ queries cache_policy.get_user_total_tokens(uid)  ← CACHE
    │    └─ queries cache_policy.fair_share_per_user          ← CACHE
    │    └─ reorder: EDF + fair-share → sorted queue
    │
    ├─ For each req in sorted queue:          ← scheduler: add loop
    │    └─ can_admit_request(uid, n, cache)   ← CACHE POLICY
    │         └─ returns True/False per user quota
    │
    ├─ on_prefill_vs_decode_decision(wq, rb, b) ← scheduler: commit
    │    └─ queries cache_policy for running user quotas   ← CACHE
    │    └─ returns "decode" if deadlines/headroom violated
    │
    ├─ on_prefill_decision(batch) OR          ← scheduler: dispatch
    │  on_decode_decision(batch)
    │    └─ charge/release headroom
    │    └─ record per-user progress
    │
    └─ on_end_of_scheduler_pass(batch)         ← scheduler: housekeeping
         └─ reconcile simulator
         └─ export metrics
```
