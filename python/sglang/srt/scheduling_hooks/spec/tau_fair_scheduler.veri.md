# τ-Token Fair Scheduler — Pure Hook Functions

This file is the **executable specification** of FairInference's τ-fair
scheduler, extracted from Algorithm 2 in the SOSP 2026 paper (§5.1–5.3).
When all `#TODO` functions are implemented and verified, the resulting C
code (linked via Cython) guarantees the **τ-fairness property**:

> For every well-behaved client and every token k generated in multi-tenant
> execution, the token's completion time t_k satisfies:
>
>     t_k ≤ T_ISO_k + τ
>
> where T_ISO_k is the token's completion time in isolated execution,
> and τ is the configurable delay tolerance.

Every hook is a **pure function**: all state arrives as explicit arguments.
The Python `NoOpSchedulingPolicy` subclass marshals args into these C calls.
All times in **microseconds** (μs), all ratios in **basis points** (0–10000).

Target: **f-star-c** — verified C via Low* → KaRaMeL, linked via Cython.

---

## Architecture

The scheduler has six hooks, matching the six decision points in Algorithm 2:

```
New request → Hook 1 (init deadline)
Scheduler pass begins → Hook 2 (prefill vs decode decision)
                      → Hook 3 (sort prefill queue)
                      → Hook 4 (prefill dispatched)
                      → Hook 5 (decode dispatched)
Scheduler pass ends   → Hook 6 (update headroom)
```

Between passes, the **alternate history simulator** (`DeadlineModel`) tracks
each request's anticipated next event based on isolated execution latency
estimates. The hooks query the alternate history to compute per-token
deadlines and make scheduling decisions.

---

## Key Invariants from the Paper

These invariants hold across all six hooks by construction:

1. **Per-token deadlines** (§5.3.2): `d_k = T_ISO_k + τ_component` where
   the first token gets `τ_prefill + τ_cache` and subsequent tokens get
   `τ_decode`.

2. **Headroom non-negativity** (§4.1, Lemma 4.1): `headroom ≥ 0` always.
   Headroom = Δdecode_ISO − Δdecode_MT; it accumulates when decodes are
   faster in multi-tenant execution than in isolation. Only positive
   headroom can be consumed by prefills.

3. **Prefill admission gate** (Algorithm 2, step 2d.ii): A prefill is only
   admitted if `accumulated_headroom ≥ prefill_cost` AND no running decode's
   deadline would be violated. Otherwise decode is forced.

4. **Fair-share prioritization** (Algorithm 2, step 2d.ii): Among requests
   with similar deadlines, users below their fair share are scheduled
   before over-quota users. This prevents well-behaved clients from being
   starved by high-demand clients.

5. **EDF ordering** (§5.1): Requests are ordered by deadline proximity:
   overdue (tier=0) before at-risk (tier=1) before safe (tier=2). Within
   each tier, fair-share ordering applies.

6. **τ-fair cache** (Algorithm 3): Each client gets a reservation
   `ρ_i = max(f_i − R_cache_i, 0)`. Well-behaved clients (u_i ≤ f_i)
   have their cached KV state protected; over-quota clients' excess
   tokens are eviction candidates.

---

## Domain Types

These types represent the state that flows through the scheduling hooks.
Every function is pure — it takes immutable inputs and returns new values
without side effects.

```veri
TARGET f-star-c
VERI_VERSION 0.0.2

import FStar.Seq
import DeadlineModel

class SchedReq:
    rid:          string(128)   # request ID
    uid:          string(128)   # client/user ID
    prompt_len:   int32         # prompt token count
    max_new_tok:  int32         # max new tokens to generate
    deadline_us:  int32         # next token deadline (μs epoch)
    is_running:   bool
    is_finished:  bool

class SchedBatch:
    reqs:       SchedReq[]
    sum_tok:    int32
    max_tok:    int32
    req_count:  int32
    is_prefill: bool
    is_empty:   bool

class FairShareState:
    uid:              string(128)
    total_kv:         int32     # u_i — total KV cache usage in tokens
    evictable_kv:     int32     # unlocked cache tokens
    fair_share:       int32     # f_i = M / |C| in tokens
    reservation:      int32     # ρ_i in tokens
    pending_prefills: int32
    running_decodes:  int32
    last_decode_ts_us: int32    # μs epoch
    last_prefill_ts_us: int32   # μs epoch
```

---

## EDF + Fair-Share Ordering Key

The priority queue for prefill scheduling uses a **compound ordering key**
(§5.1, Algorithm 2 step 2d.ii). Requests are sorted by three criteria in
order of precedence:

1. **Tier** (most important): How close the deadline is to the current time.
   - 0 = **overdue**: deadline has passed or is imminent
   - 1 = **at-risk**: deadline is within one τ of now
   - 2 = **safe**: deadline is at least τ away

2. **neg_shortfall** (how much under fair share): Requests from users who
   are well below their fair share (u_i ≪ f_i) get priority. This is
   stored as a negative number so lexical sort works: more negative = more
   under-quota = higher priority.

3. **excess** (how much over fair share): Requests from users who are over
   quota (u_i > f_i) go last. More excess = lower priority.

The tier classification is derived from the deadline: if `deadline_us + τ <
now_us`, the request is overdue; if `deadline_us - now_us >= τ`, it's safe;
otherwise it's at-risk.

```veri
class EDFOrderingKey:
    tier:          int32   # 0=overdue, 1=at-risk, 2=safe
    neg_shortfall: int32   # -max(0, f_i - u_i) — more under-quota first
    excess:        int32   # max(0, u_i - f_i) — more over-quota last

def edf_fairshare_key(
    deadline_us: int32,
    total_kv:    int32,
    fair_share:  int32,
    tau_us:      int32,
    now_us:      int32,
) -> EDFOrderingKey:
    REQUIRES (deadline_us >= 0 and tau_us >= 0 and now_us >= 0
              and total_kv >= 0 and fair_share >= 0)
    ENSURES (result.tier >= 0 and result.tier <= 2
             and result.neg_shortfall <= 0 and result.excess >= 0)
    #TODO (implement edf_fairshare_key)

def edf_compare(
    a: EDFOrderingKey,
    b: EDFOrderingKey,
) -> bool:
    # Returns true iff a has higher scheduling priority than b.
    # Priority: lower tier first, then more under-quota (more negative
    # neg_shortfall), then less over-quota (lower excess).
    ENSURES True
    #TODO (implement edf_compare)
```

---

## Fairness Predicates

These predicates come from the paper's definition of client behavior (§3).
A client is **well-behaved** if its KV cache usage is below its fair share.
The τ-fairness guarantee applies only to well-behaved clients. High-demand
clients (over quota) experience unbounded delays.

The **reservation** ρ_i = max(f_i − R_cache_i, 0) protects well-behaved
clients' cached KV state from eviction. A client within its reservation
has room to grow; a client over its reservation is an eviction candidate.

**Global slack** is the total KV cache capacity not locked by protected
tokens. When slack > 0, the system can admit requests from over-quota
users without evicting protected cache state. This is the admission
gate in Algorithm 2 step 2d.ii.

```veri
def is_well_behaved(total_kv: int32, fair_share: int32) -> bool:
    REQUIRES (total_kv >= 0 and fair_share >= 0)
    ENSURES result == (total_kv <= fair_share)
    #TODO (implement is_well_behaved)

def is_over_quota(total_kv: int32, fair_share: int32) -> bool:
    REQUIRES (total_kv >= 0 and fair_share >= 0)
    ENSURES result == (total_kv > fair_share)
    #TODO (implement is_over_quota)

def is_within_reservation(total_kv: int32, reservation: int32) -> bool:
    REQUIRES (total_kv >= 0 and reservation >= 0)
    ENSURES result == (total_kv <= reservation)
    #TODO (implement is_within_reservation)

def unevictable_tokens(total_kv: int32, evictable_kv: int32) -> int32:
    # locked = total - evictable; these tokens are protected by running decodes
    REQUIRES (total_kv >= evictable_kv and evictable_kv >= 0)
    ENSURES result == total_kv - evictable_kv
    #TODO (implement unevictable_tokens)

def eviction_spare(total_kv: int32, reservation: int32) -> int32:
    # tokens above reservation that can be evicted from over-quota users
    REQUIRES (total_kv >= 0 and reservation >= 0)
    ENSURES result >= 0
    #TODO (implement eviction_spare)
```

---

## HOOK 1 — on_new_request

**Purpose**: Initialize the first-token deadline when a new request enters
the waiting queue (Algorithm 2 step 2a).

**Algorithm (from §5.3.2)**:

The first-token deadline is the isolated prefill completion time plus the
prefill and cache delay tolerances:

```
d_0 = T_ISO_0 + τ_prefill + τ_cache
```

where:
- `T_ISO_0` is the prefill latency for this batch in isolated execution,
  computed from the affine performance model (see `DeadlineModel`)
- `τ_prefill` is the prefill scheduling delay tolerance (default: 20% of τ)
- `τ_cache` is the KV cache restoration delay tolerance (default: 80% of τ)

The alternate history simulator tracks this deadline and uses it to drive
the EDF ordering in subsequent hooks.

```veri
def on_new_request_init_deadline(
    alt_state: AltHistoryState,

    uid:               string(128),
    prompt_len:        int32,
    completion_len:    int32,
    iso_prefill_time_us: int32,
    tau_cache_us:      int32,
    tau_prefill_us:    int32,
    now_us:            int32,
) -> int32:
    REQUIRES (prompt_len > 0 and completion_len >= 0
              and iso_prefill_time_us >= 0
              and tau_cache_us >= 0 and tau_prefill_us >= 0
              and now_us >= 0)
    # The deadline must be at least now_us (cannot be in the past).
    # The deadline is exactly iso + τ components — this is the
    # τ-fairness guarantee for the first token.
    ENSURES (result >= now_us
             and result == now_us + iso_prefill_time_us + tau_cache_us + tau_prefill_us)
    #TODO (implement on_new_request_init_deadline)
```

---

## HOOK 2 — on_prefill_vs_decode_decision

**Purpose**: Decide whether to run prefill or decode work this scheduling
pass (Algorithm 2 step 2d.i).

**Algorithm (from §5.1 and §4.1)**:

The scheduler prioritizes prefills to maximize GPU utilization, but must
not violate the deadlines of currently running decode requests. The decision
uses the **accumulated headroom** (see Hook 6) and the **deadline safety check**:

1. If no candidate prefill exists → use default (prefill-first) behavior
2. If a decode request is about to miss its deadline → force decode
3. If insufficient headroom to cover the prefill cost → force decode
4. If a well-behaved user is being starved (below fair share but has pending
   requests) → force decode to free their KV cache tokens
5. Otherwise → allow prefill (run prefill batch)

The `accumulated_headroom_us` is the surplus from previous scheduling passes
where decodes ran faster than the isolated estimate. This headroom can safely
be "spent" on prefills without violating deadlines.

**Returns**: -1 = use default decision, 0 = force prefill, 1 = force decode.

```veri
def prefill_vs_decode_decision(
    alt_state: AltHistoryState,

    running_deadlines_us:   int32[],
    n_running:              int32,
    has_candidate_prefill:  bool,
    prefill_cost_us:        int32,
    accumulated_headroom_us: int32,
    delta_decode_mt_us:     int32,
    tau_us:                 int32,
    now_us:                 int32,
    any_over_quota:         bool,
    any_starved:            bool,
    max_per_user_active:    bool,
) -> int32:
    REQUIRES (n_running >= 0
              and delta_decode_mt_us >= 0
              and tau_us >= 0
              and now_us >= 0
              and prefill_cost_us >= 0
              and accumulated_headroom_us >= 0)
    # Result is always -1 (default), 0 (prefill), or 1 (decode).
    ENSURES (result == -1 or result == 0 or result == 1)
    #TODO (implement prefill_vs_decode_decision)
```

---

## HOOK 3 — on_schedule_prefill

**Purpose**: Reorder the waiting (prefill) queue by EDF + fair-share priority
(Algorithm 2 step 2d.ii).

**Algorithm**:

The queue is sorted using `edf_fairshare_key` and `edf_compare`:

1. Requests whose deadline has passed (overdue, tier=0) are scheduled first
2. Among requests with similar deadlines, users below their fair share get
   priority over over-quota users
3. This ensures that high-demand clients cannot starve well-behaved clients
   by filling the waiting queue with their own requests

The result is a permutation of the input indices — every request appears
exactly once in the sorted output.

```veri
def sort_prefill_queue(
    alt_state: AltHistoryState,

    deadlines_us: int32[],
    uids:         string(128)[],
    total_kvs:    int32[],
    fair_shares:  int32[],
    n:            int32,
    tau_us:       int32,
    now_us:       int32,
) -> int32[]:
    REQUIRES (n >= 0 and tau_us >= 0 and now_us >= 0)
    ENSURES array_len(result) == n
    #TODO (implement sort_prefill_queue)

def warn_overdue_deadlines(
    alt_state: AltHistoryState,

    deadlines_us: int32[],
    n:            int32,
    now_us:       int32,
) -> int32:
    # Returns count of requests whose deadline has already passed.
    # Used for monitoring: fires an alert if any well-behaved client
    # has overdue deadlines (indicates a τ violation).
    REQUIRES (n >= 0 and now_us >= 0)
    ENSURES result >= 0
    #TODO (implement warn_overdue_deadlines)
```

---

## HOOKS 4 + 5 — on_prefill_decision / on_decode_decision

**Purpose**: Record what was dispatched to the GPU for admission tracking
and fairness reporting.

**Usage in the paper's evaluation** (Table 2, Table 7):

`count_prefill_admitted` counts how many admitted requests are from
well-behaved clients. This is the metric used to verify that FairInference
is not unfairly admitting high-demand requests at the expense of
well-behaved clients.

`check_decode_deadlines` checks which decode requests have missed their
deadlines. A well-behaved client with overdue decode deadlines indicates
a τ violation — the fairness guarantee has been broken.

```veri
def count_prefill_admitted(
    alt_state: AltHistoryState,

    uids:        string(128)[],
    total_kvs:   int32[],
    fair_shares: int32[],
    n:           int32,
) -> int32:
    REQUIRES n >= 0
    ENSURES result >= 0 and result <= n
    #TODO (implement count_prefill_admitted)

def check_decode_deadlines(
    alt_state: AltHistoryState,

    deadlines_us: int32[],
    n:            int32,
    now_us:       int32,
) -> bool[]:
    REQUIRES (n >= 0 and now_us >= 0)
    ENSURES array_len(result) == n
    #TODO (implement check_decode_deadlines)
```

---

## HOOK 6 — on_end_of_scheduler_pass

**Purpose**: Update the accumulated headroom after one scheduling pass
(§4.1, §5.3). This is one of the most safety-critical hooks: the
headroom invariant guarantees that prefills never violate decode deadlines.

**Algorithm**:

Headroom is defined as the difference between isolated and multi-tenant
decode latency:

```
headroom = Δdecode_ISO − Δdecode_MT
```

After each scheduling pass:

- **Pure decode pass** (no prefill, no idle): headroom += Δdecode_ISO − Δdecode_MT
  This is always non-negative because `D_MT ≤ D_ISO` (Assumption §4.1).
  A faster decode in multi-tenant execution creates "slack" that can later
  be spent on prefills.

- **Pure prefill pass**: headroom -= prefill_cost
  The prefill consumes headroom that was accumulated from earlier decode
  passes. If there isn't enough headroom, the prefill should not have
  been admitted (this is enforced by Hook 2).

- **Idle or mixed pass**: headroom unchanged.

**Safety property** (Lemma 4.1): Headroom is never negative. This
guarantees that the cumulative prefill cost never exceeds the cumulative
decode speedup, which in turn guarantees that decode deadlines are never
violated by prefills.

```veri
def update_headroom(
    alt_state: AltHistoryState,

    old_headroom_us:       int32,
    was_prefill:           bool,
    prefill_cost_us:       int32,
    was_decode:            bool,
    delta_decode_iso_us:   int32,
    delta_decode_mt_us:    int32,
    was_idle:              bool,
) -> int32:
    REQUIRES (old_headroom_us >= 0
              and prefill_cost_us >= 0
              and delta_decode_iso_us >= 0
              and delta_decode_mt_us >= 0)
    # Headroom is always non-negative — the safety proof depends on this.
    # Decode creates headroom (iso - mt); prefill consumes it.
    ENSURES (result >= 0
             and (was_decode and not was_prefill and not was_idle
                  ==> result == old_headroom_us + delta_decode_iso_us - delta_decode_mt_us)
             and (was_idle or (was_prefill and was_decode)
                  ==> result == old_headroom_us))
    #TODO (implement update_headroom)
```

---

## Admission Gate & Deadline Safety

These predicates encode the **admission gate** from Algorithm 2 step 2d.ii:

> "It does not admit client requests if the token deadlines for decode
> requests can be violated from admitting this request for prefill."

**Admission rules**:

1. `can_admit_request`: A request can be admitted if the KV cache has space
   AND (the user is within their reservation OR there is global slack).
   This prevents a single high-demand client from consuming all KV cache
   at the expense of well-behaved clients.

2. `compute_global_slack`: The total KV cache capacity NOT locked by
   protected (unevictable) tokens. Slack > 0 means there's room to admit
   requests from over-quota users without evicting protected state.

3. `would_violate_decode_deadlines`: Runs the forward check: if we admit
   this prefill (which takes `prefill_cost_us` time), would any running
   decode request miss its deadline? Uses the formula from Algorithm 2:
   `d_r − now < Δdecode_MT + prefill_cost − headroom`.

```veri
def can_admit_request(
    alt_state: AltHistoryState,

    uid:               string(128),
    needed_tokens:     int32,
    total_kv:          int32,
    evictable_kv:      int32,
    fair_share:        int32,
    reservation:       int32,
    global_slack:      int32,
    cache_has_space:   bool,
) -> bool:
    REQUIRES (needed_tokens >= 0 and total_kv >= 0 and evictable_kv >= 0
              and fair_share >= 0 and reservation >= 0 and global_slack >= 0)
    ENSURES True
    #TODO (implement can_admit_request)

def compute_global_slack(
    alt_state: AltHistoryState,

    total_kvs: int32[],
    n:         int32,
    max_kv:    int32,
) -> int32:
    REQUIRES (n >= 0 and max_kv >= 0)
    ENSURES (result >= 0 and result <= max_kv)
    #TODO (implement compute_global_slack)

def would_violate_decode_deadlines(
    alt_state: AltHistoryState,

    prefill_cost_us:       int32,
    headroom_us:           int32,
    running_deadlines_us:  int32[],
    n_running:             int32,
    delta_decode_mt_us:    int32,
    now_us:                int32,
) -> bool:
    REQUIRES (prefill_cost_us >= 0
              and headroom_us >= 0
              and n_running >= 0
              and delta_decode_mt_us >= 0
              and now_us >= 0)
    ENSURES True
    #TODO (implement would_violate_decode_deadlines)
```

---

## What These Contracts Prove

When fully implemented and verified, these contracts guarantee:

1. **Bounded latency**: Every well-behaved client's token latency is bounded
   by `T_ISO + τ`. The per-token deadline (Hook 1) sets the bound, the
   EDF ordering (Hook 3) prioritizes requests closest to their deadline,
   and the headroom invariant (Hook 6) prevents prefill-induced delays.

2. **No starvation**: Well-behaved clients (u_i ≤ f_i) are always
   scheduled before over-quota clients when deadlines are similar.
   This is enforced by the fair-share ordering in the EDF key.

3. **Cache protection**: Well-behaved clients' KV cache state is protected
   from eviction by the reservation system (`is_within_reservation`,
   `eviction_spare`). Over-quota clients' excess tokens are evicted first.

4. **Resource safety**: The admission gate (`can_admit_request`,
   `would_violate_decode_deadlines`) prevents the scheduler from admitting
   requests that would cause decode deadlines to be missed or that would
   exhaust the KV cache.

5. **Headroom safety**: The headroom invariant (`headroom ≥ 0`) guarantees
   that the cumulative prefill cost never exceeds the cumulative decode
   speedup, so prefills never delay decodes beyond their deadlines.

Together these invariants imply the τ-fairness guarantee (§3).
```
