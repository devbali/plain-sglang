# Deadline Model — Per-Token Deadlines & Alternate History Simulator

This file is the **executable specification** of FairInference's performance
model and alternate-history simulator, extracted from §5.3–5.4 of the SOSP
2026 paper. When all `#TODO` functions are implemented and verified, the
resulting C code (linked via Cython) provides the per-token deadline
estimates that the τ-fair scheduler uses to make scheduling decisions.

Every function is **pure**: all state arrives as explicit arguments.
All times in **microseconds** (μs), all ratios in **basis points** (0–10000).

Target: **f-star-c** — verified C via Low* → KaRaMeL, linked via Cython.

---

## Architecture

The deadline model has two layers:

**Layer 1 — Latency estimation** (§5.4): Affine functions that predict how
long a prefill or decode batch takes in isolated execution (no interference
from other clients). These are multiplied by the number of clients |C| to
produce the isolated latency estimates used in deadline computation.

**Layer 2 — Alternate-history simulator** (§5.3.2): A state machine that
tracks each request's anticipated next event based solely on isolated
execution latency estimates. The real scheduler feeds in admission times
and completion signals; the simulator advances independently using the
constant performance model. The anticipated events are used to compute
per-token deadlines.

---

## Key Assumptions from the Paper

1. **Decode latency in MT ≤ Decode latency in isolation** (§4.1):
   `Δdecode_MT ≤ Δdecode_ISO`. This ensures headroom is always non-negative,
   which is the foundation of the τ-fairness safety proof.

2. **Affine latency model** (§5.4): Both prefill and decode latency are
   affine functions of batch size (S), max token length (M), and request
   count (N). Coefficients are fitted via OLS regression on the target
   hardware (A100 80GB with Llama-3 8B Instruct).

3. **Isolated execution multiplier**: To estimate latency in isolation,
   the multi-tenant latency estimate is multiplied by |C| (number of
   clients). This models the fact that in isolation, a client gets
   1/|C| of the GPU resources, so operations take |C|× longer.

4. **τ decomposition**: The total delay tolerance τ is split into
   τ_cache (caching delay, default 80%) and τ_prefill (scheduling delay,
   default 20%) for the first token, and τ_decode for subsequent tokens.

---

## Performance Coefficients

The affine latency model from §5.4:

```
Δprefill(μs) = (cp_ns + αp·S + βp·M + γp·N) / 1000
Δdecode(μs)  = (cd_ns + αd·S + βd·M + γd·N) / 1000
```

where:
- S = sum of token lengths across all requests in the batch
- M = maximum token length among requests in the batch
- N = number of requests in the batch

For isolated execution, multiply by |C| (number of clients):
```
Δprefill_ISO(μs) = |C| · Δprefill(μs)
Δdecode_ISO(μs)  = |C| · Δdecode(μs)
```

The paper reports these fitted values for A100 80GB with Llama-3 8B:
- Prefill: cp=9.2e-3, αp=6.1e-5, βp=2.3e-6 (all in seconds; convert to ns)
- Decode:  cd=1.2e-2, αd=6.8e-8, βd=2.6e-7, γd=5.7e-5

```veri
TARGET fstar-c
VERI_VERSION 0.0.2

import FStar.Seq

class PerfCoeffs:
    cp_ns:      int32   # prefill constant overhead (ns)
    alpha_p_nspt: int32 # prefill per-token coefficient (ns/tok)
    beta_p_nspt:  int32 # prefill per-max-token coefficient (ns/tok)
    gamma_p_nspt: int32 # prefill per-request coefficient (ns/req)
    cd_ns:      int32   # decode constant overhead (ns)
    alpha_d_nspt: int32 # decode per-token coefficient (ns/tok)
    beta_d_nspt:  int32 # decode per-max-token coefficient (ns/tok)
    gamma_d_nspt: int32 # decode per-request coefficient (ns/req)
```

---

## τ Breakdown

The total delay tolerance τ is decomposed into three components (§5.3,
Algorithm 1 steps 2a-2b):

- **τ_cache**: KV cache restoration delay tolerance. Default is 80% of τ.
  This accounts for the worst-case time to recompute or reload evicted
  KV cache state for a well-behaved client.

- **τ_prefill**: Prefill scheduling delay tolerance. Default is the
  remainder after τ_cache (typically 20% of τ). This accounts for the
  maximum time a prefill can be delayed by running decode batches.

- **τ_decode**: Per-decode-step scheduling delay tolerance. This is
  independently configured (not derived from τ) because decode steps
  are much shorter than prefills and need a tighter bound.

```veri
class TauBreakdown:
    tau_us:       int32   # total delay tolerance
    tau_cache_us: int32   # caching delay = tau_us * cache_ratio_bps / 10000
    tau_prefill_us: int32 # prefill scheduling delay = tau_us - tau_cache_us
    tau_decode_us: int32  # decode scheduling delay (independently configured)
```

---

## Per-Token Deadline

A deadline is generated for every token of every request (§5.3.2). The
deadline `d_k = T_ISO_k + τ_component` where `T_ISO_k` is the token's
predicted completion time in isolated execution, and `τ_component` depends
on whether it's the first token (τ_prefill + τ_cache) or a subsequent
token (τ_decode).

The alternate history simulator tracks these deadlines in a sorted list
for efficient EDF scheduling.

```veri
class TokenDeadline:
    rid:           string(128)
    uid:           string(128)
    token_index:   int32         # 0 = first token
    iso_latency_us: int32        # T_ISO_k — isolated completion time
    tau_applied_us: int32        # which τ component was added
    deadline_us:   int32         # d_k = T_ISO_k + τ (absolute μs epoch)
```

---

## Alternate History Request & State Machine

The alternate-history simulator maintains a **state machine** for each
request. The status phase advances monotonically:

```
    NEW
     │
     ▼  (push_request → Hook 1)
  IN_QUEUE (status=0, waiting for prefill)
     │
     ▼  (prefill admitted → Hook 4)
  ADMITTED (status=1, prefill started)
     │
     ▼  (prefill completes → anticipated event fires)
  PREFILL_DONE (status=2, awaiting decode)
     │
     ▼  (decode step → anticipated event fires, repeated)
  DECODE_DONE (status=3, decode_count tracks position)
     │
     ▼  (all tokens generated → real completion → reconcile)
  COMPLETE (status=4, no more anticipated events)
```

At each state, the simulator computes an **anticipated event**: the next
scheduling event predicted to occur, assuming no interference. For
IN_QUEUE/ADMITTED, the next event is prefill completion (ant_type=0).
For PREFILL_DONE/DECODE_DONE, the next event is a decode step (ant_type=1)
unless all tokens have been generated, in which case there is no event
(ant_type=-1).

**Status phases**:
- 0 = IN_QUEUE (waiting for prefill)
- 1 = ADMITTED (prefill started)
- 2 = PREFILL_DONE (prefill completed, awaiting decode)
- 3 = DECODE_DONE (decode in progress)
- 4 = COMPLETE (all tokens generated)

**Anticipated event types**:
- -1 = ANT_NONE (no event — request complete or uninitialized)
-  0 = ANT_PREFILL (next event is prefill completion)
-  1 = ANT_DECODE (next event is a decode step)

```veri
class AltReq:
    rid:              string(128)
    uid:              string(128)
    prompt_len:       int32        # total prompt tokens
    completion_len:   int32        # total completion tokens planned
    status:           int32        # 0=queue,1=admitted,2=prefill_done,3=decode_done,4=complete
    decode_count:     int32        # decode tokens generated so far (valid when status >= 2)
    arrival_time_us:  int32        # real wall-clock arrival (μs epoch)
    iso_prefill_time_us: int32     # T_ISO_0 — when prefill completes in isolation
    iso_decode_time_us: int32      # Δdecode_ISO per step (μs)
    ant_type:         int32        # -1=none, 0=prefill, 1=decode
    ant_end_ts_us:    int32        # predicted wall-clock end of anticipated event (μs)

CONSTRAINT AltReqInvariants:
    status >= 0 and status <= 4
    and (status < 2 ==> decode_count == 0)
    and (status >= 2 ==> decode_count >= 0 and decode_count <= completion_len)
    and (status < 4 ==> ant_type != -1)
    and (status == 4 ==> ant_type == -1)
    and (ant_type >= 0 ==> ant_end_ts_us > arrival_time_us)
    and prompt_len > 0 and completion_len >= 0 and arrival_time_us >= 0
```

---

## User Status

Requests are grouped by user (client). Within each user, request IDs are
unique. The simulator tracks all requests per user to compute per-user
KV cache usage and fair-share status.

```veri
class UserStatus:
    uid:       string(128)
    requests:  AltReq[]
    n_requests: int32

CONSTRAINT UserStatusInvariants:
    n_requests >= 0
    and array_len(requests) >= n_requests
    and FORALL i IN range(0, n_requests):
        FORALL j IN range(i + 1, n_requests):
            requests[i].rid != requests[j].rid
```

---

## Simulator State

The top-level state of the alternate-history simulator. All hooks read
from and write to this state. The `current_time_us` advances monotonically
as the simulator predicts future events.

```veri
class AltHistoryState:
    current_time_us: int32         # simulator wall-clock (μs epoch)
    num_users:       int32         # |C| — total number of users
    max_kv:          int32         # max KV cache tokens
    users:           UserStatus[]  # per-user state
    n_users:         int32         # live entries in users[]
    deadlines:       TokenDeadline[]  # sorted pending deadlines
    n_deadlines:     int32

CONSTRAINT AltHistoryInvariants:
    current_time_us >= 0 and num_users > 0 and max_kv > 0
    and n_users >= 0 and n_users <= num_users
    and n_deadlines >= 0
```

---

## Latency Model — Pure Functions

These functions implement the affine latency model from §5.4. All
coefficients are in nanoseconds; results are converted to microseconds
by integer division by 1000.

**Isolated prefill latency**: predicts how long a prefill batch takes in
isolated execution. Multiplied by |C| to account for the fact that in
isolation, a client gets only 1/|C| of GPU resources.

**Isolated decode latency**: predicts decode step latency in isolation.
Similarly multiplied by |C|.

**Multi-tenant decode latency**: predicts decode step latency in
multi-tenant execution (no multiplier). This is the actual measured
latency and is always ≤ the isolated estimate (Assumption §4.1).

```veri
def isolated_prefill_latency(
    prompt_len:  int32,
    max_tok:     int32,
    req_count:   int32,
    coeffs:      PerfCoeffs,
    num_clients: int32,
) -> int32:
    # Δprefill_ISO_μs = |C| · (cp_ns + αp·S + βp·M + γp·N) / 1000
    REQUIRES (prompt_len >= 0 and max_tok >= 0 and req_count >= 0
              and num_clients > 0 and coeffs.cp_ns >= 0)
    ENSURES result >= 0
    #TODO (implement isolated_prefill_latency)

def isolated_decode_latency(
    sum_tok:     int32,
    max_tok:     int32,
    req_count:   int32,
    coeffs:      PerfCoeffs,
    num_clients: int32,
) -> int32:
    # Δdecode_ISO_μs = |C| · (cd_ns + αd·S + βd·M + γd·N) / 1000
    REQUIRES (sum_tok >= 0 and max_tok >= 0 and req_count >= 0
              and num_clients > 0 and coeffs.cd_ns >= 0)
    ENSURES result >= 0
    #TODO (implement isolated_decode_latency)

def mt_decode_latency(
    sum_tok:   int32,
    max_tok:   int32,
    req_count: int32,
    coeffs:    PerfCoeffs,
) -> int32:
    # Δdecode_MT_μs = (cd_ns + αd·S + βd·M + γd·N) / 1000
    REQUIRES (sum_tok >= 0 and max_tok >= 0 and req_count >= 0
              and coeffs.cd_ns >= 0)
    ENSURES result >= 0
    #TODO (implement mt_decode_latency)
```

---

## Headroom Computation

Headroom is the "slack" that builds up when decodes run faster in
multi-tenant execution than in isolation (§4.1). This slack is tracked
by the scheduler (Hook 6) and enables safe prefill admission.

```
headroom = Δdecode_ISO − Δdecode_MT
```

When headroom > 0, the accumulated surplus can be used to schedule
prefills without violating decode deadlines. When prefill runs, its
cost is deducted from the headroom. The headroom safety invariant
guarantees that the cumulative prefill cost never exceeds the cumulative
decode speedup.

```veri
def compute_headroom(
    delta_decode_iso_us: int32,
    delta_decode_mt_us:  int32,
) -> int32:
    # headroom_μs = Δdecode_ISO − Δdecode_MT
    # Positive → decodes faster in MT; slack for prefills (§4.1)
    REQUIRES (delta_decode_iso_us >= 0 and delta_decode_mt_us >= 0)
    ENSURES True
    #TODO (implement compute_headroom)
```

---

## τ Decomposition

Decomposes the total delay tolerance τ into its components. The cache
ratio determines how much of τ is allocated to KV cache restoration
delays vs. prefill scheduling delays.

Per the paper's implementation (§6): "80% of the delay tolerance is
statically allocated to caching (τ_cache) and 20% to prefill (τ_prefill)."

τ_decode is set independently (typically 80ms per batch on A100) because
decode steps are much shorter than prefills.

```veri
def split_tau(
    tau_us:            int32,
    cache_ratio_bps:   int32,
    tau_decode_fixed_us: int32,
) -> TauBreakdown:
    # τ_cache = τ · ratio/10000, τ_prefill = τ − τ_cache, τ_decode = fixed
    REQUIRES (tau_us >= 0 and cache_ratio_bps >= 0 and cache_ratio_bps <= 10000
              and tau_decode_fixed_us >= 0)
    ENSURES (result.tau_us == tau_us
             and result.tau_cache_us == tau_us * cache_ratio_bps / 10000
             and result.tau_prefill_us == tau_us - result.tau_cache_us
             and result.tau_decode_us == tau_decode_fixed_us)
    #TODO (implement split_tau)
```

---

## Status Transition Functions

These functions determine what the next anticipated event is for a
request, given its current status in the alternate history.

**Status transitions**:
- New request (inferred from status=0): always → ANT_PREFILL (ant_type=0)
- Prefill done (status=1): if completion_len == 0 → ANT_NONE (complete),
  otherwise → ANT_DECODE (ant_type=1), first token generated
- Decode done (status>=2): if decode_count >= completion_len → ANT_NONE
  (complete), otherwise → ANT_DECODE (ant_type=1), next token

```veri
def compute_new_req_ant_type(req: AltReq) -> int32:
    # New request always anticipates a prefill completion.
    ENSURES result == 0
    #TODO (implement compute_new_req_ant_type)

def compute_prefill_done_ant_type(req: AltReq) -> int32:
    # Prefill just finished. If there are completion tokens to generate,
    # the next anticipated event is a decode step (1).
    # If no completion tokens (completion_len == 0), the request is
    # complete and has no anticipated event (-1).
    REQUIRES (req.status == 1 and req.prompt_len > 0)
    ENSURES (result == -1 or result == 1)
    #TODO (implement compute_prefill_done_ant_type)

def compute_decode_done_ant_type(req: AltReq) -> int32:
    # A decode step just finished. If there are more tokens to generate
    # (decode_count < completion_len), the next anticipated event is
    # another decode step (1). Otherwise the request is complete (-1).
    REQUIRES (req.status >= 2 and req.decode_count >= 0
              and req.decode_count <= req.completion_len)
    ENSURES (result == -1 or result == 1)
    #TODO (implement compute_decode_done_ant_type)
```

---

## Anticipated Event Timing Functions

These functions compute the predicted wall-clock time of the anticipated
event, assuming no interference from other clients.

- **Prefill end time**: arrival time + isolated prefill duration.
  `ant_end_ts_us = arrival_time_us + iso_prefill_time_us`

- **Decode end time**: end of previous event + isolated decode duration.
  `ant_end_ts_us = previous_ant_end_us + iso_decode_time_us`

- **Deadline from anticipated event**: the deadline is the anticipated
  event completion time plus the applicable τ component.
  `d_k = ant_end_ts_us + tau_applied_us`

For the first token, `tau_applied_us = τ_cache + τ_prefill`.
For subsequent tokens, `tau_applied_us = τ_decode`.

```veri
def compute_ant_prefill_end_us(
    arrival_time_us: int32,
    iso_prefill_time_us: int32,
) -> int32:
    # Prefill end = arrival + isolated prefill latency
    REQUIRES (arrival_time_us >= 0 and iso_prefill_time_us >= 0)
    ENSURES result >= arrival_time_us
    #TODO (implement compute_ant_prefill_end_us)

def compute_ant_decode_end_us(
    previous_ant_end_us: int32,
    iso_decode_time_us: int32,
) -> int32:
    # Decode end = previous event end + isolated decode latency
    REQUIRES (previous_ant_end_us >= 0 and iso_decode_time_us >= 0)
    ENSURES result >= previous_ant_end_us
    #TODO (implement compute_ant_decode_end_us)

def compute_deadline_from_ant(
    ant_end_ts_us:   int32,
    tau_applied_us:  int32,
) -> int32:
    # Deadline = anticipated event completion + τ delay tolerance
    REQUIRES (ant_end_ts_us >= 0 and tau_applied_us >= 0)
    ENSURES result >= ant_end_ts_us
    #TODO (implement compute_deadline_from_ant)
```

---

## Core Alternate History Functions

These functions manage the alternate-history simulator state. Each is a
pure state transformer — it takes an `AltHistoryState` and returns a new
`AltHistoryState` without mutating the original.

**Lifecycle**:
1. `alt_history_init` — creates the initial state
2. `alt_history_push_request` — adds a new request (called from Hook 1)
3. `alt_history_advance` — advances simulator time and updates anticipated
   events (called at the start of each scheduling pass)
4. `alt_history_extract_deadlines` — extracts the sorted list of pending
   deadlines for use by the EDF scheduler
5. `alt_history_complete_request` — marks a request complete (called from
   Hook 4 or Hook 5 when a decode finishes)
6. `alt_history_reconcile` — syncs the simulator with the real scheduler
   state: any request marked complete in the real world must be reflected
   in the alternate history so it stops tracking anticipated events
7. `alt_history_get_user_token_count` — returns total KV cache tokens
   for a user (used for fair-share computations)

```veri
def alt_history_init(
    num_users:  int32,
    start_time_us: int32,
    max_kv:     int32,
) -> AltHistoryState:
    # Initializes the alternate history with no requests.
    REQUIRES (num_users > 0 and start_time_us >= 0 and max_kv > 0)
    ENSURES (result.current_time_us == start_time_us
             and result.num_users == num_users
             and result.max_kv == max_kv
             and result.n_users == 0
             and result.n_deadlines == 0)
    #TODO (implement alt_history_init)

def alt_history_push_request(
    state:  AltHistoryState,
    rid:    string(128),
    uid:    string(128),
    prompt_len: int32,
    completion_len: int32,
    arrival_time_us: int32,
    coeffs: PerfCoeffs,
    tb:     TauBreakdown,
) -> AltHistoryState:
    # Adds a new request to the alternate history (Hook 1).
    # Computes isolated prefill time, anticipated prefill event,
    # and the first-token deadline from the performance model.
    REQUIRES (state.num_users > 0
              and prompt_len > 0 and completion_len >= 0
              and arrival_time_us >= 0)
    ENSURES (result.n_users >= state.n_users
             or result.n_deadlines >= state.n_deadlines + 1)
    #TODO (implement alt_history_push_request)

def alt_history_advance(
    state:  AltHistoryState,
    coeffs: PerfCoeffs,
    tb:     TauBreakdown,
) -> AltHistoryState:
    # Advances the simulator to the next anticipated event.
    # Updates statuses and computes new deadlines for affected requests.
    # Must terminate (DECREASES clause).
    REQUIRES (state.num_users > 0)
    ENSURES (result.current_time_us >= state.current_time_us
             and FORALL u IN result.users:
                 FORALL r IN u.requests:
                     r.status < 4 ==> r.ant_type != -1
             and FORALL u IN result.users:
                 FORALL r IN u.requests:
                     r.status == 4 ==> r.ant_type == -1
             and FORALL u IN result.users:
                 FORALL r IN u.requests:
                     r.ant_type >= 0 ==> r.ant_end_ts_us > r.arrival_time_us)
    DECREASES state.num_users
    #TODO (implement alt_history_advance)

def alt_history_extract_deadlines(
    state: AltHistoryState,
) -> TokenDeadline[]:
    # Extracts all pending deadlines from the alternate history,
    # sorted by deadline_us for EDF scheduling (Hook 3).
    REQUIRES state.num_users > 0
    ENSURES True
    #TODO (implement alt_history_extract_deadlines)

def alt_history_complete_request(
    state: AltHistoryState,
    rid:   string(128),
) -> AltHistoryState:
    # Marks a request as complete (status=4, ant_type=-1).
    # Called when a decode finishes in the real scheduler (Hook 4/5).
    REQUIRES True
    ENSURES FORALL u IN result.users:
                FORALL r IN u.requests:
                    r.rid == rid ==> r.status == 4 and r.ant_type == -1
    #TODO (implement alt_history_complete_request)

def alt_history_reconcile(
    state:       AltHistoryState,
    real_time_us: int32,
    completed_rids: string(128)[],
    n_completed: int32,
) -> AltHistoryState:
    # Syncs simulator time and marks real-world-completed requests done.
    # Ensures alternate history doesn't track events for completed requests.
    REQUIRES (real_time_us >= state.current_time_us and n_completed >= 0)
    ENSURES (result.current_time_us == real_time_us
             and FORALL i IN range(0, n_completed):
                 FORALL u IN result.users:
                     FORALL r IN u.requests:
                         r.rid == completed_rids[i] ==>
                         r.status == 4 and r.ant_type == -1)
    #TODO (implement alt_history_reconcile)

def alt_history_get_user_token_count(
    state: AltHistoryState,
    uid:   string(128),
) -> int32:
    # Returns total KV cache tokens for a user across all their requests.
    REQUIRES True
    ENSURES result >= 0
    #TODO (implement alt_history_get_user_token_count)
```

---

## Cython Linkage

The Python `FairInferenceSchedulingPolicy` instantiates an `AltHistoryModel`
object that wraps the C functions via Cython. The hooks call into the
alternate history to compute deadlines.

```python
# cython pseudocode — Python hooks call these C functions

cdef class AltHistoryModel:
    cdef AltHistoryState* _state
    cdef PerfCoeffs _coeffs
    cdef TauBreakdown _tb

    def push_request(self, rid, uid, prompt_len, completion_len, arrival_us):
        self._state = alt_history_push_request(
            self._state, rid, uid, prompt_len, completion_len,
            arrival_us, self._coeffs, self._tb)

    def advance(self):
        self._state = alt_history_advance(self._state, self._coeffs, self._tb)

    def get_deadlines(self):
        return alt_history_extract_deadlines(self._state)

    def complete_request(self, rid):
        self._state = alt_history_complete_request(self._state, rid)

    def reconcile(self, real_time_us, completed_rids):
        self._state = alt_history_reconcile(
            self._state, real_time_us, completed_rids, len(completed_rids))

    def get_user_token_count(self, uid):
        return alt_history_get_user_token_count(self._state, uid)
```
