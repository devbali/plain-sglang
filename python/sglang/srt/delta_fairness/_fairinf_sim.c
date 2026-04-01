/*
 * _fairinf_sim.c — CPython C extension for the fairinf simulator hot path.
 *
 * Implements the exact same logic as UserTimeline.rebuild_from_real_state()
 * in doc_policy_simulator.py, running entirely without the GIL.
 *
 * Public API (called from Python):
 *
 *   results = _fairinf_sim.rebuild_kernel(
 *       rids,               # List[str]
 *       arrival_times,      # List[float]
 *       prompt_lens,        # List[int]
 *       real_decode_counts, # List[int]
 *       prefill_dones,      # List[int]  (0/1)
 *       is_completes,       # List[int]  (0/1)
 *       max_kv_tokens,      # int  (-1 = unlimited)
 *       fairinf_n,          # int
 *   ) -> List[Tuple[str, int, float, int]]
 *        Each tuple: (rid, event_type, end_timestamp, completion_number)
 *        event_type: 0 = prefill, 1 = decode
 *        Only requests that received an anticipated event are included.
 *        Requests that ran out of steps get a prefill event with end_ts=inf.
 *
 *   _fairinf_sim.patch_simulator(sim)
 *        Monkey-patches an AlternateHistorySimulator instance so that every
 *        UserTimeline.rebuild_from_real_state() call goes through the C kernel
 *        instead of the Python loop.  Used by USE_C_SIM in simulator_thread.py
 *        and directly by tests.
 *
 * Build:
 *   cd delta_fairness/
 *   python3.12 setup_fairinf_sim.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/*
 * Allocations inside run_rebuild_kernel happen with the GIL released.
 * PyMem_Malloc requires the GIL in CPython 3.12+, so we use plain malloc/free
 * for all allocations that live entirely inside the GIL-free section.
 * PyMem_Malloc / PyMem_Free are still used in py_rebuild_kernel for the
 * reqs[] and tracked_objs[] arrays which are allocated/freed with GIL held.
 */
#define SIM_MALLOC  malloc
#define SIM_FREE    free
#define SIM_REALLOC realloc

/* -------------------------------------------------------------------------
 * Constants mirrored from Python sources
 * ------------------------------------------------------------------------- */
#define RETRACTION_PENALTY_SECONDS  0.030

/* time_estimation.py constants */
#define DECODE_CONST_OVERHEAD   0.020   /* CONST_INTERVAL_DECODE */
#define PREFILL_CONST_OVERHEAD  0.080   /* CONST_INTERVAL_PREFILL */
#define TBT_DELTA               0.080
#define MAX_POOLED_DECODE_LATENCY  0.07092250719117761  /* pre-computed */
#define PREFILL_PIECEWISE_BOUND  4064.0

/* -------------------------------------------------------------------------
 * Time estimation — exact translation of time_estimation.py
 * ------------------------------------------------------------------------- */

static double isolated_prefill_time_estimation(
        double total_batch_sum, double max_token_size,
        double batch_length, double n)
{
    double v;
    if (total_batch_sum <= PREFILL_PIECEWISE_BOUND) {
        v = 9.15608285e-03
            + 6.14834557e-05 * n * total_batch_sum
            + 2.26526916e-06 * n * max_token_size
            + -1.52741501e-05 * n * batch_length;
    } else {
        v = -4.03734644e-02
            + 6.62669482e-05 * n * total_batch_sum
            + 1.42083211e-05 * n * max_token_size
            + -1.02748344e-04 * n * batch_length;
    }
    return v > 5e-3 ? v : 5e-3;
}

static double isolated_decode_time_estimation(
        double total_batch_sum, double max_token_size,
        double batch_length, double n)
{
    double inner = 1.18629080e-02
        + 6.81862494e-08 * n * total_batch_sum
        + 2.62872519e-07 * n * max_token_size
        + 5.65921863e-05 * n * batch_length;
    double v = inner > MAX_POOLED_DECODE_LATENCY ? inner : MAX_POOLED_DECODE_LATENCY;
    return DECODE_CONST_OVERHEAD + v + TBT_DELTA;
}

/* -------------------------------------------------------------------------
 * Per-request state for the C simulation
 * ------------------------------------------------------------------------- */

#define RID_MAX 128

typedef struct {
    char     rid[RID_MAX];
    double   arrival_ts;
    int      prompt_len;
    int      real_decode_count;
    int      prefill_done;    /* from requests_real */
    int      is_complete;

    /* output written by rebuild_kernel */
    int      ant_type;        /* -1=none, 0=prefill, 1=decode */
    double   ant_end_ts;
    int      ant_completion;
} CReq;

/* -------------------------------------------------------------------------
 * Simple open-addressing hash map: rid (string) -> index into CReq array
 * ------------------------------------------------------------------------- */

#define HT_EMPTY  -1

typedef struct {
    char key[RID_MAX];
    int  val;           /* index into reqs[] */
} HTSlot;

typedef struct {
    HTSlot  *slots;
    int      cap;       /* power of 2 */
    int      mask;
} HashMap;

static void hm_init(HashMap *hm, int n)
{
    int cap = 4;
    while (cap < n * 2) cap <<= 1;
    hm->slots = (HTSlot *)SIM_MALLOC(cap * sizeof(HTSlot));
    hm->cap   = cap;
    hm->mask  = cap - 1;
    for (int i = 0; i < cap; i++) {
        hm->slots[i].key[0] = '\0';
        hm->slots[i].val    = HT_EMPTY;
    }
}

static void hm_free(HashMap *hm)
{
    SIM_FREE(hm->slots);
    hm->slots = NULL;
}

/* FNV-1a hash */
static unsigned int hm_hash(const char *s)
{
    unsigned int h = 2166136261u;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

static void hm_put(HashMap *hm, const char *key, int val)
{
    unsigned int h = hm_hash(key) & hm->mask;
    while (hm->slots[h].val != HT_EMPTY &&
           strcmp(hm->slots[h].key, key) != 0) {
        h = (h + 1) & hm->mask;
    }
    strncpy(hm->slots[h].key, key, RID_MAX - 1);
    hm->slots[h].key[RID_MAX - 1] = '\0';
    hm->slots[h].val = val;
}

static int hm_get(const HashMap *hm, const char *key)
{
    unsigned int h = hm_hash(key) & hm->mask;
    for (;;) {
        if (hm->slots[h].val == HT_EMPTY)   return HT_EMPTY;
        if (strcmp(hm->slots[h].key, key) == 0) return hm->slots[h].val;
        h = (h + 1) & hm->mask;
    }
}

/* -------------------------------------------------------------------------
 * Deque of indices (for waiting_rids)
 * ------------------------------------------------------------------------- */

typedef struct {
    int   *buf;
    int    head, tail, cap;
} Deque;

static void dq_init(Deque *dq, int cap)
{
    dq->buf  = (int *)SIM_MALLOC(cap * sizeof(int));
    dq->head = 0;
    dq->tail = 0;
    dq->cap  = cap;
}

static void dq_free(Deque *dq) { SIM_FREE(dq->buf); dq->buf = NULL; }
static int  dq_len(const Deque *dq) { return dq->tail - dq->head; }
static int  dq_empty(const Deque *dq) { return dq->tail == dq->head; }

/* grow if needed */
static void dq_grow(Deque *dq)
{
    int len = dq_len(dq);
    int new_cap = dq->cap * 2;
    int *nb = (int *)SIM_MALLOC(new_cap * sizeof(int));
    for (int i = 0; i < len; i++)
        nb[i] = dq->buf[dq->head + i];
    SIM_FREE(dq->buf);
    dq->buf  = nb;
    dq->head = 0;
    dq->tail = len;
    dq->cap  = new_cap;
}

static void dq_push_back(Deque *dq, int v)
{
    if (dq->tail == dq->cap) dq_grow(dq);
    dq->buf[dq->tail++] = v;
}

static void dq_push_front(Deque *dq, int v)
{
    if (dq->head == 0) {
        /* No room at the front — shift everything right by 1. */
        int len = dq_len(dq);
        /* Ensure we have space: tail + 1 <= cap */
        if (dq->tail + 1 > dq->cap) dq_grow(dq);
        memmove(dq->buf + dq->head + 1, dq->buf + dq->head, len * sizeof(int));
        /* head stays at 0, tail advances by 1, then we write buf[0] */
        dq->tail = dq->head + len + 1;
        dq->buf[dq->head] = v;
    } else {
        dq->buf[--dq->head] = v;
    }
}

static int dq_pop_front(Deque *dq)
{
    return dq->buf[dq->head++];
}

static int dq_peek_front(const Deque *dq)
{
    return dq->buf[dq->head];
}

/* -------------------------------------------------------------------------
 * Dynamic int array (for active_rids)
 * ------------------------------------------------------------------------- */

typedef struct {
    int  *buf;
    int   len, cap;
} IArr;

static void ia_init(IArr *a, int cap)
{
    a->buf = (int *)SIM_MALLOC(cap * sizeof(int));
    a->len = 0;
    a->cap = cap;
}

static void ia_free(IArr *a) { SIM_FREE(a->buf); a->buf = NULL; }

static void ia_push(IArr *a, int v)
{
    if (a->len == a->cap) {
        a->cap *= 2;
        a->buf = (int *)SIM_REALLOC(a->buf, a->cap * sizeof(int));
    }
    a->buf[a->len++] = v;
}

static void ia_remove(IArr *a, int v)
{
    for (int i = 0; i < a->len; i++) {
        if (a->buf[i] == v) {
            a->buf[i] = a->buf[--a->len];
            return;
        }
    }
}

/* -------------------------------------------------------------------------
 * Per-request simulation state (separate from input CReq to keep input clean)
 * ------------------------------------------------------------------------- */

typedef struct {
    int    sim_decode_count;
    int    active_kv;        /* prompt_len + sim_decode_count; 0 if not active */
    int    anticipated;      /* 1 = already recorded an anticipated event */
} SimState;

/* -------------------------------------------------------------------------
 * Core kernel — no Python objects touched, GIL not required
 *
 * Implements rebuild_from_real_state:
 *   1. Sort all requests by arrival_ts
 *   2. Run isolated scheduler steps until all have anticipated events or
 *      max_steps reached
 *   3. Remaining waiting requests without events get prefill(inf)
 *
 * ANTICIPATED EVENT SEMANTICS:
 * The anticipated event is always the NEXT event after the real-world state,
 * with a timestamp derived solely from the isolation sim's clock:
 *
 *   - real_dc=5, sim reached decode 8 at sim-time T:
 *       anticipated = decode 6 at (T - 2*step_dur)   [back-calculated]
 *   - real_dc=5, sim stopped at decode 2 at sim-time T:
 *       anticipated = decode 6 at (T + 4*step_dur)   [extrapolated forward]
 *   - request still queued in sim:
 *       anticipated = prefill with end_ts=inf
 *
 * Timestamps may be in the past relative to wall clock. This is correct.
 * Do NOT floor to wall clock here — callers interpret timestamps.
 * ------------------------------------------------------------------------- */

static void run_rebuild_kernel(
        CReq *reqs, int n,
        int max_kv_tokens,   /* -1 = unlimited */
        int fairinf_n,
        double until_timestamp)  /* stop when current_time >= this; -1 = no limit */
{
    if (n <= 0) return;

    /* --- sort by arrival_ts (insertion sort — n is small) --- */
    int *order = (int *)SIM_MALLOC(n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int key = order[i];
        double kts = reqs[key].arrival_ts;
        int j = i - 1;
        while (j >= 0 && reqs[order[j]].arrival_ts > kts) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    /* --- init per-req sim state --- */
    SimState *ss = (SimState *)SIM_MALLOC(n * sizeof(SimState));
    for (int i = 0; i < n; i++) {
        ss[i].sim_decode_count = 0;
        ss[i].active_kv        = 0;
        ss[i].anticipated      = 0;
        reqs[i].ant_type       = -1;  /* none */
    }

    /* --- waiting deque (indices in order[]) order) --- */
    Deque waiting;
    dq_init(&waiting, n + 4);
    for (int i = 0; i < n; i++)
        dq_push_back(&waiting, order[i]);

    /* --- active list --- */
    IArr active;
    ia_init(&active, 8);

    /* rid -> index hash map for O(1) lookup */
    HashMap hm;
    hm_init(&hm, n);
    for (int i = 0; i < n; i++)
        hm_put(&hm, reqs[i].rid, i);

    double current_time = reqs[order[0]].arrival_ts;
    int active_kv_total = 0;
    int anticipated_count = 0;

    int max_steps = n * 4;
    if (max_steps < 200) max_steps = 200;
    if (max_steps > 2000) max_steps = 2000;

    /* ------------------------------------------------------------------ */
    for (int step = 0; step < max_steps; step++) {

        if (anticipated_count >= n) break;
        if (until_timestamp >= 0.0 && current_time >= until_timestamp) break;

        /* --- peek at waiting queue: any request ready? --- */
        int any_ready = 0;
        double next_arrival = -1.0;  /* -1 = none */

        if (!dq_empty(&waiting)) {
            int fi = dq_peek_front(&waiting);
            double fts = reqs[fi].arrival_ts;
            if (fts <= current_time) {
                any_ready = 1;
            } else {
                next_arrival = fts;
            }
        }

        /* ============================================================
         * PREFILL branch
         * ============================================================ */
        if (any_ready) {
            /* Build prefill batch — heap-allocated to handle up to n requests */
            int *batch = (int *)SIM_MALLOC(n * sizeof(int));
            int blen = 0;
            int batch_tokens = 0;

            /* iterate waiting from front */
            int *tmp = (int *)SIM_MALLOC(n * sizeof(int));
            int tlen = 0;
            while (!dq_empty(&waiting)) {
                int idx = dq_pop_front(&waiting);
                double ats = reqs[idx].arrival_ts;
                if (ats > current_time) {
                    /* put it back — nothing further can be ready */
                    tmp[tlen++] = idx;
                    break;
                }
                int pt = reqs[idx].prompt_len;
                if (max_kv_tokens >= 0 &&
                    active_kv_total + batch_tokens + pt > max_kv_tokens) {
                    tmp[tlen++] = idx;
                    break;
                }
                batch[blen++] = idx;
                batch_tokens += pt;
            }
            /* drain remaining waiting items to tmp so we can rebuild deque */
            while (!dq_empty(&waiting))
                tmp[tlen++] = dq_pop_front(&waiting);
            /* rebuild deque: batch items removed, rest preserved in order */
            for (int i = 0; i < tlen; i++)
                dq_push_back(&waiting, tmp[i]);

            if (blen == 0) {
                /* Nothing fit — treat like no active_rids situation */
                SIM_FREE(batch);
                SIM_FREE(tmp);
                goto check_active;
            }

            /* Compute prefill duration */
            int sum_p = 0, max_p = 0;
            for (int i = 0; i < blen; i++) {
                int pt = reqs[batch[i]].prompt_len;
                sum_p += pt;
                if (pt > max_p) max_p = pt;
            }
            double dur = isolated_prefill_time_estimation(
                (double)sum_p, (double)max_p, (double)blen, (double)fairinf_n);
            current_time += dur;

            for (int i = 0; i < blen; i++) {
                int idx = batch[i];
                int pt  = reqs[idx].prompt_len;
                ia_push(&active, idx);
                ss[idx].active_kv    = pt + 1;
                active_kv_total     += pt + 1;
                /* Record anticipated prefill event if not yet prefilled in reality */
                if (!ss[idx].anticipated && !reqs[idx].prefill_done) {
                    reqs[idx].ant_type       = 0;   /* prefill */
                    reqs[idx].ant_end_ts     = current_time;
                    reqs[idx].ant_completion = 0;
                    ss[idx].anticipated      = 1;
                    anticipated_count++;
                }
            }
            SIM_FREE(batch);
            SIM_FREE(tmp);
            continue;  /* next step */
        }

        check_active:
        if (active.len > 0) {
            /* --------------------------------------------------------
             * Try retraction
             * -------------------------------------------------------- */
            int did_retract = 0;
            if (max_kv_tokens >= 0 &&
                active_kv_total + active.len > max_kv_tokens &&
                active.len > 1)
            {
                /* Evict the request with the lowest real_decode_count (fewest real completions) */
                int evict_pos = 0;
                for (int i = 1; i < active.len; i++) {
                    if (reqs[active.buf[i]].real_decode_count <
                        reqs[active.buf[evict_pos]].real_decode_count)
                        evict_pos = i;
                }
                int evict_idx = active.buf[evict_pos];
                /* Remove from active */
                active.buf[evict_pos] = active.buf[--active.len];
                /* Push to front of waiting */
                dq_push_front(&waiting, evict_idx);

                active_kv_total -= ss[evict_idx].active_kv;
                ss[evict_idx].active_kv        = 0;
                ss[evict_idx].sim_decode_count = 0;

                /* Only clear anticipated if it wasn't a prefill event */
                if (ss[evict_idx].anticipated &&
                    reqs[evict_idx].ant_type != 0 /* not prefill */) {
                    ss[evict_idx].anticipated = 0;
                    anticipated_count--;
                    reqs[evict_idx].ant_type = -1;
                }
                current_time += RETRACTION_PENALTY_SECONDS;
                did_retract = 1;
                (void)did_retract;
            }

            if (active.len == 0) continue;

            /* --------------------------------------------------------
             * Decode step
             * -------------------------------------------------------- */
            int total_tok = 0, max_tok = 0;
            for (int i = 0; i < active.len; i++) {
                int kv = ss[active.buf[i]].active_kv;
                total_tok += kv;
                if (kv > max_tok) max_tok = kv;
            }
            if (total_tok == 0) continue;

            double dur = isolated_decode_time_estimation(
                (double)total_tok, (double)max_tok,
                (double)active.len, (double)fairinf_n);

            /* Check arrival interrupt */
            if (next_arrival > 0.0 && current_time + dur > next_arrival) {
                current_time = next_arrival;
                continue;
            }

            current_time += dur;

            /* Advance each active request */
            int new_active_len = 0;
            for (int i = 0; i < active.len; i++) {
                int idx = active.buf[i];
                ss[idx].sim_decode_count++;
                ss[idx].active_kv++;
                active_kv_total++;

                if (!ss[idx].anticipated) {
                    int real_dc = reqs[idx].real_decode_count;
                    if (ss[idx].sim_decode_count > real_dc) {
                        /* anticipated = real+1 at the sim-time when that decode
                         * would have completed. Back-calculate from current_time:
                         * sim is now at sim_dc, we want the time of (real_dc+1).
                         * steps_past = sim_dc - (real_dc+1); each took dur seconds. */
                        int steps_past = ss[idx].sim_decode_count - (real_dc + 1);
                        reqs[idx].ant_type       = 1;   /* decode */
                        reqs[idx].ant_end_ts     = current_time - steps_past * dur;
                        reqs[idx].ant_completion = real_dc + 1;
                        ss[idx].anticipated      = 1;
                        anticipated_count++;
                    }
                }

                /* Keep in active if: not (anticipated && is_complete) */
                int drop = ss[idx].anticipated && reqs[idx].is_complete;
                if (!drop) {
                    active.buf[new_active_len++] = idx;
                } else {
                    active_kv_total -= ss[idx].active_kv;
                    ss[idx].active_kv = 0;
                }
            }
            active.len = new_active_len;

            continue;
        }

        /* Nothing active, nothing ready, no next arrival → done */
        if (next_arrival < 0.0) break;
        current_time = next_arrival;
    }
    /* ------------------------------------------------------------------ */

    /* Remaining waiting requests without an anticipated event → prefill(inf) */
    while (!dq_empty(&waiting)) {
        int idx = dq_pop_front(&waiting);
        if (!ss[idx].anticipated) {
            reqs[idx].ant_type       = 0;       /* prefill */
            reqs[idx].ant_end_ts     = Py_HUGE_VAL;  /* inf */
            reqs[idx].ant_completion = 0;
            ss[idx].anticipated      = 1;
        }
    }

    /* Active requests without an anticipated event: sim stopped (max_steps or
     * until_timestamp) before sim_decode_count exceeded real_decode_count.
     * Extrapolate forward: the sim is at sim_dc at current_time, and needs
     * (real_dc+1 - sim_dc) more steps to reach the anticipated decode.
     * INVARIANT: anticipated = logical real+1 at pure sim-time. Extrapolate from
     * sim's current position — do not floor to wall clock. */
    {
        for (int i = 0; i < active.len; i++) {
            int idx = active.buf[i];
            if (!ss[idx].anticipated) {
                int sim_dc  = ss[idx].sim_decode_count;
                int real_dc = reqs[idx].real_decode_count;
                if (sim_dc >= real_dc) continue;  /* over-served: no deadline */
                int next_n = real_dc + 1;
                int ctx    = reqs[idx].prompt_len + next_n;
                double dur = isolated_decode_time_estimation(
                    (double)ctx, (double)ctx, 1.0, (double)fairinf_n);
                int steps_remaining = next_n - sim_dc;
                reqs[idx].ant_type       = 1;   /* decode */
                reqs[idx].ant_end_ts     = current_time + steps_remaining * dur;
                reqs[idx].ant_completion = next_n;
                ss[idx].anticipated      = 1;
            }
        }
    }

    SIM_FREE(order);
    SIM_FREE(ss);
    dq_free(&waiting);
    ia_free(&active);
    hm_free(&hm);
}

/* -------------------------------------------------------------------------
 * Python-callable: rebuild_kernel(rids, arrivals, prompt_lens,
 *                                 decode_counts, prefill_dones, is_completes,
 *                                 max_kv, fairinf_n)
 * Returns List[Tuple[str, int, float, int]]
 * ------------------------------------------------------------------------- */

static PyObject *
py_rebuild_kernel(PyObject *self, PyObject *args)
{
    PyObject *py_rids, *py_arrivals, *py_prompt_lens;
    PyObject *py_decode_counts, *py_prefill_dones, *py_is_completes;
    int max_kv, fairinf_n;

    if (!PyArg_ParseTuple(args, "OOOOOOii",
            &py_rids, &py_arrivals, &py_prompt_lens,
            &py_decode_counts, &py_prefill_dones, &py_is_completes,
            &max_kv, &fairinf_n))
        return NULL;

    Py_ssize_t n = PyList_Size(py_rids);
    if (n < 0) return NULL;
    if (n == 0) return PyList_New(0);

    /* --- Extract primitive arrays from Python lists (GIL held) --- */
    CReq *reqs = (CReq *)PyMem_Malloc(n * sizeof(CReq));
    if (!reqs) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *rid_obj = PyList_GET_ITEM(py_rids, i);
        const char *rid_s = PyUnicode_AsUTF8(rid_obj);
        if (!rid_s) { PyMem_Free(reqs); return NULL; }
        strncpy(reqs[i].rid, rid_s, RID_MAX - 1);
        reqs[i].rid[RID_MAX - 1] = '\0';

        reqs[i].arrival_ts        = PyFloat_AsDouble(PyList_GET_ITEM(py_arrivals, i));
        reqs[i].prompt_len        = (int)PyLong_AsLong(PyList_GET_ITEM(py_prompt_lens, i));
        reqs[i].real_decode_count = (int)PyLong_AsLong(PyList_GET_ITEM(py_decode_counts, i));
        reqs[i].prefill_done      = (int)PyLong_AsLong(PyList_GET_ITEM(py_prefill_dones, i));
        reqs[i].is_complete       = (int)PyLong_AsLong(PyList_GET_ITEM(py_is_completes, i));

        if (PyErr_Occurred()) { PyMem_Free(reqs); return NULL; }
    }

    /* --- Release GIL for the simulation --- */
    Py_BEGIN_ALLOW_THREADS
    run_rebuild_kernel(reqs, (int)n, max_kv, fairinf_n, -1.0);
    Py_END_ALLOW_THREADS

    /* --- Build result list (GIL re-acquired) --- */
    PyObject *result = PyList_New(0);
    if (!result) { PyMem_Free(reqs); return NULL; }

    for (Py_ssize_t i = 0; i < n; i++) {
        if (reqs[i].ant_type < 0) continue;  /* no event */
        if (reqs[i].is_complete) continue;   /* completed requests have no deadlines */
        PyObject *tup = Py_BuildValue(
            "(sidi)",
            reqs[i].rid,
            (int)reqs[i].ant_type,
            reqs[i].ant_end_ts,
            (int)reqs[i].ant_completion
        );
        if (!tup) { Py_DECREF(result); PyMem_Free(reqs); return NULL; }
        if (PyList_Append(result, tup) < 0) {
            Py_DECREF(tup); Py_DECREF(result); PyMem_Free(reqs); return NULL;
        }
        Py_DECREF(tup);
    }

    PyMem_Free(reqs);
    return result;
}

/* -------------------------------------------------------------------------
 * C-backed rebuild_from_real_state replacement.
 *
 * This is called as a bound method on a UserTimeline Python object.
 * It reads request_timelines / requests_real from the Python object,
 * calls run_rebuild_kernel (GIL released), then writes next_anticipated_event
 * back to each TrackedRequest.
 *
 * Signature matches UserTimeline.rebuild_from_real_state:
 *   def rebuild_from_real_state(self, _unused=None,
 *                               until_timestamp=None,
 *                               timing_breakdown=None) -> None
 * ------------------------------------------------------------------------- */

static PyObject *
c_rebuild_from_real_state(PyObject *self_capsule, PyObject *args, PyObject *kwargs)
{
    /* self_capsule is a PyCapsule holding a pointer to the UserTimeline PyObject */
    PyObject *ut;  /* UserTimeline Python object */
    if (PyCapsule_CheckExact(self_capsule)) {
        ut = (PyObject *)PyCapsule_GetPointer(self_capsule, "UserTimeline");
    } else {
        ut = self_capsule;  /* called directly with the UserTimeline as self */
    }

    /* Parse until_timestamp from kwargs if provided */
    double until_ts = -1.0;
    if (kwargs) {
        PyObject *py_until = PyDict_GetItemString(kwargs, "until_timestamp");
        if (py_until && py_until != Py_None)
            until_ts = PyFloat_AsDouble(py_until);
    }
    (void)args;

    /* --- Read request_timelines dict --- */
    PyObject *request_timelines = PyObject_GetAttrString(ut, "request_timelines");
    if (!request_timelines) return NULL;

    Py_ssize_t n = PyDict_Size(request_timelines);
    if (n == 0) {
        Py_DECREF(request_timelines);
        Py_RETURN_NONE;
    }

    /* --- Read max_kv_tokens and fairinf_n --- */
    PyObject *py_max_kv = PyObject_GetAttrString(ut, "max_kv_tokens");
    int max_kv = -1;
    if (py_max_kv && py_max_kv != Py_None)
        max_kv = (int)PyLong_AsLong(py_max_kv);
    Py_XDECREF(py_max_kv);

    PyObject *py_fn = PyObject_GetAttrString(ut, "fairinf_n");
    int fairinf_n = 1;
    if (py_fn) fairinf_n = (int)PyLong_AsLong(py_fn);
    Py_XDECREF(py_fn);

    if (PyErr_Occurred()) { Py_DECREF(request_timelines); return NULL; }

    /* --- Read requests_real dict --- */
    PyObject *requests_real = PyObject_GetAttrString(ut, "requests_real");
    if (!requests_real) { Py_DECREF(request_timelines); return NULL; }

    /* --- Extract C arrays from Python dicts --- */
    CReq *reqs = (CReq *)PyMem_Malloc(n * sizeof(CReq));
    if (!reqs) {
        Py_DECREF(request_timelines);
        Py_DECREF(requests_real);
        return PyErr_NoMemory();
    }

    /* rid -> index, so we can write back results */
    /* We'll store the Python rid strings and TrackedRequest objects for write-back */
    PyObject **tracked_objs = (PyObject **)PyMem_Malloc(n * sizeof(PyObject *));
    if (!tracked_objs) {
        PyMem_Free(reqs);
        Py_DECREF(request_timelines);
        Py_DECREF(requests_real);
        return PyErr_NoMemory();
    }

    PyObject *rid_key, *tracked_val;
    Py_ssize_t pos = 0;
    int idx = 0;

    while (PyDict_Next(request_timelines, &pos, &rid_key, &tracked_val)) {
        if (idx >= n) break;

        const char *rid_s = PyUnicode_AsUTF8(rid_key);
        if (!rid_s) goto err;
        strncpy(reqs[idx].rid, rid_s, RID_MAX - 1);
        reqs[idx].rid[RID_MAX - 1] = '\0';

        /* arrival_ts: tracked.timeline.history[0].end_timestamp */
        PyObject *timeline_obj = PyObject_GetAttrString(tracked_val, "timeline");
        if (!timeline_obj) goto err;
        PyObject *history = PyObject_GetAttrString(timeline_obj, "history");
        Py_DECREF(timeline_obj);
        if (!history) goto err;

        double arrival_ts = 0.0;
        if (PyList_Size(history) > 0) {
            PyObject *first_ev = PyList_GET_ITEM(history, 0);
            PyObject *ets = PyObject_GetAttrString(first_ev, "end_timestamp");
            if (ets) { arrival_ts = PyFloat_AsDouble(ets); Py_DECREF(ets); }
        }
        Py_DECREF(history);

        /* prompt_len: len(tracked.req.origin_input_ids) */
        PyObject *req_obj = PyObject_GetAttrString(tracked_val, "req");
        if (!req_obj) goto err;
        PyObject *origin = PyObject_GetAttrString(req_obj, "origin_input_ids");
        Py_DECREF(req_obj);
        if (!origin) goto err;
        int prompt_len = (int)PySequence_Size(origin);
        Py_DECREF(origin);

        /* requests_real entry */
        int real_dc = 0, prefill_done = 0, is_complete = 0;
        PyObject *status = PyDict_GetItem(requests_real, rid_key);
        if (status && status != Py_None) {
            PyObject *v;
            v = PyObject_GetAttrString(status, "decode_count");
            if (v) { real_dc = (int)PyLong_AsLong(v); Py_DECREF(v); }
            v = PyObject_GetAttrString(status, "prefill_done");
            if (v) { prefill_done = PyObject_IsTrue(v); Py_DECREF(v); }
            v = PyObject_GetAttrString(status, "is_complete");
            if (v) { is_complete = PyObject_IsTrue(v); Py_DECREF(v); }
        }
        if (PyErr_Occurred()) goto err;

        reqs[idx].arrival_ts        = arrival_ts;
        reqs[idx].prompt_len        = prompt_len;
        reqs[idx].real_decode_count = real_dc;
        reqs[idx].prefill_done      = prefill_done;
        reqs[idx].is_complete       = is_complete;
        reqs[idx].ant_type          = -1;

        /* Also clear next_anticipated_event on the Python object now
           (mirrors the Python: tracked.timeline.next_anticipated_event = None) */
        PyObject *tl2 = PyObject_GetAttrString(tracked_val, "timeline");
        if (!tl2) goto err;
        if (PyObject_SetAttrString(tl2, "next_anticipated_event", Py_None) < 0) {
            Py_DECREF(tl2); goto err;
        }
        Py_DECREF(tl2);

        tracked_objs[idx] = tracked_val;  /* borrowed ref, dict keeps it alive */
        idx++;
        continue;

    err:
        PyMem_Free(reqs);
        PyMem_Free(tracked_objs);
        Py_DECREF(request_timelines);
        Py_DECREF(requests_real);
        return NULL;
    }
    n = idx;  /* actual count (dict may have had issues) */

    Py_DECREF(requests_real);

    /* --- Release GIL and run the kernel --- */
    Py_BEGIN_ALLOW_THREADS
    run_rebuild_kernel(reqs, (int)n, max_kv, fairinf_n, until_ts);
    Py_END_ALLOW_THREADS

    /* --- Write back anticipated events to Python objects --- */
    /* We need the Python event classes. Import lazily. */
    static PyObject *RequestPrefillEvent_cls = NULL;
    static PyObject *RequestDecodeEvent_cls  = NULL;

    if (RequestPrefillEvent_cls == NULL) {
        PyObject *mod = PyImport_ImportModule(
            "sglang.srt.delta_fairness.doc_policy_simulator");
        if (!mod) goto writeback_err;
        RequestPrefillEvent_cls = PyObject_GetAttrString(mod, "RequestPrefillEvent");
        RequestDecodeEvent_cls  = PyObject_GetAttrString(mod, "RequestDecodeEvent");
        Py_DECREF(mod);
        if (!RequestPrefillEvent_cls || !RequestDecodeEvent_cls) goto writeback_err;
    }

    for (int i = 0; i < n; i++) {
        if (reqs[i].ant_type < 0) continue;

        PyObject *event = NULL;
        if (reqs[i].ant_type == 0) {
            /* RequestPrefillEvent(req_id=rid, duration=0.0, end_timestamp=ets) */
            double ets = reqs[i].ant_end_ts;
            /* Convert C inf to Python float inf */
            PyObject *ets_py = (isinf(ets) || ets >= 1e300)
                ? PyFloat_FromDouble(Py_HUGE_VAL)
                : PyFloat_FromDouble(ets);
            event = PyObject_CallFunction(RequestPrefillEvent_cls, "sdd",
                reqs[i].rid, 0.0, PyFloat_AS_DOUBLE(ets_py));
            Py_DECREF(ets_py);
        } else {
            /* RequestDecodeEvent(req_id=rid, duration=dur, end_timestamp=ets,
                                  completion_number=cn) */
            event = PyObject_CallFunction(RequestDecodeEvent_cls, "sddi",
                reqs[i].rid, 0.0, reqs[i].ant_end_ts,
                reqs[i].ant_completion);
        }
        if (!event) goto writeback_err;

        /* tracked.timeline.next_anticipated_event = event */
        PyObject *tl = PyObject_GetAttrString(tracked_objs[i], "timeline");
        if (!tl) { Py_DECREF(event); goto writeback_err; }
        int rc = PyObject_SetAttrString(tl, "next_anticipated_event", event);
        Py_DECREF(tl);
        Py_DECREF(event);
        if (rc < 0) goto writeback_err;
    }

    PyMem_Free(reqs);
    PyMem_Free(tracked_objs);
    Py_DECREF(request_timelines);
    Py_RETURN_NONE;

writeback_err:
    PyMem_Free(reqs);
    PyMem_Free(tracked_objs);
    Py_DECREF(request_timelines);
    return NULL;
}

/* -------------------------------------------------------------------------
 * patch_simulator(sim) — replaces UserTimeline.rebuild_from_real_state with
 * the C-backed implementation.
 *
 * We expose c_rebuild_from_real_state as a module-level C function
 * "_c_rebuild_from_real_state", then exec a tiny Python snippet that wraps it
 * as a proper Python function and sets it on the UserTimeline class.
 * This avoids all descriptor-protocol complexity with PyCFunction_New.
 * ------------------------------------------------------------------------- */

static PyObject *
py_c_rebuild_from_real_state_exposed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    /* Called as _fairinf_sim._c_rebuild_from_real_state(ut, ...) */
    PyObject *ut = NULL;
    if (!PyArg_ParseTuple(args, "O|Ozi",
            &ut, NULL, NULL, NULL)) {
        /* Accept (self,) or (self, _unused) or (self, _unused, until_ts, breakdown) */
        PyErr_Clear();
        if (PyTuple_Size(args) < 1) {
            PyErr_SetString(PyExc_TypeError, "_c_rebuild_from_real_state needs UserTimeline");
            return NULL;
        }
        ut = PyTuple_GET_ITEM(args, 0);
    }
    if (!ut || ut == Py_None) {
        PyErr_SetString(PyExc_TypeError, "_c_rebuild_from_real_state: invalid self");
        return NULL;
    }
    return c_rebuild_from_real_state(ut, args, kwargs);
}

static PyMethodDef c_rebuild_exposed_def = {
    "_c_rebuild_from_real_state",
    (PyCFunction)py_c_rebuild_from_real_state_exposed,
    METH_VARARGS | METH_KEYWORDS,
    NULL
};

static PyObject *
py_patch_simulator(PyObject *self_mod, PyObject *args)
{
    PyObject *sim;
    if (!PyArg_ParseTuple(args, "O", &sim)) return NULL;

    /* Get the _fairinf_sim module object so we can add the helper to it */
    PyObject *this_mod = PyImport_ImportModule("sglang.srt.delta_fairness._fairinf_sim");
    if (!this_mod) return NULL;

    /* Add _c_rebuild_from_real_state as a module function if not already there */
    if (!PyObject_HasAttrString(this_mod, "_c_rebuild_from_real_state")) {
        PyObject *cfunc = PyCFunction_New(&c_rebuild_exposed_def, NULL);
        if (!cfunc) { Py_DECREF(this_mod); return NULL; }
        int rc = PyObject_SetAttrString(this_mod, "_c_rebuild_from_real_state", cfunc);
        Py_DECREF(cfunc);
        if (rc < 0) { Py_DECREF(this_mod); return NULL; }
    }

    Py_DECREF(this_mod);

    /* Exec a Python snippet that installs the method on UserTimeline.
     * Using exec is the simplest way to get a proper Python function that
     * Python's descriptor protocol will bind correctly as an instance method. */
    const char *patch_code =
        "import sglang.srt.delta_fairness._fairinf_sim as _fsim\n"
        "import sglang.srt.delta_fairness.doc_policy_simulator as _dsim\n"
        "def _c_rebuild(self, _unused=None, until_timestamp=None, timing_breakdown=None):\n"
        "    return _fsim._c_rebuild_from_real_state(self)\n"
        "_dsim.UserTimeline.rebuild_from_real_state = _c_rebuild\n";

    PyObject *result = PyRun_String(patch_code, Py_file_input,
                                    PyEval_GetBuiltins(), PyEval_GetBuiltins());
    if (!result) return NULL;
    Py_DECREF(result);

    Py_RETURN_NONE;
}

/* -------------------------------------------------------------------------
 * Module method table
 * ------------------------------------------------------------------------- */

static PyMethodDef FairinfSimMethods[] = {
    {"rebuild_kernel", py_rebuild_kernel,
     METH_VARARGS,
     "rebuild_kernel(rids, arrivals, prompt_lens, decode_counts, "
     "prefill_dones, is_completes, max_kv, fairinf_n) -> list of (rid, type, ts, cn)"},
    {"patch_simulator", py_patch_simulator,
     METH_VARARGS,
     "patch_simulator(sim) -> None\n"
     "Replace UserTimeline.rebuild_from_real_state with the C kernel on sim."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fairinf_sim_module = {
    PyModuleDef_HEAD_INIT,
    "_fairinf_sim",
    "C-accelerated fairinf simulation kernel (GIL-free rebuild_from_real_state)",
    -1,
    FairinfSimMethods,
};

PyMODINIT_FUNC
PyInit__fairinf_sim(void)
{
    return PyModule_Create(&fairinf_sim_module);
}
