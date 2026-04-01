/*
 * _fairinf_worker.c — CPython C extension providing a stateful CSimulator type
 * for the fairinf worker thread.
 *
 * This replaces AlternateHistorySimulator for the worker thread path, providing
 * GIL-free simulation kernels and direct C-level state tracking.
 *
 * Build:
 *   cd delta_fairness/
 *   python3.12 setup_fairinf_worker.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Use plain malloc/free for GIL-free sections */
#define W_MALLOC  malloc
#define W_FREE    free
#define W_REALLOC realloc

/* -------------------------------------------------------------------------
 * Time constants — MUST match time_estimation.py exactly
 * -------------------------------------------------------------------------
 * DECODE_CONST_PER_SCHEDULING_PASS_OVERHEAD = 0.200
 * CONST_INTERVAL_DECODE = 0.200 / 10 = 0.020
 * TBT_DELTA = 0.080
 * MAX_POOLED_DECODE_LATENCY = pooled_decode_time_estimation(328784, 8192, 256, 1)
 *   = 0.020 + max(5e-3, 1.18629080e-02 + 6.81862494e-08*328784
 *                        + 2.62872519e-07*8192 + 5.65921863e-05*256)
 *   = 0.020 + 0.05092250719117761 = 0.07092250719117761
 */
#define W_DECODE_CONST_OVERHEAD   0.020
#define W_TBT_DELTA               0.080
/* MAX_POOLED_DECODE_LATENCY as used in isolated_decode_time_estimation comparison:
 * The Python function compares the inner linear term against MAX_POOLED_DECODE_LATENCY,
 * which is the FULL return value of pooled_decode_time_estimation (includes CONST_INTERVAL_DECODE).
 * So in isolated_decode:
 *   inner = CONST_INTERVAL_DECODE + max(MAX_POOLED_DECODE_LATENCY, linear_term) + TBT_DELTA
 * The linear_term is compared against MAX_POOLED_DECODE_LATENCY directly.
 */
#define W_MAX_POOLED_DECODE_LATENCY  0.07092250719117761
#define W_PREFILL_PIECEWISE_BOUND    4064.0
#define W_RETRACTION_PENALTY         0.030

#define W_RID_MAX 128
#define W_UID_MAX  64
#define HT_EMPTY  (-1)

/* -------------------------------------------------------------------------
 * Time estimation functions — exact translation of time_estimation.py
 * -------------------------------------------------------------------------*/

static double w_isolated_prefill_time_estimation(
        double total_batch_sum, double max_token_size,
        double batch_length, double n)
{
    double v;
    if (total_batch_sum <= W_PREFILL_PIECEWISE_BOUND) {
        v =  9.15608285e-03
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

static double w_isolated_decode_time_estimation(
        double total_batch_sum, double max_token_size,
        double batch_length, double n)
{
    double inner = 1.18629080e-02
        + 6.81862494e-08 * n * total_batch_sum
        + 2.62872519e-07 * n * max_token_size
        + 5.65921863e-05 * n * batch_length;
    double v = inner > W_MAX_POOLED_DECODE_LATENCY ? inner : W_MAX_POOLED_DECODE_LATENCY;
    return W_DECODE_CONST_OVERHEAD + v + W_TBT_DELTA;
}

/* -------------------------------------------------------------------------
 * Open-addressing hash map: string key -> int value
 * -------------------------------------------------------------------------*/

typedef struct {
    char key[W_RID_MAX];
    int  val;
} w_HTSlot;

typedef struct {
    w_HTSlot *slots;
    int       cap;
    int       mask;
} w_HashMap;

static unsigned int w_hm_hash(const char *s)
{
    unsigned int h = 2166136261u;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

static void w_hm_init(w_HashMap *hm, int n)
{
    int cap = 4;
    while (cap < n * 2) cap <<= 1;
    hm->slots = (w_HTSlot *)W_MALLOC(cap * sizeof(w_HTSlot));
    hm->cap   = cap;
    hm->mask  = cap - 1;
    for (int i = 0; i < cap; i++) {
        hm->slots[i].key[0] = '\0';
        hm->slots[i].val    = HT_EMPTY;
    }
}

static void w_hm_free(w_HashMap *hm)
{
    W_FREE(hm->slots);
    hm->slots = NULL;
    hm->cap = 0;
    hm->mask = 0;
}

static void w_hm_grow(w_HashMap *hm)
{
    int old_cap = hm->cap;
    w_HTSlot *old_slots = hm->slots;
    int new_cap = old_cap * 2;
    hm->slots = (w_HTSlot *)W_MALLOC(new_cap * sizeof(w_HTSlot));
    hm->cap   = new_cap;
    hm->mask  = new_cap - 1;
    for (int i = 0; i < new_cap; i++) {
        hm->slots[i].key[0] = '\0';
        hm->slots[i].val    = HT_EMPTY;
    }
    for (int i = 0; i < old_cap; i++) {
        if (old_slots[i].val != HT_EMPTY) {
            unsigned int h = w_hm_hash(old_slots[i].key) & hm->mask;
            while (hm->slots[h].val != HT_EMPTY)
                h = (h + 1) & hm->mask;
            hm->slots[h] = old_slots[i];
        }
    }
    W_FREE(old_slots);
}

/* Count of used slots (approx — we track separately for safety) */
static void w_hm_put(w_HashMap *hm, const char *key, int val)
{
    /* grow if >50% full */
    /* We just grow defensively - check load before insert */
    unsigned int h = w_hm_hash(key) & hm->mask;
    while (hm->slots[h].val != HT_EMPTY &&
           strcmp(hm->slots[h].key, key) != 0) {
        h = (h + 1) & hm->mask;
    }
    strncpy(hm->slots[h].key, key, W_RID_MAX - 1);
    hm->slots[h].key[W_RID_MAX - 1] = '\0';
    hm->slots[h].val = val;
}

static int w_hm_get(const w_HashMap *hm, const char *key)
{
    if (hm->slots == NULL) return HT_EMPTY;
    unsigned int h = w_hm_hash(key) & hm->mask;
    for (;;) {
        if (hm->slots[h].val == HT_EMPTY) return HT_EMPTY;
        if (strcmp(hm->slots[h].key, key) == 0) return hm->slots[h].val;
        h = (h + 1) & hm->mask;
    }
}

static void w_hm_delete(w_HashMap *hm, const char *key)
{
    if (hm->slots == NULL) return;
    unsigned int h = w_hm_hash(key) & hm->mask;
    for (;;) {
        if (hm->slots[h].val == HT_EMPTY) return;
        if (strcmp(hm->slots[h].key, key) == 0) {
            /* Robin Hood deletion: shift subsequent slots */
            hm->slots[h].key[0] = '\0';
            hm->slots[h].val = HT_EMPTY;
            unsigned int j = (h + 1) & hm->mask;
            while (hm->slots[j].val != HT_EMPTY) {
                w_HTSlot s = hm->slots[j];
                hm->slots[j].key[0] = '\0';
                hm->slots[j].val = HT_EMPTY;
                w_hm_put(hm, s.key, s.val);
                j = (j + 1) & hm->mask;
            }
            return;
        }
        h = (h + 1) & hm->mask;
    }
}

/* -------------------------------------------------------------------------
 * Rebuild kernel data structures (w_ prefixed copies from _fairinf_sim.c)
 * -------------------------------------------------------------------------*/

typedef struct {
    char     rid[W_RID_MAX];
    double   arrival_ts;
    int      prompt_len;
    int      real_decode_count;
    int      prefill_done;
    int      is_complete;
    /* output */
    int      ant_type;   /* -1=none, 0=prefill, 1=decode */
    double   ant_end_ts;
    int      ant_completion;
} w_CReq;

typedef struct {
    int    *buf;
    int     head, tail, cap;
} w_Deque;

static void w_dq_init(w_Deque *dq, int cap)
{
    dq->buf  = (int *)W_MALLOC(cap * sizeof(int));
    dq->head = 0;
    dq->tail = 0;
    dq->cap  = cap;
}
static void w_dq_free(w_Deque *dq) { W_FREE(dq->buf); dq->buf = NULL; }
static int  w_dq_len(const w_Deque *dq) { return dq->tail - dq->head; }
static int  w_dq_empty(const w_Deque *dq) { return dq->tail == dq->head; }

static void w_dq_grow(w_Deque *dq)
{
    int len = w_dq_len(dq);
    int new_cap = dq->cap * 2;
    int *nb = (int *)W_MALLOC(new_cap * sizeof(int));
    for (int i = 0; i < len; i++)
        nb[i] = dq->buf[dq->head + i];
    W_FREE(dq->buf);
    dq->buf  = nb;
    dq->head = 0;
    dq->tail = len;
    dq->cap  = new_cap;
}

static void w_dq_push_back(w_Deque *dq, int v)
{
    if (dq->tail == dq->cap) w_dq_grow(dq);
    dq->buf[dq->tail++] = v;
}

static void w_dq_push_front(w_Deque *dq, int v)
{
    if (dq->head == 0) {
        int len = w_dq_len(dq);
        if (dq->tail + 1 > dq->cap) w_dq_grow(dq);
        memmove(dq->buf + dq->head + 1, dq->buf + dq->head, len * sizeof(int));
        dq->tail = dq->head + len + 1;
        dq->buf[dq->head] = v;
    } else {
        dq->buf[--dq->head] = v;
    }
}

static int w_dq_pop_front(w_Deque *dq) { return dq->buf[dq->head++]; }
static int w_dq_peek_front(const w_Deque *dq) { return dq->buf[dq->head]; }

typedef struct {
    int  *buf;
    int   len, cap;
} w_IArr;

static void w_ia_init(w_IArr *a, int cap)
{
    a->buf = (int *)W_MALLOC(cap * sizeof(int));
    a->len = 0;
    a->cap = cap;
}
static void w_ia_free(w_IArr *a) { W_FREE(a->buf); a->buf = NULL; }
static void w_ia_push(w_IArr *a, int v)
{
    if (a->len == a->cap) {
        a->cap *= 2;
        a->buf = (int *)W_REALLOC(a->buf, a->cap * sizeof(int));
    }
    a->buf[a->len++] = v;
}

typedef struct {
    int    sim_decode_count;
    int    active_kv;
    int    anticipated;
} w_SimState;

/* Core rebuild kernel — no Python objects, GIL-free */
static void w_run_rebuild_kernel(
        w_CReq *reqs, int n,
        int max_kv_tokens,
        int fairinf_n)
{
    if (n <= 0) return;

    int *order = (int *)W_MALLOC(n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    /* insertion sort by arrival_ts */
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

    w_SimState *ss = (w_SimState *)W_MALLOC(n * sizeof(w_SimState));
    for (int i = 0; i < n; i++) {
        ss[i].sim_decode_count = 0;
        ss[i].active_kv        = 0;
        ss[i].anticipated      = 0;
        reqs[i].ant_type       = -1;
    }

    w_Deque waiting;
    w_dq_init(&waiting, n + 4);
    for (int i = 0; i < n; i++)
        w_dq_push_back(&waiting, order[i]);

    w_IArr active;
    w_ia_init(&active, 8);

    w_HashMap hm;
    w_hm_init(&hm, n);
    for (int i = 0; i < n; i++)
        w_hm_put(&hm, reqs[i].rid, i);

    double current_time = reqs[order[0]].arrival_ts;
    int active_kv_total = 0;
    int anticipated_count = 0;

    int max_steps = n * 4;
    if (max_steps < 200) max_steps = 200;
    if (max_steps > 2000) max_steps = 2000;

    for (int step = 0; step < max_steps; step++) {
        if (anticipated_count >= n) break;

        int any_ready = 0;
        double next_arrival = -1.0;

        if (!w_dq_empty(&waiting)) {
            int fi = w_dq_peek_front(&waiting);
            double fts = reqs[fi].arrival_ts;
            if (fts <= current_time) {
                any_ready = 1;
            } else {
                next_arrival = fts;
            }
        }

        if (any_ready) {
            int *batch = (int *)W_MALLOC(n * sizeof(int));
            int blen = 0;
            int batch_tokens = 0;
            int *tmp = (int *)W_MALLOC(n * sizeof(int));
            int tlen = 0;
            while (!w_dq_empty(&waiting)) {
                int idx = w_dq_pop_front(&waiting);
                double ats = reqs[idx].arrival_ts;
                if (ats > current_time) {
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
            while (!w_dq_empty(&waiting))
                tmp[tlen++] = w_dq_pop_front(&waiting);
            for (int i = 0; i < tlen; i++)
                w_dq_push_back(&waiting, tmp[i]);

            if (blen == 0) {
                W_FREE(batch);
                W_FREE(tmp);
                goto check_active;
            }

            int sum_p = 0, max_p = 0;
            for (int i = 0; i < blen; i++) {
                int pt = reqs[batch[i]].prompt_len;
                sum_p += pt;
                if (pt > max_p) max_p = pt;
            }
            double dur = w_isolated_prefill_time_estimation(
                (double)sum_p, (double)max_p, (double)blen, (double)fairinf_n);
            current_time += dur;

            for (int i = 0; i < blen; i++) {
                int idx = batch[i];
                int pt  = reqs[idx].prompt_len;
                w_ia_push(&active, idx);
                ss[idx].active_kv    = pt + 1;
                active_kv_total     += pt + 1;
                if (!ss[idx].anticipated && !reqs[idx].prefill_done) {
                    reqs[idx].ant_type       = 0;
                    reqs[idx].ant_end_ts     = current_time;
                    reqs[idx].ant_completion = 0;
                    ss[idx].anticipated      = 1;
                    anticipated_count++;
                }
            }
            W_FREE(batch);
            W_FREE(tmp);
            continue;
        }

        check_active:
        if (active.len > 0) {
            int did_retract = 0;
            if (max_kv_tokens >= 0 &&
                active_kv_total + active.len > max_kv_tokens &&
                active.len > 1)
            {
                /* Evict the request with the fewest real completions (real_decode_count) */
                int evict_pos = 0;
                for (int i = 1; i < active.len; i++) {
                    if (reqs[active.buf[i]].real_decode_count <
                        reqs[active.buf[evict_pos]].real_decode_count)
                        evict_pos = i;
                }
                int evict_idx = active.buf[evict_pos];
                active.buf[evict_pos] = active.buf[--active.len];
                w_dq_push_front(&waiting, evict_idx);

                active_kv_total -= ss[evict_idx].active_kv;
                ss[evict_idx].active_kv        = 0;
                ss[evict_idx].sim_decode_count = 0;
                /* Python _isolated_retract also resets prefill_done=False so that
                 * the retracted request can earn a new prefill anticipated event. */
                reqs[evict_idx].prefill_done = 0;

                if (ss[evict_idx].anticipated &&
                    reqs[evict_idx].ant_type != 0) {
                    ss[evict_idx].anticipated = 0;
                    anticipated_count--;
                    reqs[evict_idx].ant_type = -1;
                }
                current_time += W_RETRACTION_PENALTY;
                did_retract = 1;
            }

            /* After retraction, go back to the top of the loop so that the
             * newly-freed KV budget can be used to prefill waiting requests.
             * This matches Python's behaviour: _isolated_retract returns
             * immediately and the next _advance_scheduler_step call handles
             * the prefill step (not a decode in the same step). */
            if (did_retract) continue;

            if (active.len == 0) continue;

            int total_tok = 0, max_tok = 0;
            for (int i = 0; i < active.len; i++) {
                int kv = ss[active.buf[i]].active_kv;
                total_tok += kv;
                if (kv > max_tok) max_tok = kv;
            }
            if (total_tok == 0) continue;

            double dur = w_isolated_decode_time_estimation(
                (double)total_tok, (double)max_tok,
                (double)active.len, (double)fairinf_n);

            if (next_arrival > 0.0 && current_time + dur > next_arrival) {
                current_time = next_arrival;
                continue;
            }

            current_time += dur;

            int new_active_len = 0;
            for (int i = 0; i < active.len; i++) {
                int idx = active.buf[i];
                ss[idx].sim_decode_count++;
                ss[idx].active_kv++;
                active_kv_total++;

                if (!ss[idx].anticipated) {
                    int real_dc = reqs[idx].real_decode_count;
                    if (ss[idx].sim_decode_count > real_dc) {
                        /* anticipated = real+1 at pure sim-time.
                         * Back-calculate: sim is at sim_dc now, we want time of (real_dc+1).
                         * steps_past = sim_dc - (real_dc+1); each took dur seconds. */
                        int steps_past = ss[idx].sim_decode_count - (real_dc + 1);
                        reqs[idx].ant_type       = 1;
                        reqs[idx].ant_end_ts     = current_time - steps_past * dur;
                        reqs[idx].ant_completion = real_dc + 1;
                        ss[idx].anticipated      = 1;
                        anticipated_count++;
                    }
                }

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

        if (next_arrival < 0.0) break;
        current_time = next_arrival;
    }

    while (!w_dq_empty(&waiting)) {
        int idx = w_dq_pop_front(&waiting);
        if (!ss[idx].anticipated) {
            reqs[idx].ant_type       = 0;
            reqs[idx].ant_end_ts     = Py_HUGE_VAL;
            reqs[idx].ant_completion = 0;
            ss[idx].anticipated      = 1;
        }
    }

    /* Fallback for active requests that never crossed real_dc (sim ran out of steps).
     * Extrapolate forward: sim is at sim_dc at current_time, needs (real_dc+1 - sim_dc)
     * more steps. Pure sim-time, no wall-clock floor. */
    for (int i = 0; i < active.len; i++) {
        int idx = active.buf[i];
        if (!ss[idx].anticipated) {
            int sim_dc  = ss[idx].sim_decode_count;
            int real_dc = reqs[idx].real_decode_count;
            if (sim_dc >= real_dc) continue;  /* over-served: no deadline */
            int next_n = real_dc + 1;
            int ctx    = reqs[idx].prompt_len + next_n;
            double step_dur = w_isolated_decode_time_estimation(
                (double)ctx, (double)ctx, 1.0, (double)fairinf_n);
            int steps_remaining = next_n - sim_dc;
            reqs[idx].ant_type       = 1;
            reqs[idx].ant_end_ts     = current_time + steps_remaining * step_dur;
            reqs[idx].ant_completion = next_n;
            ss[idx].anticipated      = 1;
        }
    }

    W_FREE(order);
    W_FREE(ss);
    w_dq_free(&waiting);
    w_ia_free(&active);
    w_hm_free(&hm);
}

/* -------------------------------------------------------------------------
 * CTrackedReq: per-request state stored in the CSimulator slab
 * -------------------------------------------------------------------------*/
typedef struct {
    char   rid[W_RID_MAX];
    char   uid[W_UID_MAX];
    double arrival_ts;
    int    prompt_len;
    int    output_len;
    int    fill_len;        /* -1 if None */
    int    delta_prefill_us;
    int    delta_decode_us;
    /* requests_real */
    int    prefill_done;
    int    decode_count;
    int    is_complete;
    /* most_recent_event_real: type 0=Start, 1=Prefill, 2=Decode */
    int    mre_type;
    double mre_ts;
    int    mre_cn;
    /* compact history: -1=none, 1=Prefill, 2=Decode */
    int    hist_type;
    double hist_ts;
    int    hist_cn;
    /* next_anticipated_event: -1=none, 0=Prefill, 1=Decode */
    int    ant_type;
    double ant_ts;
    int    ant_cn;
    double latest_sim_completion_ts;
    int    alive;
} CTrackedReq;

/* -------------------------------------------------------------------------
 * CUserTimeline: per-user state stored in the CSimulator slab
 * -------------------------------------------------------------------------*/
#define MAX_REQS_PER_USER 512
typedef struct {
    char uid[W_UID_MAX];
    int  req_indices[MAX_REQS_PER_USER];
    int  req_count;
    int  alive;
} CUserTimeline;

/* -------------------------------------------------------------------------
 * CSimulatorObject: the Python type
 * -------------------------------------------------------------------------*/
typedef struct {
    PyObject_HEAD
    int    max_kv_tokens;
    int    fairinf_n;
    int    enable_timeline_logging;

    /* Request slab */
    CTrackedReq *reqs;
    int          reqs_cap;
    int          reqs_len;
    w_HashMap    req_map;   /* rid -> index */

    /* User slab */
    CUserTimeline *users;
    int            users_cap;
    int            users_len;
    w_HashMap      user_map;  /* uid -> index */

    /* Python objects held */
    PyObject *timeline_writer;  /* TIMELINE_WRITER or Py_None */
    PyObject *time_func;        /* time.time */

    /* Classes for event construction */
    PyObject *RequestPrefillEvent_cls;
    PyObject *RequestDecodeEvent_cls;
} CSimulatorObject;

static PyTypeObject CSimulatorType;  /* forward declaration */

/* -------------------------------------------------------------------------
 * Helper: convert Unix timestamp to ISO 8601 string
 * Format: "YYYY-MM-DDTHH:MM:SS.mmm+00:00"
 * -------------------------------------------------------------------------*/
static void _ts_to_iso(double ts, char *buf, size_t buflen)
{
    time_t sec = (time_t)ts;
    int ms = (int)((ts - (double)sec) * 1000.0 + 0.5);
    if (ms >= 1000) { sec++; ms -= 1000; }
    if (ms < 0) { ms = 0; }
    struct tm tm_val;
#ifdef _WIN32
    gmtime_s(&tm_val, &sec);
#else
    gmtime_r(&sec, &tm_val);
#endif
    char tmp[32];
    strftime(tmp, sizeof(tmp), "%Y-%m-%dT%H:%M:%S", &tm_val);
    snprintf(buf, buflen, "%s.%03d+00:00", tmp, ms);
}

/* -------------------------------------------------------------------------
 * Helper: get current time via self->time_func
 * -------------------------------------------------------------------------*/
static double _get_now(CSimulatorObject *self)
{
    PyObject *result = PyObject_CallNoArgs(self->time_func);
    if (!result) return 0.0;
    double t = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return t;
}

/* -------------------------------------------------------------------------
 * Helper: call TIMELINE_WRITER methods
 * -------------------------------------------------------------------------*/
static int _call_timeline_writer(
    CSimulatorObject *self,
    const char *method_name,
    const char *rid,
    const char *uid,
    const char *iso_ts,
    int completion_number,
    int is_decode_done)
{
    if (!self->enable_timeline_logging) return 0;
    if (self->timeline_writer == Py_None) return 0;

    PyObject *tw = self->timeline_writer;
    PyObject *result = NULL;

    if (strcmp(method_name, "mark_isolated_start") == 0) {
        PyObject *kwargs = PyDict_New();
        if (!kwargs) return -1;
        PyObject *ts_str = PyUnicode_FromString(iso_ts);
        if (!ts_str) { Py_DECREF(kwargs); return -1; }
        PyDict_SetItemString(kwargs, "timestamp_iso", ts_str);
        Py_DECREF(ts_str);
        PyObject *args = PyTuple_Pack(2,
            PyUnicode_FromString(rid),
            PyUnicode_FromString(uid));
        if (!args) { Py_DECREF(kwargs); return -1; }
        result = PyObject_Call(
            PyObject_GetAttrString(tw, "mark_isolated_start"),
            args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    } else if (strcmp(method_name, "mark_isolated_prefill_done") == 0) {
        PyObject *kwargs = PyDict_New();
        if (!kwargs) return -1;
        PyObject *ts_str = PyUnicode_FromString(iso_ts);
        if (!ts_str) { Py_DECREF(kwargs); return -1; }
        PyDict_SetItemString(kwargs, "timestamp_iso", ts_str);
        Py_DECREF(ts_str);
        PyObject *args = PyTuple_Pack(2,
            PyUnicode_FromString(rid),
            PyUnicode_FromString(uid));
        if (!args) { Py_DECREF(kwargs); return -1; }
        result = PyObject_Call(
            PyObject_GetAttrString(tw, "mark_isolated_prefill_done"),
            args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    } else if (strcmp(method_name, "mark_isolated_decode_done") == 0) {
        PyObject *kwargs = PyDict_New();
        if (!kwargs) return -1;
        PyObject *ts_str = PyUnicode_FromString(iso_ts);
        if (!ts_str) { Py_DECREF(kwargs); return -1; }
        PyDict_SetItemString(kwargs, "timestamp_iso", ts_str);
        Py_DECREF(ts_str);
        PyObject *cn_obj = PyLong_FromLong(completion_number);
        if (!cn_obj) { Py_DECREF(kwargs); return -1; }
        PyDict_SetItemString(kwargs, "completion_number", cn_obj);
        Py_DECREF(cn_obj);
        PyObject *args = PyTuple_Pack(2,
            PyUnicode_FromString(rid),
            PyUnicode_FromString(uid));
        if (!args) { Py_DECREF(kwargs); return -1; }
        result = PyObject_Call(
            PyObject_GetAttrString(tw, "mark_isolated_decode_done"),
            args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    } else if (strcmp(method_name, "mark_isolated_completed") == 0) {
        PyObject *kwargs = PyDict_New();
        if (!kwargs) return -1;
        PyObject *ts_str = PyUnicode_FromString(iso_ts);
        if (!ts_str) { Py_DECREF(kwargs); return -1; }
        PyDict_SetItemString(kwargs, "timestamp_iso", ts_str);
        Py_DECREF(ts_str);
        PyObject *args = PyTuple_Pack(2,
            PyUnicode_FromString(rid),
            PyUnicode_FromString(uid));
        if (!args) { Py_DECREF(kwargs); return -1; }
        result = PyObject_Call(
            PyObject_GetAttrString(tw, "mark_isolated_completed"),
            args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
    }

    (void)is_decode_done;
    if (result) {
        Py_DECREF(result);
        return 0;
    }
    /* Ignore timeline writer errors to not crash the worker */
    PyErr_Clear();
    return 0;
}

/* -------------------------------------------------------------------------
 * Helper: ensure hashmap has room (grow if >50% full)
 * -------------------------------------------------------------------------*/
static void _ensure_hm_room(w_HashMap *hm, int n_items)
{
    if (hm->slots == NULL) {
        w_hm_init(hm, n_items + 8);
        return;
    }
    /* count used slots - approximate by checking if cap/2 is enough */
    /* Just grow if we're about to exceed 50% */
    if (n_items * 2 >= hm->cap) {
        w_hm_grow(hm);
    }
}

/* -------------------------------------------------------------------------
 * Helper: find or allocate a req slot
 * Returns index, or -1 on error
 * -------------------------------------------------------------------------*/
static int _alloc_req_slot(CSimulatorObject *self, const char *rid, const char *uid)
{
    int idx = w_hm_get(&self->req_map, rid);
    if (idx != HT_EMPTY) return idx;

    /* Find a free slot */
    for (int i = 0; i < self->reqs_len; i++) {
        if (!self->reqs[i].alive) {
            idx = i;
            goto found;
        }
    }
    /* Need to grow */
    if (self->reqs_len < self->reqs_cap) {
        idx = self->reqs_len++;
    } else {
        /* Grow slab */
        int new_cap = self->reqs_cap * 2;
        CTrackedReq *new_reqs = (CTrackedReq *)W_REALLOC(
            self->reqs, new_cap * sizeof(CTrackedReq));
        if (!new_reqs) {
            PyErr_NoMemory();
            return -1;
        }
        self->reqs = new_reqs;
        self->reqs_cap = new_cap;
        idx = self->reqs_len++;
    }

found:
    memset(&self->reqs[idx], 0, sizeof(CTrackedReq));
    strncpy(self->reqs[idx].rid, rid, W_RID_MAX - 1);
    strncpy(self->reqs[idx].uid, uid, W_UID_MAX - 1);
    self->reqs[idx].fill_len = -1;
    self->reqs[idx].hist_type = -1;
    self->reqs[idx].ant_type = -1;
    self->reqs[idx].mre_type = 0;
    _ensure_hm_room(&self->req_map, self->reqs_len);
    w_hm_put(&self->req_map, rid, idx);
    return idx;
}

/* -------------------------------------------------------------------------
 * Helper: find or allocate a user slot
 * -------------------------------------------------------------------------*/
static int _alloc_user_slot(CSimulatorObject *self, const char *uid)
{
    int idx = w_hm_get(&self->user_map, uid);
    if (idx != HT_EMPTY) {
        if (!self->users[idx].alive) {
            self->users[idx].alive = 1;
            self->users[idx].req_count = 0;
        }
        return idx;
    }

    /* Find a free slot */
    for (int i = 0; i < self->users_len; i++) {
        if (!self->users[i].alive) {
            idx = i;
            goto found;
        }
    }
    if (self->users_len < self->users_cap) {
        idx = self->users_len++;
    } else {
        int new_cap = self->users_cap * 2;
        CUserTimeline *new_users = (CUserTimeline *)W_REALLOC(
            self->users, new_cap * sizeof(CUserTimeline));
        if (!new_users) {
            PyErr_NoMemory();
            return -1;
        }
        self->users = new_users;
        self->users_cap = new_cap;
        idx = self->users_len++;
    }

found:
    memset(&self->users[idx], 0, sizeof(CUserTimeline));
    strncpy(self->users[idx].uid, uid, W_UID_MAX - 1);
    self->users[idx].alive = 1;
    self->users[idx].req_count = 0;
    _ensure_hm_room(&self->user_map, self->users_len);
    w_hm_put(&self->user_map, uid, idx);
    return idx;
}

/* -------------------------------------------------------------------------
 * Helper: add req_idx to user if not already present
 * -------------------------------------------------------------------------*/
static void _add_req_to_user(CSimulatorObject *self, int user_idx, int req_idx)
{
    CUserTimeline *ut = &self->users[user_idx];
    for (int i = 0; i < ut->req_count; i++) {
        if (ut->req_indices[i] == req_idx) return;
    }
    if (ut->req_count < MAX_REQS_PER_USER) {
        ut->req_indices[ut->req_count++] = req_idx;
    }
}

/* =========================================================================
 * CSimulator.__init__
 * =========================================================================*/
static int CSimulator_init(CSimulatorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"max_kv_tokens", "fairinf_n", "enable_timeline_logging", NULL};
    int max_kv = -1, fairinf_n = 1, enable_logging = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|i", kwlist,
                                     &max_kv, &fairinf_n, &enable_logging))
        return -1;

    self->max_kv_tokens = max_kv;
    self->fairinf_n = fairinf_n > 0 ? fairinf_n : 1;
    self->enable_timeline_logging = enable_logging;

    /* Init request slab */
    self->reqs_cap = 64;
    self->reqs_len = 0;
    self->reqs = (CTrackedReq *)W_MALLOC(self->reqs_cap * sizeof(CTrackedReq));
    if (!self->reqs) { PyErr_NoMemory(); return -1; }
    w_hm_init(&self->req_map, 64);

    /* Init user slab */
    self->users_cap = 16;
    self->users_len = 0;
    self->users = (CUserTimeline *)W_MALLOC(self->users_cap * sizeof(CUserTimeline));
    if (!self->users) { W_FREE(self->reqs); PyErr_NoMemory(); return -1; }
    w_hm_init(&self->user_map, 16);

    /* Import TIMELINE_WRITER */
    PyObject *tw_mod = PyImport_ImportModule("sglang.srt.request_timeline");
    if (!tw_mod) { PyErr_Clear(); tw_mod = NULL; }
    if (tw_mod) {
        self->timeline_writer = PyObject_GetAttrString(tw_mod, "TIMELINE_WRITER");
        Py_DECREF(tw_mod);
        if (!self->timeline_writer) { PyErr_Clear(); self->timeline_writer = Py_None; Py_INCREF(Py_None); }
    } else {
        self->timeline_writer = Py_None;
        Py_INCREF(Py_None);
    }

    /* Import time.time */
    PyObject *time_mod = PyImport_ImportModule("time");
    if (!time_mod) { return -1; }
    self->time_func = PyObject_GetAttrString(time_mod, "time");
    Py_DECREF(time_mod);
    if (!self->time_func) return -1;

    /* Import event classes */
    PyObject *sim_mod = PyImport_ImportModule(
        "sglang.srt.delta_fairness.doc_policy_simulator");
    if (!sim_mod) { PyErr_Clear(); sim_mod = NULL; }
    if (sim_mod) {
        self->RequestPrefillEvent_cls = PyObject_GetAttrString(sim_mod, "RequestPrefillEvent");
        self->RequestDecodeEvent_cls  = PyObject_GetAttrString(sim_mod, "RequestDecodeEvent");
        Py_DECREF(sim_mod);
        if (!self->RequestPrefillEvent_cls) { PyErr_Clear(); self->RequestPrefillEvent_cls = Py_None; Py_INCREF(Py_None); }
        if (!self->RequestDecodeEvent_cls)  { PyErr_Clear(); self->RequestDecodeEvent_cls  = Py_None; Py_INCREF(Py_None); }
    } else {
        self->RequestPrefillEvent_cls = Py_None; Py_INCREF(Py_None);
        self->RequestDecodeEvent_cls  = Py_None; Py_INCREF(Py_None);
    }

    return 0;
}

static void CSimulator_dealloc(CSimulatorObject *self)
{
    W_FREE(self->reqs);
    W_FREE(self->users);
    w_hm_free(&self->req_map);
    w_hm_free(&self->user_map);
    Py_XDECREF(self->timeline_writer);
    Py_XDECREF(self->time_func);
    Py_XDECREF(self->RequestPrefillEvent_cls);
    Py_XDECREF(self->RequestDecodeEvent_cls);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* =========================================================================
 * set_timeline_writer(tw): update the stored timeline writer reference
 * =========================================================================*/
static PyObject *
CSimulator_set_timeline_writer(CSimulatorObject *self, PyObject *args)
{
    PyObject *tw;
    if (!PyArg_ParseTuple(args, "O", &tw)) return NULL;
    Py_INCREF(tw);
    Py_DECREF(self->timeline_writer);
    self->timeline_writer = tw;
    Py_RETURN_NONE;
}

/* =========================================================================
 * process_new_request(uid, rid, prompt_len, fill_len, output_len, arrival_ts,
 *                     delta_prefill_us, delta_decode_us)
 * =========================================================================*/
static PyObject *
CSimulator_process_new_request(CSimulatorObject *self, PyObject *args)
{
    const char *uid, *rid;
    int prompt_len, fill_len, output_len;
    double arrival_ts;
    int delta_prefill_us, delta_decode_us;

    if (!PyArg_ParseTuple(args, "ssiiidii",
                          &uid, &rid, &prompt_len, &fill_len, &output_len,
                          &arrival_ts, &delta_prefill_us, &delta_decode_us))
        return NULL;

    int idx = _alloc_req_slot(self, rid, uid);
    if (idx < 0) return NULL;

    CTrackedReq *tr = &self->reqs[idx];
    strncpy(tr->rid, rid, W_RID_MAX - 1);
    strncpy(tr->uid, uid, W_UID_MAX - 1);
    tr->arrival_ts = arrival_ts;
    tr->prompt_len = prompt_len;
    tr->output_len = output_len;
    tr->fill_len   = fill_len;
    tr->delta_prefill_us = delta_prefill_us;
    tr->delta_decode_us  = delta_decode_us;
    tr->prefill_done = 0;
    tr->decode_count = 0;
    tr->is_complete  = 0;
    tr->mre_type = 0;
    tr->mre_ts   = arrival_ts;
    tr->mre_cn   = 0;
    tr->hist_type = -1;
    tr->hist_ts   = 0.0;
    tr->hist_cn   = 0;
    tr->latest_sim_completion_ts = 0.0;

    /* Compute initial anticipated prefill */
    double context_tokens = (fill_len >= 0)
        ? (double)fill_len
        : (double)(prompt_len + output_len);
    double dur = w_isolated_prefill_time_estimation(
        context_tokens, context_tokens, 1.0, (double)self->fairinf_n);
    tr->ant_type = 0;
    tr->ant_ts   = arrival_ts + dur;
    tr->ant_cn   = 0;

    tr->alive = 1;

    /* Ensure user exists */
    int user_idx = _alloc_user_slot(self, uid);
    if (user_idx < 0) return NULL;
    _add_req_to_user(self, user_idx, idx);

    /* Timeline logging */
    if (self->enable_timeline_logging) {
        char iso_buf[40];
        _ts_to_iso(arrival_ts, iso_buf, sizeof(iso_buf));
        _call_timeline_writer(self, "mark_isolated_start", rid, uid, iso_buf, 0, 0);
    }

    Py_RETURN_NONE;
}

/* =========================================================================
 * note_prefill_done(uid, rid, prompt_len, output_len)
 * =========================================================================*/
static PyObject *
CSimulator_note_prefill_done(CSimulatorObject *self, PyObject *args)
{
    const char *uid, *rid;
    int prompt_len, output_len;
    if (!PyArg_ParseTuple(args, "ssii", &uid, &rid, &prompt_len, &output_len))
        return NULL;

    int idx = w_hm_get(&self->req_map, rid);
    if (idx == HT_EMPTY) Py_RETURN_NONE;

    CTrackedReq *tr = &self->reqs[idx];
    if (tr->hist_type < 1) {
        /* No prefill event yet — compute iso_ts */
        double iso_ts = tr->arrival_ts + w_isolated_prefill_time_estimation(
            (double)prompt_len, (double)prompt_len, 1.0, (double)self->fairinf_n);
        tr->hist_type = 1;
        tr->hist_ts   = iso_ts;
        tr->hist_cn   = 0;
        tr->mre_type  = 1;
        tr->mre_ts    = iso_ts;
        tr->mre_cn    = 0;
        tr->latest_sim_completion_ts = iso_ts;
    }
    tr->prefill_done = 1;

    Py_RETURN_NONE;
}

/* =========================================================================
 * finished_prefill(entries) — entries is list of (uid, rid, prompt_len, output_len)
 * =========================================================================*/
static PyObject *
CSimulator_finished_prefill(CSimulatorObject *self, PyObject *args)
{
    PyObject *entries;
    if (!PyArg_ParseTuple(args, "O", &entries)) return NULL;

    Py_ssize_t nentries = PyList_Size(entries);
    if (nentries < 0) return NULL;

    for (Py_ssize_t i = 0; i < nentries; i++) {
        PyObject *tup = PyList_GET_ITEM(entries, i);
        const char *uid = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0));
        const char *rid = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1));
        int prompt_len  = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        int output_len  = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 3));
        if (PyErr_Occurred()) return NULL;

        int idx = w_hm_get(&self->req_map, rid);
        if (idx == HT_EMPTY) continue;
        CTrackedReq *tr = &self->reqs[idx];

        double iso_ts;
        if (tr->ant_type == 0) {
            iso_ts = tr->ant_ts;
        } else {
            iso_ts = tr->arrival_ts + w_isolated_prefill_time_estimation(
                (double)prompt_len, (double)prompt_len, 1.0, (double)self->fairinf_n);
        }

        tr->hist_type = 1;
        tr->hist_ts   = iso_ts;
        tr->hist_cn   = 0;
        tr->prefill_done = 1;
        tr->mre_type  = 1;
        tr->mre_ts    = iso_ts;
        tr->mre_cn    = 0;
        tr->latest_sim_completion_ts = iso_ts;

        /* Seed anticipated first decode */
        double now = _get_now(self);
        int first_n = output_len + 1;
        int ctx = prompt_len + first_n;
        double dec_dur = w_isolated_decode_time_estimation(
            (double)ctx, (double)ctx, 1.0, (double)self->fairinf_n);
        double base = iso_ts > now ? iso_ts : now;
        tr->ant_type = 1;
        tr->ant_ts   = base + dec_dur;
        tr->ant_cn   = first_n;

        /* Timeline logging */
        if (self->enable_timeline_logging) {
            char iso_buf[40];
            _ts_to_iso(iso_ts, iso_buf, sizeof(iso_buf));
            _call_timeline_writer(self, "mark_isolated_prefill_done", rid, uid, iso_buf, 0, 0);
        }
    }

    Py_RETURN_NONE;
}

/* =========================================================================
 * finished_decode(entries, decode_rounds) — entries is list of (uid, rid, prompt_len, output_len)
 * =========================================================================*/
static PyObject *
CSimulator_finished_decode(CSimulatorObject *self, PyObject *args)
{
    PyObject *entries;
    int decode_rounds = 1;
    if (!PyArg_ParseTuple(args, "O|i", &entries, &decode_rounds)) return NULL;

    Py_ssize_t nentries = PyList_Size(entries);
    if (nentries < 0) return NULL;
    if (decode_rounds < 1) decode_rounds = 1;

    int fn = self->fairinf_n;

    for (Py_ssize_t i = 0; i < nentries; i++) {
        PyObject *tup = PyList_GET_ITEM(entries, i);
        const char *uid = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0));
        const char *rid = PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1));
        int prompt_len  = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        int output_len  = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 3));
        if (PyErr_Occurred()) return NULL;

        int idx = w_hm_get(&self->req_map, rid);
        if (idx == HT_EMPTY) continue;
        CTrackedReq *tr = &self->reqs[idx];

        int last_n;
        double base_ts;

        if (tr->hist_type == 2) {
            last_n   = tr->hist_cn;
            base_ts  = tr->hist_ts;
        } else if (tr->hist_type == 1) {
            last_n   = 0;
            base_ts  = tr->hist_ts;
        } else {
            /* Synthesize prefill */
            double dur = w_isolated_prefill_time_estimation(
                (double)prompt_len, (double)prompt_len, 1.0, (double)fn);
            base_ts = tr->arrival_ts + dur;
            last_n  = 0;
            tr->hist_type = 1;
            tr->hist_ts   = base_ts;
            tr->hist_cn   = 0;
            tr->prefill_done = 1;
            tr->mre_type  = 1;
            tr->mre_ts    = base_ts;
            tr->mre_cn    = 0;
        }

        int new_final_n = last_n + decode_rounds;
        if (output_len > new_final_n) new_final_n = output_len;

        double cur_ts = base_ts;
        for (int n = last_n + 1; n <= new_final_n; n++) {
            int ctx = prompt_len + n;
            double dur = w_isolated_decode_time_estimation(
                (double)ctx, (double)ctx, 1.0, (double)fn);
            cur_ts += dur;
            if (self->enable_timeline_logging) {
                char iso_buf[40];
                _ts_to_iso(cur_ts, iso_buf, sizeof(iso_buf));
                _call_timeline_writer(self, "mark_isolated_decode_done", rid, uid, iso_buf, n, 1);
            }
        }

        tr->hist_type  = 2;
        tr->hist_ts    = cur_ts;
        tr->hist_cn    = new_final_n;
        tr->mre_type   = 2;
        tr->mre_ts     = cur_ts;
        tr->mre_cn     = new_final_n;
        tr->prefill_done = 1;
        tr->decode_count = new_final_n;
        tr->latest_sim_completion_ts = cur_ts;

        /* Seed next anticipated decode — pure sim-time, no wall-clock floor */
        int next_n = new_final_n + 1;
        int ctx = prompt_len + next_n;
        double dec_dur = w_isolated_decode_time_estimation(
            (double)ctx, (double)ctx, 1.0, (double)fn);
        tr->ant_type = 1;
        tr->ant_ts   = cur_ts + dec_dur;
        tr->ant_cn   = next_n;
    }

    Py_RETURN_NONE;
}

/* =========================================================================
 * mark_request_finished(rid, uid, output_len)
 * =========================================================================*/
static PyObject *
CSimulator_mark_request_finished(CSimulatorObject *self, PyObject *args)
{
    const char *rid, *uid;
    int output_len;
    if (!PyArg_ParseTuple(args, "ssi", &rid, &uid, &output_len)) return NULL;

    int idx = w_hm_get(&self->req_map, rid);
    if (idx == HT_EMPTY) Py_RETURN_NONE;

    CTrackedReq *tr = &self->reqs[idx];
    int fn = self->fairinf_n;
    int final_n = output_len;
    int last_n;
    double base_ts;

    if (tr->hist_type >= 1) {
        base_ts = tr->hist_ts;
        last_n  = (tr->hist_type == 2) ? tr->hist_cn : 0;
    } else {
        base_ts = tr->arrival_ts;
        last_n  = 0;
    }

    for (int n = last_n + 1; n <= final_n; n++) {
        int ctx = tr->prompt_len + n;
        double dur = w_isolated_decode_time_estimation(
            (double)ctx, (double)ctx, 1.0, (double)fn);
        base_ts += dur;
        if (self->enable_timeline_logging) {
            char iso_buf[40];
            _ts_to_iso(base_ts, iso_buf, sizeof(iso_buf));
            _call_timeline_writer(self, "mark_isolated_decode_done", rid, uid, iso_buf, n, 1);
        }
    }

    if (final_n > last_n || tr->hist_type < 1) {
        tr->hist_type = 2;
        tr->hist_ts   = base_ts;
        tr->hist_cn   = final_n;
        tr->latest_sim_completion_ts = base_ts;
    }

    if (tr->latest_sim_completion_ts > 0.0 && self->enable_timeline_logging) {
        char iso_buf[40];
        _ts_to_iso(tr->latest_sim_completion_ts, iso_buf, sizeof(iso_buf));
        _call_timeline_writer(self, "mark_isolated_completed", rid, uid, iso_buf, 0, 0);
    }

    tr->is_complete = 1;
    tr->alive = 0;

    /* Remove from user's req_indices */
    int user_idx = w_hm_get(&self->user_map, uid);
    if (user_idx != HT_EMPTY) {
        CUserTimeline *ut = &self->users[user_idx];
        for (int i = 0; i < ut->req_count; i++) {
            if (ut->req_indices[i] == idx) {
                ut->req_indices[i] = ut->req_indices[--ut->req_count];
                break;
            }
        }
    }

    /* Remove from req_map */
    w_hm_delete(&self->req_map, rid);

    Py_RETURN_NONE;
}

/* =========================================================================
 * sync_live_users(waiting_tuples, running_tuples)
 * =========================================================================*/

typedef struct {
    char uid[W_UID_MAX];
    char rid[W_RID_MAX];
    int  prompt_len;
    int  output_len;
} LiveEntry;

static PyObject *
CSimulator_sync_live_users(CSimulatorObject *self, PyObject *args)
{
    PyObject *py_waiting, *py_running;
    if (!PyArg_ParseTuple(args, "OO", &py_waiting, &py_running)) return NULL;

    Py_ssize_t n_wait = PyList_Size(py_waiting);
    Py_ssize_t n_run  = PyList_Size(py_running);
    if (n_wait < 0 || n_run < 0) return NULL;

    /* Extract all tuples into C arrays (GIL held) */
    LiveEntry *waiting_entries = NULL;
    LiveEntry *running_entries = NULL;

    if (n_wait > 0) {
        waiting_entries = (LiveEntry *)W_MALLOC(n_wait * sizeof(LiveEntry));
        if (!waiting_entries) return PyErr_NoMemory();
    }
    if (n_run > 0) {
        running_entries = (LiveEntry *)W_MALLOC(n_run * sizeof(LiveEntry));
        if (!running_entries) { W_FREE(waiting_entries); return PyErr_NoMemory(); }
    }

    for (Py_ssize_t i = 0; i < n_wait; i++) {
        PyObject *tup = PyList_GET_ITEM(py_waiting, i);
        strncpy(waiting_entries[i].uid,
            PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0)), W_UID_MAX - 1);
        strncpy(waiting_entries[i].rid,
            PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1)), W_RID_MAX - 1);
        waiting_entries[i].prompt_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        waiting_entries[i].output_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 3));
        if (PyErr_Occurred()) { W_FREE(waiting_entries); W_FREE(running_entries); return NULL; }
    }
    for (Py_ssize_t i = 0; i < n_run; i++) {
        PyObject *tup = PyList_GET_ITEM(py_running, i);
        strncpy(running_entries[i].uid,
            PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 0)), W_UID_MAX - 1);
        strncpy(running_entries[i].rid,
            PyUnicode_AsUTF8(PyTuple_GET_ITEM(tup, 1)), W_RID_MAX - 1);
        running_entries[i].prompt_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 2));
        running_entries[i].output_len = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 3));
        if (PyErr_Occurred()) { W_FREE(waiting_entries); W_FREE(running_entries); return NULL; }
    }

    /* Pre-grow slabs and hashmaps BEFORE releasing GIL to avoid infinite probe
     * loops inside the GIL-free section (w_hm_put loops forever on full maps). */
    {
        int n_new = (int)(n_wait + n_run);

        /* Grow req slab */
        while (self->reqs_len + n_new > self->reqs_cap) {
            int new_cap = self->reqs_cap * 2;
            CTrackedReq *new_reqs = (CTrackedReq *)W_REALLOC(
                self->reqs, new_cap * sizeof(CTrackedReq));
            if (!new_reqs) {
                W_FREE(waiting_entries); W_FREE(running_entries);
                return PyErr_NoMemory();
            }
            self->reqs = new_reqs;
            self->reqs_cap = new_cap;
        }

        /* Grow user slab */
        while (self->users_len + n_new > self->users_cap) {
            int new_cap = self->users_cap * 2;
            CUserTimeline *new_users = (CUserTimeline *)W_REALLOC(
                self->users, new_cap * sizeof(CUserTimeline));
            if (!new_users) {
                W_FREE(waiting_entries); W_FREE(running_entries);
                return PyErr_NoMemory();
            }
            self->users = new_users;
            self->users_cap = new_cap;
        }

        /* Grow req_map: keep load factor <= 0.5 */
        while ((self->reqs_len + n_new) * 2 >= self->req_map.cap)
            w_hm_grow(&self->req_map);

        /* Grow user_map: keep load factor <= 0.5 */
        while ((self->users_len + n_new) * 2 >= self->user_map.cap)
            w_hm_grow(&self->user_map);
    }

    /* GIL-free section */
    Py_BEGIN_ALLOW_THREADS

    /* Build live rids set */
    int n_total = (int)(n_wait + n_run);
    w_HashMap live_set;
    w_hm_init(&live_set, n_total + 8);
    for (int i = 0; i < n_wait; i++)
        w_hm_put(&live_set, waiting_entries[i].rid, 1);
    for (int i = 0; i < n_run; i++)
        w_hm_put(&live_set, running_entries[i].rid, 1);

    /* Process waiting entries */
    for (int i = 0; i < (int)n_wait; i++) {
        LiveEntry *e = &waiting_entries[i];
        int idx = w_hm_get(&self->req_map, e->rid);
        if (idx == HT_EMPTY) {
            /* Create minimal slot */
            /* Need to do this without PyMem — can't call _alloc_req_slot (uses Python memory) */
            /* Find free slot */
            int new_idx = -1;
            for (int j = 0; j < self->reqs_len; j++) {
                if (!self->reqs[j].alive) { new_idx = j; break; }
            }
            if (new_idx < 0) {
                if (self->reqs_len < self->reqs_cap) {
                    new_idx = self->reqs_len++;
                }
                /* If no room, skip this entry */
                if (new_idx < 0) continue;
            }
            memset(&self->reqs[new_idx], 0, sizeof(CTrackedReq));
            strncpy(self->reqs[new_idx].rid, e->rid, W_RID_MAX - 1);
            strncpy(self->reqs[new_idx].uid, e->uid, W_UID_MAX - 1);
            self->reqs[new_idx].arrival_ts = 0.0;
            self->reqs[new_idx].fill_len = -1;
            self->reqs[new_idx].hist_type = -1;
            self->reqs[new_idx].ant_type = -1;
            self->reqs[new_idx].alive = 1;
            w_hm_put(&self->req_map, e->rid, new_idx);
            idx = new_idx;
        }
        CTrackedReq *tr = &self->reqs[idx];
        tr->output_len = e->output_len;
        tr->prompt_len = e->prompt_len;
        /* Reset waiting state */
        tr->prefill_done = 0;
        tr->decode_count = 0;
        tr->hist_type = -1;
        tr->hist_ts   = 0.0;
        tr->hist_cn   = 0;
        tr->mre_type  = 0;
        tr->mre_ts    = tr->arrival_ts;
        tr->mre_cn    = 0;
        tr->alive     = 1;
        strncpy(tr->uid, e->uid, W_UID_MAX - 1);

        /* Ensure user slot */
        int uidx = w_hm_get(&self->user_map, e->uid);
        if (uidx == HT_EMPTY) {
            int new_uidx = -1;
            for (int j = 0; j < self->users_len; j++) {
                if (!self->users[j].alive) { new_uidx = j; break; }
            }
            if (new_uidx < 0 && self->users_len < self->users_cap) {
                new_uidx = self->users_len++;
            }
            if (new_uidx >= 0) {
                memset(&self->users[new_uidx], 0, sizeof(CUserTimeline));
                strncpy(self->users[new_uidx].uid, e->uid, W_UID_MAX - 1);
                self->users[new_uidx].alive = 1;
                w_hm_put(&self->user_map, e->uid, new_uidx);
                uidx = new_uidx;
            }
        } else if (!self->users[uidx].alive) {
            self->users[uidx].alive = 1;
            self->users[uidx].req_count = 0;
        }
        if (uidx >= 0) {
            CUserTimeline *ut = &self->users[uidx];
            int found = 0;
            for (int j = 0; j < ut->req_count; j++) {
                if (ut->req_indices[j] == idx) { found = 1; break; }
            }
            if (!found && ut->req_count < MAX_REQS_PER_USER)
                ut->req_indices[ut->req_count++] = idx;
        }
    }

    /* Process running entries */
    for (int i = 0; i < (int)n_run; i++) {
        LiveEntry *e = &running_entries[i];
        int idx = w_hm_get(&self->req_map, e->rid);
        if (idx == HT_EMPTY) {
            int new_idx = -1;
            for (int j = 0; j < self->reqs_len; j++) {
                if (!self->reqs[j].alive) { new_idx = j; break; }
            }
            if (new_idx < 0 && self->reqs_len < self->reqs_cap) {
                new_idx = self->reqs_len++;
            }
            if (new_idx < 0) continue;
            memset(&self->reqs[new_idx], 0, sizeof(CTrackedReq));
            strncpy(self->reqs[new_idx].rid, e->rid, W_RID_MAX - 1);
            strncpy(self->reqs[new_idx].uid, e->uid, W_UID_MAX - 1);
            self->reqs[new_idx].arrival_ts = 0.0;
            self->reqs[new_idx].fill_len = -1;
            self->reqs[new_idx].hist_type = -1;
            self->reqs[new_idx].ant_type = -1;
            self->reqs[new_idx].alive = 1;
            w_hm_put(&self->req_map, e->rid, new_idx);
            idx = new_idx;
        }
        CTrackedReq *tr = &self->reqs[idx];
        tr->output_len = e->output_len;
        tr->prompt_len = e->prompt_len;
        /* Derive state from history */
        tr->prefill_done = (tr->hist_type >= 1) ? 1 : 0;
        tr->decode_count = (tr->hist_type == 2) ? tr->hist_cn : 0;
        tr->alive = 1;
        strncpy(tr->uid, e->uid, W_UID_MAX - 1);

        /* Ensure user slot */
        int uidx = w_hm_get(&self->user_map, e->uid);
        if (uidx == HT_EMPTY) {
            int new_uidx = -1;
            for (int j = 0; j < self->users_len; j++) {
                if (!self->users[j].alive) { new_uidx = j; break; }
            }
            if (new_uidx < 0 && self->users_len < self->users_cap) {
                new_uidx = self->users_len++;
            }
            if (new_uidx >= 0) {
                memset(&self->users[new_uidx], 0, sizeof(CUserTimeline));
                strncpy(self->users[new_uidx].uid, e->uid, W_UID_MAX - 1);
                self->users[new_uidx].alive = 1;
                w_hm_put(&self->user_map, e->uid, new_uidx);
                uidx = new_uidx;
            }
        } else if (!self->users[uidx].alive) {
            self->users[uidx].alive = 1;
            self->users[uidx].req_count = 0;
        }
        if (uidx >= 0) {
            CUserTimeline *ut = &self->users[uidx];
            int found = 0;
            for (int j = 0; j < ut->req_count; j++) {
                if (ut->req_indices[j] == idx) { found = 1; break; }
            }
            if (!found && ut->req_count < MAX_REQS_PER_USER)
                ut->req_indices[ut->req_count++] = idx;
        }
    }

    /* Remove dead requests and update users */
    for (int i = 0; i < self->users_len; i++) {
        if (!self->users[i].alive) continue;
        CUserTimeline *ut = &self->users[i];
        int new_count = 0;
        for (int j = 0; j < ut->req_count; j++) {
            int ridx = ut->req_indices[j];
            if (ridx < 0 || ridx >= self->reqs_len) continue;
            CTrackedReq *tr = &self->reqs[ridx];
            if (w_hm_get(&live_set, tr->rid) != HT_EMPTY) {
                ut->req_indices[new_count++] = ridx;
            }
        }
        ut->req_count = new_count;
        if (new_count == 0) {
            ut->alive = 0;
            w_hm_delete(&self->user_map, ut->uid);
        }
    }

    /* Mark dead requests */
    for (int i = 0; i < self->reqs_len; i++) {
        if (!self->reqs[i].alive) continue;
        if (w_hm_get(&live_set, self->reqs[i].rid) == HT_EMPTY) {
            self->reqs[i].alive = 0;
            w_hm_delete(&self->req_map, self->reqs[i].rid);
        }
    }

    w_hm_free(&live_set);

    Py_END_ALLOW_THREADS

    W_FREE(waiting_entries);
    W_FREE(running_entries);

    Py_RETURN_NONE;
}

/* =========================================================================
 * rebuild_all_users(known_fair_uids=None)
 * =========================================================================*/
static PyObject *
CSimulator_rebuild_all_users(CSimulatorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"known_fair_uids", NULL};
    PyObject *known_fair_uids = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &known_fair_uids))
        return NULL;

    /* Extract known_fair_uids into a C hashmap (GIL held) */
    w_HashMap fair_set;
    int has_fair_filter = 0;

    if (known_fair_uids != Py_None && known_fair_uids != NULL) {
        has_fair_filter = 1;
        PyObject *iter = PyObject_GetIter(known_fair_uids);
        if (!iter) return NULL;
        /* count first */
        Py_ssize_t sz = PyObject_Length(known_fair_uids);
        if (sz < 0) sz = 16;
        w_hm_init(&fair_set, (int)sz + 8);
        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            const char *uid_s = PyUnicode_AsUTF8(item);
            if (uid_s) w_hm_put(&fair_set, uid_s, 1);
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) { w_hm_free(&fair_set); return NULL; }
    }

    /* Build per-user w_CReq arrays (GIL held) */
    /* We'll store them in a flat array and track offsets */
    int n_users = 0;
    for (int i = 0; i < self->users_len; i++) {
        if (!self->users[i].alive) continue;
        if (has_fair_filter && w_hm_get(&fair_set, self->users[i].uid) == HT_EMPTY) continue;
        n_users++;
    }

    /* Per-user: count of reqs */
    typedef struct {
        int user_idx;
        int n_reqs;
        w_CReq *reqs;
        int *req_indices;  /* back-map to CTrackedReq indices */
    } UserWork;

    UserWork *uw = NULL;
    if (n_users > 0) {
        uw = (UserWork *)W_MALLOC(n_users * sizeof(UserWork));
        if (!uw) {
            if (has_fair_filter) w_hm_free(&fair_set);
            return PyErr_NoMemory();
        }
    }

    int uw_idx = 0;
    for (int i = 0; i < self->users_len; i++) {
        if (!self->users[i].alive) continue;
        if (has_fair_filter && w_hm_get(&fair_set, self->users[i].uid) == HT_EMPTY) continue;

        CUserTimeline *ut = &self->users[i];
        int n = 0;
        /* Count alive reqs */
        for (int j = 0; j < ut->req_count; j++) {
            int ridx = ut->req_indices[j];
            if (ridx >= 0 && ridx < self->reqs_len && self->reqs[ridx].alive) n++;
        }
        uw[uw_idx].user_idx = i;
        uw[uw_idx].n_reqs = n;
        uw[uw_idx].reqs = NULL;
        uw[uw_idx].req_indices = NULL;
        if (n > 0) {
            uw[uw_idx].reqs = (w_CReq *)W_MALLOC(n * sizeof(w_CReq));
            uw[uw_idx].req_indices = (int *)W_MALLOC(n * sizeof(int));
            if (!uw[uw_idx].reqs || !uw[uw_idx].req_indices) {
                /* cleanup */
                for (int k = 0; k <= uw_idx; k++) {
                    W_FREE(uw[k].reqs);
                    W_FREE(uw[k].req_indices);
                }
                W_FREE(uw);
                if (has_fair_filter) w_hm_free(&fair_set);
                return PyErr_NoMemory();
            }
            int ridx_out = 0;
            for (int j = 0; j < ut->req_count; j++) {
                int ridx = ut->req_indices[j];
                if (ridx < 0 || ridx >= self->reqs_len || !self->reqs[ridx].alive) continue;
                CTrackedReq *tr = &self->reqs[ridx];
                w_CReq *cr = &uw[uw_idx].reqs[ridx_out];
                strncpy(cr->rid, tr->rid, W_RID_MAX - 1);
                cr->rid[W_RID_MAX - 1] = '\0';
                cr->arrival_ts        = tr->arrival_ts;
                cr->prompt_len        = tr->prompt_len;
                cr->real_decode_count = tr->decode_count;
                cr->prefill_done      = tr->prefill_done;
                cr->is_complete       = tr->is_complete;
                cr->ant_type          = -1;
                cr->ant_end_ts        = 0.0;
                cr->ant_completion    = 0;
                uw[uw_idx].req_indices[ridx_out] = ridx;
                ridx_out++;
            }
        }
        uw_idx++;
    }

    /* GIL-free: run rebuild kernel for each user */
    Py_BEGIN_ALLOW_THREADS

    for (int u = 0; u < n_users; u++) {
        if (uw[u].n_reqs <= 0) continue;
        w_run_rebuild_kernel(uw[u].reqs, uw[u].n_reqs,
                             self->max_kv_tokens, self->fairinf_n);
    }

    Py_END_ALLOW_THREADS

    /* Write back ant_type/ant_ts/ant_cn to CTrackedReq (GIL held) */
    for (int u = 0; u < n_users; u++) {
        for (int j = 0; j < uw[u].n_reqs; j++) {
            int ridx = uw[u].req_indices[j];
            if (ridx < 0 || ridx >= self->reqs_len) continue;
            CTrackedReq *tr = &self->reqs[ridx];
            w_CReq *cr = &uw[u].reqs[j];
            if (cr->ant_type >= 0) {
                tr->ant_type = cr->ant_type;
                tr->ant_ts   = cr->ant_end_ts;
                tr->ant_cn   = cr->ant_completion;
            }
        }
        W_FREE(uw[u].reqs);
        W_FREE(uw[u].req_indices);
    }

    W_FREE(uw);
    if (has_fair_filter) w_hm_free(&fair_set);

    Py_RETURN_NONE;
}

/* =========================================================================
 * build_deadline_candidates(waiting_rids, running_rids, fair_uids,
 *   fair_decode_uids, delta_prefill_s, delta_decode_s,
 *   pooled_prefill_s, pooled_decode_s)
 *
 * Returns (deadline_list, waiting_prefill_deadlines_dict, ordered_waiting_rids_tuple)
 * =========================================================================*/

typedef struct {
    char   rid[W_RID_MAX];
    char   uid[W_UID_MAX];
    int    event_type;   /* 0=prefill, 1=decode */
    double deadline;
    double start_deadline;
    double ant_ts;
    int    ant_cn;
    double arrival_ts;
} DLCandidate;

static PyObject *
CSimulator_build_deadline_candidates(CSimulatorObject *self, PyObject *args)
{
    PyObject *py_waiting_rids, *py_running_rids;
    PyObject *py_fair_uids, *py_fair_decode_uids;
    double delta_prefill_s, delta_decode_s;
    double pooled_prefill_s, pooled_decode_s;

    if (!PyArg_ParseTuple(args, "OOOOdddd",
                          &py_waiting_rids, &py_running_rids,
                          &py_fair_uids, &py_fair_decode_uids,
                          &delta_prefill_s, &delta_decode_s,
                          &pooled_prefill_s, &pooled_decode_s))
        return NULL;

    /* Extract inputs (GIL held) */
    Py_ssize_t n_wait = PyList_Size(py_waiting_rids);
    Py_ssize_t n_run  = PyList_Size(py_running_rids);
    if (n_wait < 0 || n_run < 0) return NULL;

    /* Build waiting/running rid sets */
    w_HashMap waiting_set;
    w_hm_init(&waiting_set, (int)n_wait + 8);
    for (Py_ssize_t i = 0; i < n_wait; i++) {
        PyObject *s = PyList_GET_ITEM(py_waiting_rids, i);
        const char *rid_s = PyUnicode_AsUTF8(s);
        if (!rid_s) { w_hm_free(&waiting_set); return NULL; }
        w_hm_put(&waiting_set, rid_s, 1);
    }

    w_HashMap running_set;
    w_hm_init(&running_set, (int)n_run + 8);
    for (Py_ssize_t i = 0; i < n_run; i++) {
        PyObject *s = PyList_GET_ITEM(py_running_rids, i);
        const char *rid_s = PyUnicode_AsUTF8(s);
        if (!rid_s) { w_hm_free(&waiting_set); w_hm_free(&running_set); return NULL; }
        w_hm_put(&running_set, rid_s, 1);
    }

    /* Extract fair_uids */
    w_HashMap fair_uids_set;
    int has_fair_uids = 0;
    if (py_fair_uids != Py_None && py_fair_uids != NULL) {
        has_fair_uids = 1;
        PyObject *iter = PyObject_GetIter(py_fair_uids);
        if (!iter) { w_hm_free(&waiting_set); w_hm_free(&running_set); return NULL; }
        Py_ssize_t sz = PyObject_Length(py_fair_uids);
        if (sz < 0) sz = 8;
        w_hm_init(&fair_uids_set, (int)sz + 8);
        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            const char *uid_s = PyUnicode_AsUTF8(item);
            if (uid_s) w_hm_put(&fair_uids_set, uid_s, 1);
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) {
            w_hm_free(&waiting_set); w_hm_free(&running_set);
            w_hm_free(&fair_uids_set); return NULL;
        }
    }

    /* Extract fair_decode_uids */
    w_HashMap fair_decode_set;
    int has_fair_decode = 0;
    if (py_fair_decode_uids != Py_None && py_fair_decode_uids != NULL) {
        has_fair_decode = 1;
        PyObject *iter = PyObject_GetIter(py_fair_decode_uids);
        if (!iter) {
            w_hm_free(&waiting_set); w_hm_free(&running_set);
            if (has_fair_uids) w_hm_free(&fair_uids_set);
            return NULL;
        }
        Py_ssize_t sz = PyObject_Length(py_fair_decode_uids);
        if (sz < 0) sz = 8;
        w_hm_init(&fair_decode_set, (int)sz + 8);
        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            const char *uid_s = PyUnicode_AsUTF8(item);
            if (uid_s) w_hm_put(&fair_decode_set, uid_s, 1);
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) {
            w_hm_free(&waiting_set); w_hm_free(&running_set);
            if (has_fair_uids) w_hm_free(&fair_uids_set);
            w_hm_free(&fair_decode_set); return NULL;
        }
    }

    /* Allocate output arrays */
    int n_reqs = self->reqs_len;
    DLCandidate *prefill_candidates = NULL;
    int n_prefill = 0;
    /* waiting_prefill_deadlines: rid -> start_deadline */
    /* We'll collect them as C arrays and build Python dict later */
    char **wpd_rids   = NULL;
    double *wpd_vals  = NULL;
    int    n_wpd      = 0;

    if (n_reqs > 0) {
        prefill_candidates = (DLCandidate *)W_MALLOC(n_reqs * sizeof(DLCandidate));
        wpd_rids = (char **)W_MALLOC(n_reqs * sizeof(char *));
        wpd_vals = (double *)W_MALLOC(n_reqs * sizeof(double));
        /* allocate strings for wpd */
        for (int i = 0; i < n_reqs; i++) {
            wpd_rids[i] = (char *)W_MALLOC(W_RID_MAX);
        }
        if (!prefill_candidates || !wpd_rids || !wpd_vals) {
            /* cleanup and fail */
            goto cleanup_and_error;
        }
    }

    DLCandidate earliest_decode;
    int has_earliest_decode = 0;
    memset(&earliest_decode, 0, sizeof(earliest_decode));

    int fn = self->fairinf_n;

    /* GIL-free computation */
    Py_BEGIN_ALLOW_THREADS

    for (int i = 0; i < self->reqs_len; i++) {
        if (!self->reqs[i].alive) continue;
        CTrackedReq *tr = &self->reqs[i];

        int in_waiting = (w_hm_get(&waiting_set, tr->rid) != HT_EMPTY);
        int in_running = (w_hm_get(&running_set, tr->rid) != HT_EMPTY);

        if (!in_waiting && !in_running) continue;

        /* Compute upcoming event based on mre_type and ant */
        int upcoming_type = -1;
        double upcoming_ts = 0.0;
        int upcoming_cn = 0;

        int mre_type = tr->mre_type;

        if (mre_type == 0) {
            /* Start: upcoming = ant if ant_type >= 0 */
            if (tr->ant_type >= 0) {
                upcoming_type = tr->ant_type;
                upcoming_ts   = tr->ant_ts;
                upcoming_cn   = tr->ant_cn;
            }
        } else if (mre_type == 1) {
            /* Prefill done: upcoming = ant if it's a decode event */
            if (tr->ant_type == 1) {
                upcoming_type = 1;
                upcoming_ts   = tr->ant_ts;
                upcoming_cn   = tr->ant_cn;
            }
        } else if (mre_type == 2) {
            /* Decode at completion_number=mre_cn */
            /* upcoming = ant if ant_type==1 and ant_cn > mre_cn */
            if (tr->ant_type == 1 && tr->ant_cn > tr->mre_cn) {
                upcoming_type = 1;
                upcoming_ts   = tr->ant_ts;
                upcoming_cn   = tr->ant_cn;
            }
        }

        /* Fallback for running with no upcoming decode */
        if (in_running && upcoming_type == -1 &&
            (mre_type == 1 || mre_type == 2))
        {
            int next_n = (mre_type == 2) ? tr->mre_cn + 1 : 1;
            int ctx = tr->prompt_len + next_n;
            double dur = w_isolated_decode_time_estimation(
                (double)ctx, (double)ctx, 1.0, (double)fn);
            upcoming_type = 1;
            upcoming_ts   = tr->mre_ts + dur;
            upcoming_cn   = next_n;
        }

        if (upcoming_type == 0 && in_waiting) {
            /* Prefill candidate */
            if (has_fair_uids && w_hm_get(&fair_uids_set, tr->uid) == HT_EMPTY) {
                /* Not fair: set start_deadline = HUGE_VAL */
                if (n_wpd < n_reqs) {
                    strncpy(wpd_rids[n_wpd], tr->rid, W_RID_MAX - 1);
                    wpd_vals[n_wpd] = Py_HUGE_VAL;
                    n_wpd++;
                }
                continue;
            }
            double deadline = upcoming_ts + delta_prefill_s;
            double start_dl = deadline - pooled_prefill_s;
            DLCandidate *c = &prefill_candidates[n_prefill++];
            strncpy(c->rid, tr->rid, W_RID_MAX - 1);
            strncpy(c->uid, tr->uid, W_UID_MAX - 1);
            c->event_type     = 0;
            c->deadline       = deadline;
            c->start_deadline = start_dl;
            c->ant_ts         = upcoming_ts;
            c->ant_cn         = upcoming_cn;
            c->arrival_ts     = tr->arrival_ts;
            if (n_wpd < n_reqs) {
                strncpy(wpd_rids[n_wpd], tr->rid, W_RID_MAX - 1);
                wpd_vals[n_wpd] = start_dl;
                n_wpd++;
            }
        } else if (upcoming_type == 1 && in_running) {
            /* Decode candidate */
            if (has_fair_decode && w_hm_get(&fair_decode_set, tr->uid) == HT_EMPTY) continue;
            double deadline = upcoming_ts + delta_decode_s;
            double start_dl = deadline - pooled_decode_s;
            if (!has_earliest_decode || start_dl < earliest_decode.start_deadline) {
                strncpy(earliest_decode.rid, tr->rid, W_RID_MAX - 1);
                strncpy(earliest_decode.uid, tr->uid, W_UID_MAX - 1);
                earliest_decode.event_type     = 1;
                earliest_decode.deadline       = deadline;
                earliest_decode.start_deadline = start_dl;
                earliest_decode.ant_ts         = upcoming_ts;
                earliest_decode.ant_cn         = upcoming_cn;
                earliest_decode.arrival_ts     = tr->arrival_ts;
                has_earliest_decode = 1;
            }
        }
    }

    /* Sort prefill_candidates by (start_deadline, deadline, arrival_ts) — insertion sort */
    #define PREFILL_CAP 32
    if (n_prefill > 1) {
        /* Use insertion sort */
        for (int i = 1; i < n_prefill; i++) {
            DLCandidate tmp = prefill_candidates[i];
            int j = i - 1;
            while (j >= 0) {
                DLCandidate *prev = &prefill_candidates[j];
                int swap = 0;
                if (prev->start_deadline > tmp.start_deadline) swap = 1;
                else if (prev->start_deadline == tmp.start_deadline) {
                    if (prev->deadline > tmp.deadline) swap = 1;
                    else if (prev->deadline == tmp.deadline) {
                        if (prev->arrival_ts > tmp.arrival_ts) swap = 1;
                    }
                }
                if (!swap) break;
                prefill_candidates[j + 1] = *prev;
                j--;
            }
            prefill_candidates[j + 1] = tmp;
        }
        if (n_prefill > PREFILL_CAP) n_prefill = PREFILL_CAP;
    }

    Py_END_ALLOW_THREADS

    /* Build Python result (GIL held) */
    PyObject *deadline_list = PyList_New(0);
    if (!deadline_list) goto cleanup_and_error;

    /* Earliest decode first */
    if (has_earliest_decode) {
        PyObject *tup = PyTuple_New(7);
        if (!tup) { Py_DECREF(deadline_list); goto cleanup_and_error; }
        PyTuple_SET_ITEM(tup, 0, PyUnicode_FromString(earliest_decode.rid));
        PyTuple_SET_ITEM(tup, 1, PyUnicode_FromString(earliest_decode.uid));
        PyTuple_SET_ITEM(tup, 2, PyUnicode_FromString("decode"));
        PyTuple_SET_ITEM(tup, 3, PyFloat_FromDouble(earliest_decode.deadline));
        PyTuple_SET_ITEM(tup, 4, PyFloat_FromDouble(earliest_decode.start_deadline));
        PyTuple_SET_ITEM(tup, 5, PyFloat_FromDouble(earliest_decode.ant_ts));
        PyTuple_SET_ITEM(tup, 6, PyLong_FromLong(earliest_decode.ant_cn));
        PyList_Append(deadline_list, tup);
        Py_DECREF(tup);
    }

    for (int i = 0; i < n_prefill; i++) {
        DLCandidate *c = &prefill_candidates[i];
        PyObject *tup = PyTuple_New(7);
        if (!tup) { Py_DECREF(deadline_list); goto cleanup_and_error; }
        PyTuple_SET_ITEM(tup, 0, PyUnicode_FromString(c->rid));
        PyTuple_SET_ITEM(tup, 1, PyUnicode_FromString(c->uid));
        PyTuple_SET_ITEM(tup, 2, PyUnicode_FromString("prefill"));
        PyTuple_SET_ITEM(tup, 3, PyFloat_FromDouble(c->deadline));
        PyTuple_SET_ITEM(tup, 4, PyFloat_FromDouble(c->start_deadline));
        PyTuple_SET_ITEM(tup, 5, PyFloat_FromDouble(c->ant_ts));
        PyTuple_SET_ITEM(tup, 6, PyLong_FromLong(c->ant_cn));
        PyList_Append(deadline_list, tup);
        Py_DECREF(tup);
    }

    /* Build waiting_prefill_deadlines dict */
    PyObject *wpd_dict = PyDict_New();
    if (!wpd_dict) { Py_DECREF(deadline_list); goto cleanup_and_error; }
    for (int i = 0; i < n_wpd; i++) {
        PyObject *key = PyUnicode_FromString(wpd_rids[i]);
        PyObject *val;
        if (isinf(wpd_vals[i])) {
            val = PyFloat_FromDouble(Py_HUGE_VAL);
        } else {
            val = PyFloat_FromDouble(wpd_vals[i]);
        }
        PyDict_SetItem(wpd_dict, key, val);
        Py_DECREF(key);
        Py_DECREF(val);
    }

    /* Build ordered_waiting_rids tuple */
    PyObject *ordered_rids = PyTuple_New(n_prefill);
    if (!ordered_rids) { Py_DECREF(deadline_list); Py_DECREF(wpd_dict); goto cleanup_and_error; }
    for (int i = 0; i < n_prefill; i++) {
        PyTuple_SET_ITEM(ordered_rids, i, PyUnicode_FromString(prefill_candidates[i].rid));
    }

    /* Free C arrays */
    if (prefill_candidates) W_FREE(prefill_candidates);
    if (wpd_rids) {
        for (int i = 0; i < n_reqs; i++) W_FREE(wpd_rids[i]);
        W_FREE(wpd_rids);
    }
    if (wpd_vals) W_FREE(wpd_vals);
    w_hm_free(&waiting_set);
    w_hm_free(&running_set);
    if (has_fair_uids) w_hm_free(&fair_uids_set);
    if (has_fair_decode) w_hm_free(&fair_decode_set);

    return PyTuple_Pack(3, deadline_list, wpd_dict, ordered_rids);

cleanup_and_error:
    if (prefill_candidates) W_FREE(prefill_candidates);
    if (wpd_rids) {
        for (int i = 0; i < n_reqs; i++) W_FREE(wpd_rids[i]);
        W_FREE(wpd_rids);
    }
    if (wpd_vals) W_FREE(wpd_vals);
    w_hm_free(&waiting_set);
    w_hm_free(&running_set);
    if (has_fair_uids) w_hm_free(&fair_uids_set);
    if (has_fair_decode) w_hm_free(&fair_decode_set);
    return NULL;
}

/* =========================================================================
 * get_all_req_data() -> list of dicts (for debugging)
 * =========================================================================*/
static PyObject *
CSimulator_get_all_req_data(CSimulatorObject *self, PyObject *args)
{
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    for (int i = 0; i < self->reqs_len; i++) {
        if (!self->reqs[i].alive) continue;
        CTrackedReq *tr = &self->reqs[i];

        PyObject *d = PyDict_New();
        if (!d) { Py_DECREF(result); return NULL; }

        #define SET_STR(k, v)  PyDict_SetItemString(d, k, PyUnicode_FromString(v))
        #define SET_INT(k, v)  PyDict_SetItemString(d, k, PyLong_FromLong(v))
        #define SET_FLT(k, v)  PyDict_SetItemString(d, k, PyFloat_FromDouble(v))

        SET_STR("rid", tr->rid);
        SET_STR("uid", tr->uid);
        SET_FLT("arrival_ts", tr->arrival_ts);
        SET_INT("prompt_len", tr->prompt_len);
        SET_INT("output_len", tr->output_len);
        SET_INT("fill_len", tr->fill_len);
        SET_INT("delta_prefill_us", tr->delta_prefill_us);
        SET_INT("delta_decode_us", tr->delta_decode_us);
        SET_INT("prefill_done", tr->prefill_done);
        SET_INT("decode_count", tr->decode_count);
        SET_INT("is_complete", tr->is_complete);
        SET_INT("mre_type", tr->mre_type);
        SET_FLT("mre_ts", tr->mre_ts);
        SET_INT("mre_cn", tr->mre_cn);
        SET_INT("hist_type", tr->hist_type);
        SET_FLT("hist_ts", tr->hist_ts);
        SET_INT("hist_cn", tr->hist_cn);
        SET_INT("ant_type", tr->ant_type);
        SET_FLT("ant_ts", tr->ant_ts);
        SET_INT("ant_cn", tr->ant_cn);
        SET_FLT("latest_sim_completion_ts", tr->latest_sim_completion_ts);
        SET_INT("alive", tr->alive);

        #undef SET_STR
        #undef SET_INT
        #undef SET_FLT

        PyList_Append(result, d);
        Py_DECREF(d);
    }

    return result;
}

/* =========================================================================
 * Method table
 * =========================================================================*/
static PyMethodDef CSimulator_methods[] = {
    {"set_timeline_writer", (PyCFunction)CSimulator_set_timeline_writer, METH_VARARGS,
     "Update the stored timeline writer reference."},
    {"process_new_request", (PyCFunction)CSimulator_process_new_request, METH_VARARGS,
     "Register a new request with its parameters."},
    {"note_prefill_done", (PyCFunction)CSimulator_note_prefill_done, METH_VARARGS,
     "Note that prefill was done for a request (mutation path)."},
    {"finished_prefill", (PyCFunction)CSimulator_finished_prefill, METH_VARARGS,
     "Record prefill completion for a batch of requests."},
    {"finished_decode", (PyCFunction)CSimulator_finished_decode, METH_VARARGS,
     "Record decode step(s) completion for a batch of requests."},
    {"mark_request_finished", (PyCFunction)CSimulator_mark_request_finished, METH_VARARGS,
     "Mark a request as fully complete and backfill timeline."},
    {"sync_live_users", (PyCFunction)CSimulator_sync_live_users, METH_VARARGS,
     "Synchronize live user state from waiting/running queues."},
    {"rebuild_all_users", (PyCFunction)CSimulator_rebuild_all_users, METH_VARARGS | METH_KEYWORDS,
     "Run isolated simulation rebuild for all (or filtered) users."},
    {"build_deadline_candidates", (PyCFunction)CSimulator_build_deadline_candidates, METH_VARARGS,
     "Build deadline candidate list from current state."},
    {"get_all_req_data", (PyCFunction)CSimulator_get_all_req_data, METH_NOARGS,
     "Return all alive request data as a list of dicts (debugging)."},
    {NULL, NULL, 0, NULL}
};

/* =========================================================================
 * Type definition
 * =========================================================================*/
static PyTypeObject CSimulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_fairinf_worker.CSimulator",
    .tp_basicsize = sizeof(CSimulatorObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)CSimulator_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Stateful C simulator for the fairinf worker thread.",
    .tp_methods   = CSimulator_methods,
    .tp_init      = (initproc)CSimulator_init,
    .tp_new       = PyType_GenericNew,
};

/* =========================================================================
 * Module definition
 * =========================================================================*/
static PyModuleDef _fairinf_worker_module = {
    PyModuleDef_HEAD_INIT,
    "_fairinf_worker",
    "C extension providing a stateful CSimulator for the fairinf worker thread.",
    -1,
    NULL,
};

PyMODINIT_FUNC
PyInit__fairinf_worker(void)
{
    if (PyType_Ready(&CSimulatorType) < 0) return NULL;

    PyObject *m = PyModule_Create(&_fairinf_worker_module);
    if (!m) return NULL;

    Py_INCREF(&CSimulatorType);
    if (PyModule_AddObject(m, "CSimulator", (PyObject *)&CSimulatorType) < 0) {
        Py_DECREF(&CSimulatorType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
