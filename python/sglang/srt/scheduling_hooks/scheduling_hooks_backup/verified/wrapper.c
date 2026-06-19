/* Thin C wrapper around verified KaRaMeL C functions.
 *
 * Exposes the core scheduling functions with plain int32_t interfaces
 * that can be called from Python via ctypes (no Cython needed).
 *
 * Compile:
 *   gcc -shared -fPIC -I. -I./internal -o verified_sched.so wrapper.c TauFairScheduler.c DeadlineModel.c
 */

#include "TauFairScheduler.h"
#include "DeadlineModel.h"
#include <stdint.h>

/* ── EDF Ordering Key ─────────────────────────────────────────── */

int32_t verified_edf_tier(int32_t deadline_us, int32_t total_kv,
                          int32_t fair_share, int32_t tau_us, int32_t now_us) {
    return TauFairScheduler_edf_fairshare_key(deadline_us, total_kv,
                                              fair_share, tau_us, now_us).tier;
}

/* ── First-Token Deadline ─────────────────────────────────────── */

int32_t verified_first_deadline(int32_t iso_prefill_us, int32_t tau_cache_us,
                                int32_t tau_prefill_us, int32_t now_us) {
    return TauFairScheduler_on_new_request_init_deadline(
        iso_prefill_us, tau_cache_us, tau_prefill_us, now_us);
}

/* ── Headroom Update ──────────────────────────────────────────── */

int32_t verified_update_headroom(int32_t old_h, int32_t was_prefill,
                                 int32_t cost, int32_t was_decode,
                                 int32_t iso, int32_t mt, int32_t idle) {
    return TauFairScheduler_update_headroom(
        old_h, (bool)was_prefill, cost, (bool)was_decode, iso, mt, (bool)idle);
}

/* ── Admission ────────────────────────────────────────────────── */

int32_t verified_can_admit(int32_t needed, int32_t total_kv,
                           int32_t evictable_kv, int32_t fair_share,
                           int32_t reservation, int32_t slack,
                           int32_t has_space) {
    return TauFairScheduler_can_admit_request(
        needed, total_kv, evictable_kv, fair_share,
        reservation, slack, (bool)has_space);
}
