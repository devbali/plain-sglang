# cython: language_level=3
"""Cython wrapper for verified τ-fair scheduler C functions.

Compiled from .veri.md specs through Veri DSL → F* → KaRaMeL → C.
"""
cimport _verified_cdef

cdef class EDFOrderingKey:
    """EDF + fair-share ordering key from verified C."""
    cdef _verified_cdef.TauFairScheduler_EDFOrderingKey _c_key

    def __init__(self, int tier, int neg_shortfall, int excess):
        self._c_key.tier = tier
        self._c_key.neg_shortfall = neg_shortfall
        self._c_key.excess = excess

    @property
    def tier(self): return self._c_key.tier
    @property
    def neg_shortfall(self): return self._c_key.neg_shortfall
    @property
    def excess(self): return self._c_key.excess

    def __lt__(self, other):
        return _verified_cdef.TauFairScheduler_edf_compare(
            self._c_key, (<EDFOrderingKey>other)._c_key)


def edf_fairshare_key(int deadline_us, int total_kv, int fair_share,
                       int tau_us, int now_us):
    """Compute the EDF + fair-share ordering key (verified C)."""
    cdef _verified_cdef.TauFairScheduler_EDFOrderingKey key
    key = _verified_cdef.TauFairScheduler_edf_fairshare_key(
        deadline_us, total_kv, fair_share, tau_us, now_us)
    return EDFOrderingKey(key.tier, key.neg_shortfall, key.excess)


def edf_compare(EDFOrderingKey a, EDFOrderingKey b):
    """Return True if a has higher scheduling priority than b (verified C)."""
    return _verified_cdef.TauFairScheduler_edf_compare(a._c_key, b._c_key)


def on_new_request_init_deadline(
    int iso_prefill_time_us, int tau_cache_us, int tau_prefill_us, int now_us
):
    """Compute first-token deadline d_0 = T_ISO_0 + τ (verified C)."""
    return _verified_cdef.TauFairScheduler_on_new_request_init_deadline(
        iso_prefill_time_us, tau_cache_us, tau_prefill_us, now_us)


def prefill_vs_decode_decision(
    list running_deadlines_us, int n_running, bool has_candidate_prefill,
    int prefill_cost_us, int accumulated_headroom_us,
    int delta_decode_mt_us, int tau_us, int now_us,
    bool any_over_quota, bool any_starved, bool max_per_user_active
):
    """Decide prefill vs decode at scheduling pass start (verified C)."""
    cdef int[:] deadlines = array('i', running_deadlines_us[:n_running])
    return _verified_cdef.TauFairScheduler_prefill_vs_decode_decision(
        &deadlines[0], n_running, has_candidate_prefill, prefill_cost_us,
        accumulated_headroom_us, delta_decode_mt_us, tau_us, now_us,
        any_over_quota, any_starved, max_per_user_active)


def sort_prefill_queue(
    list deadlines_us, list uids, list total_kvs, list fair_shares,
    int n, int tau_us, int now_us
):
    """Sort prefill queue by EDF + fair-share priority (verified C).

    Returns a list of indices [0..n) in sorted order.
    """
    cdef int[:] deadlines = array('i', deadlines_us[:n])
    cdef int[:] kvs = array('i', total_kvs[:n])
    cdef int[:] shares = array('i', fair_shares[:n])
    # Call the verified C function
    cdef int[:] result = _verified_cdef.TauFairScheduler_sort_prefill_queue(
        &deadlines[0], &kvs[0], &shares[0], n, tau_us, now_us)
    return list(result)


def update_headroom(
    int old_headroom_us, bool was_prefill, int prefill_cost_us,
    bool was_decode, int delta_decode_iso_us, int delta_decode_mt_us,
    bool was_idle
):
    """Update accumulated headroom after one scheduling pass (verified C)."""
    return _verified_cdef.TauFairScheduler_update_headroom(
        old_headroom_us, was_prefill, prefill_cost_us,
        was_decode, delta_decode_iso_us, delta_decode_mt_us, was_idle)


def can_admit_request(
    int needed_tokens, int total_kv, int evictable_kv,
    int fair_share, int reservation, int global_slack, bool cache_has_space
):
    """Check if a request can be admitted (verified C)."""
    return _verified_cdef.TauFairScheduler_can_admit_request(
        needed_tokens, total_kv, evictable_kv,
        fair_share, reservation, global_slack, cache_has_space)


def compute_global_slack(list total_kvs, int n, int max_kv):
    """Compute global KV cache slack (verified C)."""
    cdef int[:] kvs = array('i', total_kvs[:n])
    return _verified_cdef.TauFairScheduler_compute_global_slack(
        &kvs[0], n, max_kv)


def would_violate_decode_deadlines(
    int prefill_cost_us, int headroom_us,
    list running_deadlines_us, int n_running,
    int delta_decode_mt_us, int now_us
):
    """Check if admitting a prefill would violate decode deadlines (verified C)."""
    cdef int[:] deadlines = array('i', running_deadlines_us[:n_running])
    return _verified_cdef.TauFairScheduler_would_violate_decode_deadlines(
        prefill_cost_us, headroom_us, &deadlines[0], n_running,
        delta_decode_mt_us, now_us)


def warn_overdue_deadlines(list deadlines_us, int n, int now_us):
    """Count requests whose deadlines have passed (verified C)."""
    cdef int[:] deadlines = array('i', deadlines_us[:n])
    return _verified_cdef.TauFairScheduler_warn_overdue_deadlines(
        &deadlines[0], n, now_us)
