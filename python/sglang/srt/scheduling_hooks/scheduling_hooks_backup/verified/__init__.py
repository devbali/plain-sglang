"""Verified C functions for τ-fair scheduling.

These are wrappers around C code compiled from the .veri.md specs
through the veri-build pipeline (Veri DSL → F* → KaRaMeL → C).

To build the shared library:
    cd scheduling_hooks/verified
    gcc -shared -fPIC -I. -I./internal -o _verified.so \
        wrapper.c TauFairScheduler.c DeadlineModel.c
"""

import ctypes
import os
from pathlib import Path

_HERE = Path(__file__).parent
_SO_PATH = _HERE / "_verified.so"
_LIB = None

def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _SO_PATH.exists():
        raise ImportError(f"_verified.so not built. Run: cd {_HERE} && "
                          f"gcc -shared -fPIC -I. -I./internal -o _verified.so "
                          f"wrapper.c TauFairScheduler.c DeadlineModel.c")
    _LIB = ctypes.CDLL(str(_SO_PATH))
    # Set return types
    _LIB.verified_edf_tier.restype = ctypes.c_int32
    _LIB.verified_first_deadline.restype = ctypes.c_int32
    _LIB.verified_update_headroom.restype = ctypes.c_int32
    _LIB.verified_can_admit.restype = ctypes.c_int32
    return _LIB


def edf_tier(deadline_us, total_kv, fair_share, tau_us, now_us):
    """Return the EDF tier (0=overdue, 1=at-risk, 2=safe)."""
    lib = _load()
    return lib.verified_edf_tier(deadline_us, total_kv, fair_share, tau_us, now_us)


def first_deadline(iso_prefill_us, tau_cache_us, tau_prefill_us, now_us):
    """Compute first-token deadline: d_0 = T_ISO + τ."""
    lib = _load()
    return lib.verified_first_deadline(iso_prefill_us, tau_cache_us, tau_prefill_us, now_us)


def update_headroom(old_h, was_prefill, cost, was_decode, iso, mt, idle):
    """Update accumulated headroom per scheduling pass."""
    lib = _load()
    return lib.verified_update_headroom(old_h, was_prefill, cost, was_decode, iso, mt, idle)


def can_admit(needed, total_kv, evictable, fair_share, reservation, slack, has_space):
    """Check admission gate: within reservation or slack available."""
    lib = _load()
    return bool(lib.verified_can_admit(needed, total_kv, evictable, fair_share, reservation, slack, has_space))


HAS_C_EXTENSION = _SO_PATH.exists()
