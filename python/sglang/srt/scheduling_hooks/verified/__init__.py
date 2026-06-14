"""Verified C functions for τ-fair scheduling.

These functions were compiled from the .veri.md specs through the
veri-build pipeline (Veri DSL → F* → KaRaMeL → C). They implement
the paper's Algorithm 2: τ-fair scheduler with EDF ordering,
headroom tracking, admission gating, and deadline safety checks.
"""

# Try to import the Cython extension; fall back to pure Python
try:
    from ._verified import (
        edf_fairshare_key,
        edf_compare,
        on_new_request_init_deadline,
        prefill_vs_decode_decision,
        sort_prefill_queue,
        update_headroom,
        can_admit_request,
        compute_global_slack,
        would_violate_decode_deadlines,
        warn_overdue_deadlines,
    )
    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False
