"""FairInference scheduling policy — EDF + fair-share admission control.

Implements Algorithm 2 from the SOSP 2026 paper (§5.1). The scheduling
hooks enforce τ-fairness:
  - Per-token deadlines derived from isolated execution estimates
  - EDF ordering with 3-tier priority (overdue/at-risk/safe)
  - Fair-share prioritization: well-behaved clients before over-quota
  - Headroom tracking for safe prefill admission
  - Admission gating to prevent decode deadline violations

Wire via --scheduling-policy-path:

    --scheduling-policy-path sglang.srt.scheduling_hooks.fairinf_policy.FairInferenceSchedulingPolicy
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.scheduling_hooks import NoOpSchedulingPolicy

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verified C extension (optional — falls back to pure Python)
# ---------------------------------------------------------------------------
_C_VERIFIED = False
_c_tier = _c_deadline = _c_headroom = _c_admit = None

try:
    from sglang.srt.scheduling_hooks.verified import (
        edf_tier as _c_tier,
        first_deadline as _c_deadline,
        update_headroom as _c_headroom,
        can_admit as _c_admit,
        HAS_C_EXTENSION,
    )
    _C_VERIFIED = HAS_C_EXTENSION
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pure Python implementations (mirrors verified C spec)
# ---------------------------------------------------------------------------

class _EDFKey:
    """EDF + fair-share ordering key (matches spec `edf_fairshare_key`)."""
    __slots__ = ('tier', 'neg_shortfall', 'excess')

    def __init__(self, deadline_us: int, total_kv: int, fair_share: int,
                 tau_us: int, now_us: int):
        # Tier classification (§5.1)
        slack = deadline_us - now_us
        if slack <= 0:
            self.tier = 0  # overdue
        elif slack < tau_us:
            self.tier = 1  # at-risk
        else:
            self.tier = 2  # safe
        self.neg_shortfall = -max(0, fair_share - total_kv)
        self.excess = max(0, total_kv - fair_share)

    def compare(self, other: _EDFKey) -> bool:
        """Return True if self has higher scheduling priority than other."""
        if self.tier != other.tier:
            return self.tier < other.tier
        if self.neg_shortfall != other.neg_shortfall:
            return self.neg_shortfall < other.neg_shortfall
        return self.excess < other.excess


def _edf_key(deadline_us: int, total_kv: int, fair_share: int,
             tau_us: int, now_us: int) -> _EDFKey:
    if _C_VERIFIED:
        try:
            tier = _c_tier(deadline_us, total_kv, fair_share, tau_us, now_us)
            # Construct key from verified tier computation
            key = _EDFKey.__new__(_EDFKey)
            key.tier = tier
            key.neg_shortfall = -max(0, fair_share - total_kv)
            key.excess = max(0, total_kv - fair_share)
            return key
        except Exception:
            pass
    return _EDFKey(deadline_us, total_kv, fair_share, tau_us, now_us)


# ---------------------------------------------------------------------------
# FairInferenceSchedulingPolicy
# ---------------------------------------------------------------------------

class FairInferenceSchedulingPolicy(NoOpSchedulingPolicy):
    """FairInference policy that bounds token-level latency using EDF scheduling.

    Matches Algorithm 2 from the paper.  Uses the verified C extension when
    available; falls back to identical pure-Python logic otherwise.
    """

    def __init__(self, n: int = 4, tau_us: int = 1_000_000,
                 tau_cache_us: int = 800_000):
        super().__init__()
        self.n = n                       # number of clients (|C|)
        self.tau_us = tau_us             # total delay tolerance (μs)
        self.tau_cache_us = tau_cache_us # cache delay tolerance (μs)
        self.tau_prefill_us = tau_us - tau_cache_us  # prefill delay (μs)

        # Per-user state
        self._user_kv: Dict[str, int] = {}        # u_i — KV tokens
        self._user_evictable_kv: Dict[str, int] = {}  # evictable KV tokens
        self._fair_share: int = 0                  # f_i (computed at runtime)

        # Headroom tracking (§4.1, Hook 6)
        self._headroom_us: int = 0

        # Deadlines for running decode requests (UID → deadline_us)
        self._running_deadlines: Dict[str, int] = {}

        # Pending prefill queue deadlines
        self._pending_deadlines: Dict[str, int] = {}

    # ═══════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════

    def on_new_request(self, req: "Req") -> None:
        """Hook 1: Initialize the first-token deadline (Algorithm 2 step 2a).

        deadline = now + iso_prefill_time + tau_prefill + tau_cache
        """
        uid = getattr(req, 'uid', None) or 'default'
        now_us = int(time.time() * 1_000_000)

        # Estimate isolated prefill time (simplified — uses inputs as estimate)
        prompt_tokens = len(req.origin_input_text) if req.origin_input_text else 0
        iso_prefill_us = max(1, prompt_tokens) * 100  # rough estimate

        if _C_VERIFIED:
            try:
                deadline = _c_deadline(iso_prefill_us, self.tau_cache_us,
                                       self.tau_prefill_us, now_us)
            except Exception:
                deadline = now_us + iso_prefill_us + self.tau_cache_us + self.tau_prefill_us
        else:
            deadline = now_us + iso_prefill_us + self.tau_cache_us + self.tau_prefill_us

        self._pending_deadlines[uid] = deadline

    # ═══════════════════════════════════════════════════════════
    # Scheduling decisions
    # ═══════════════════════════════════════════════════════════

    def on_schedule_prefill(
        self,
        waiting_queue: "List[Req]",
        running_batch: "ScheduleBatch",
        prefill_adder: "PrefillAdder",
    ) -> Optional["List[Req]"]:
        """Hook 3: Sort prefill queue by EDF + fair-share (Algorithm 2 step 2d.ii)."""
        if not waiting_queue:
            return None

        now_us = int(time.time() * 1_000_000)
        fair_share = self._fair_share_in_tokens()

        def sort_key(req):
            uid = getattr(req, 'uid', None) or 'default'
            deadline = self._pending_deadlines.get(uid, now_us + self.tau_us)
            kv = self._user_kv.get(uid, 0)
            return _edf_key(deadline, kv, fair_share, self.tau_us, now_us)

        return sorted(waiting_queue, key=sort_key,
                      reverse=True)  # reverse because __lt__ means lower priority

    def on_prefill_vs_decode_decision(
        self,
        waiting_queue: "List[Req]",
        running_batch: "ScheduleBatch",
        new_prefill_batch: Optional["ScheduleBatch"],
    ) -> Optional[str]:
        """Hook 2: Decide prefill vs decode (Algorithm 2 step 2d.i)."""
        now_us = int(time.time() * 1_000_000)
        has_prefill = new_prefill_batch is not None
        n_running = len(running_batch.reqs) if running_batch else 0

        # Always decode if no prefill candidate
        if not has_prefill:
            return None  # use default (prefill-first)

        # Check for overdue running decodes
        any_overdue = False
        for req in running_batch.reqs:
            uid = getattr(req, 'uid', None) or 'default'
            deadline = self._running_deadlines.get(uid, 0)
            if deadline > 0 and deadline < now_us:
                any_overdue = True
                break

        # Check for starved users (below fair share, have pending work)
        fair_share = self._fair_share_in_tokens()
        any_starved = False
        for req in waiting_queue[:10]:
            uid = getattr(req, 'uid', None) or 'default'
            kv = self._user_kv.get(uid, 0)
            if kv <= fair_share:
                any_starved = True
                break

        # Compute prefill cost estimate
        prefill_tokens = sum(
            len(r.origin_input_text) for r in (new_prefill_batch.reqs or [])
            if hasattr(r, 'origin_input_text') and r.origin_input_text
        ) or 1000
        prefill_cost_us = prefill_tokens * 50  # rough estimate

        # Safety check: would this prefill violate decode deadlines?
        running_deadlines = [
            self._running_deadlines.get(getattr(r, 'uid', 'default'), 0)
            for r in running_batch.reqs
        ] if running_batch else []
        delta_decode_mt_us = 50_000  # rough estimate: 50ms per decode step

        would_violate = self._py_would_violate(
                prefill_cost_us, running_deadlines, delta_decode_mt_us, now_us)

        # Decision logic (matching paper Algorithm 2 step 2d.i)
        if any_overdue or would_violate:
            return "decode"
        if any_starved and self._headroom_us < prefill_cost_us:
            return "decode"
        # Headroom check
        if self._headroom_us >= prefill_cost_us or not any_starved:
            return "prefill"
        return "decode"

    # ═══════════════════════════════════════════════════════════
    # Batch dispatch tracking
    # ═══════════════════════════════════════════════════════════

    def on_prefill_decision(self, batch: "ScheduleBatch") -> None:
        """Hook 4: Track prefill admission for fairness reporting."""
        if batch is None:
            return
        for req in batch.reqs:
            uid = getattr(req, 'uid', None) or 'default'
            self._user_kv[uid] = self._user_kv.get(uid, 0) + 1

    def on_decode_decision(self, batch: "ScheduleBatch") -> None:
        """Hook 5: Track decode progress and update running deadlines."""
        if batch is None:
            return
        now_us = int(time.time() * 1_000_000)
        for req in batch.reqs:
            uid = getattr(req, 'uid', None) or 'default'
            # Update KV tracking
            self._user_kv[uid] = self._user_kv.get(uid, 0) + 1
            # Update running deadline (advances per decode step)
            self._running_deadlines[uid] = now_us + 50_000  # next step

    # ═══════════════════════════════════════════════════════════
    # End of pass
    # ═══════════════════════════════════════════════════════════

    def on_end_of_scheduler_pass(self, batch: Optional["ScheduleBatch"]) -> None:
        """Hook 6: Update headroom and reconcile state.

        Headroom accumulates when decodes are faster than isolated estimate,
        is consumed by prefills.
        """
        if batch is None or batch.is_empty:
            return  # idle pass — headroom unchanged

        was_prefill = batch.is_prefill
        was_decode = not was_prefill
        prefill_cost = 0
        delta_iso = 80_000   # ~80ms per decode batch in isolation
        delta_mt = 50_000    # ~50ms per decode batch in MT

        if was_prefill:
            prefill_cost = sum(
                len(r.origin_input_text) * 50 for r in batch.reqs
                if hasattr(r, 'origin_input_text') and r.origin_input_text
            ) or 50000

        if _C_VERIFIED:
            try:
                self._headroom_us = _c_headroom(
                    self._headroom_us, was_prefill, prefill_cost,
                    was_decode, delta_iso, delta_mt, False)
            except Exception:
                self._py_update_headroom(was_prefill, prefill_cost,
                                         was_decode, delta_iso, delta_mt)
        else:
            self._py_update_headroom(was_prefill, prefill_cost,
                                     was_decode, delta_iso, delta_mt)

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _fair_share_in_tokens(self) -> int:
        """Compute fair share: total KV capacity / number of clients."""
        if self.n <= 0:
            return 0
        # Approximate from memory fraction and model capacity
        # f_i = M / |C| where M ≈ 328,773 tokens on A100 80GB
        return 328773 // self.n

    def _py_update_headroom(self, was_prefill: bool, prefill_cost: int,
                            was_decode: bool, delta_iso: int, delta_mt: int):
        """Update headroom — pure Python (matches spec `update_headroom`)."""
        if was_decode and not was_prefill:
            self._headroom_us = max(0, self._headroom_us + delta_iso - delta_mt)
        elif was_prefill and not was_decode:
            self._headroom_us = max(0, self._headroom_us - prefill_cost)

    def _py_would_violate(self, prefill_cost: int,
                          running_deadlines: List[int],
                          delta_decode_mt: int, now_us: int) -> bool:
        """Check if admitting a prefill would violate decode deadlines."""
        effective_cost = prefill_cost - self._headroom_us
        for deadline in running_deadlines:
            if deadline > 0 and deadline - now_us < delta_decode_mt + effective_cost:
                return True
        return False
