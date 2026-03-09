from __future__ import annotations

"""Delta fairness variant with forced-prefill disabled."""

from typing import Dict, List, Optional, Sequence, Tuple

from .delta_fairness_policy import DeltaFairnessPolicy

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder




class MateiDeltaFairPolicy(DeltaFairnessPolicy):
    """Delta fairness policy that never forces prefill requests. 
    Maintains an alternate history of each user's isolated decodes and computes forced decodes"""

    def sorted_waiting_queue(self, waiting_queue: List["Req"]):
        return waiting_queue
    
    def fairinf_prioritize_force_prefill():
        return False
