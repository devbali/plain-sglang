"""Static partition policies for cache and scheduling.

Static partition divides KV cache memory evenly among N users.
Each user gets ``total_memory // N`` tokens.  Once a user hits their
share they cannot admit new requests, and only their excess tokens are
eligible for eviction.

Two policy classes:

- ``StaticPartitionCachePolicy`` — wire into ``RadixCache.cache_hooks_policy``
- ``StaticPartitionSchedulingPolicy`` — wire into ``Scheduler.scheduling_hooks_policy``

Usage (command-line)::

    --scheduling-policy-path sglang.srt.cache_hooks.static_partition_policy.StaticPartitionSchedulingPolicy
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.cache_hooks import NoOpCachePolicy
from sglang.srt.scheduling_hooks import NoOpSchedulingPolicy

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


class StaticPartitionCachePolicy(NoOpCachePolicy):
    """Cache policy that enforces per-user static partition.

    Key invariants:
    - Each of N users gets ``max_per_user`` tokens.
    - Eviction only targets users who are over their quota.
    - New requests from at-quota users are rejected.
    - Retraction prioritizes over-quota users.

    ``max_per_user`` is computed as ``total_memory // n`` unless explicitly set.
    """

    def __init__(
        self,
        n: int,
        total_memory_tokens: int = 0,
        max_per_user: Optional[int] = None,
    ):
        super().__init__()
        self.n = n
        self.max_per_user = max_per_user or (total_memory_tokens // n if n > 0 else 0)
        self._lock = threading.Lock()
        self._initialized = total_memory_tokens > 0 or max_per_user is not None

        # uid -> total tokens in cache (prefix + decode)
        self._user_total: Dict[str, int] = {}
        # uid -> evictable tokens in cache (unlocked tree nodes)
        self._user_evictable: Dict[str, int] = {}

    # ── lifecycle ───────────────────────────────────────────────

    def on_cache_reset(self, cache: "RadixCache") -> None:
        with self._lock:
            self._user_total.clear()
            self._user_evictable.clear()

    # ── insert ──────────────────────────────────────────────────

    def on_insert(self, node: "TreeNode", owner_uid: Optional[str]) -> None:
        uid = owner_uid
        if not uid:
            return
        if node.value is None or not hasattr(node.value, "__len__"):
            return
        try:
            num = len(node.value)
        except Exception:
            return
        with self._lock:
            self._user_total[uid] = self._user_total.get(uid, 0) + num
            self._user_evictable[uid] = self._user_evictable.get(uid, 0) + num

    # ── eviction ────────────────────────────────────────────────

    def can_evict_node(self, node: "TreeNode") -> bool:
        """Only evict nodes belonging to over-quota users."""
        uid = node.owner
        if not uid:
            # Untagged nodes: always evictable (fallback)
            return True
        with self._lock:
            total = self._user_total.get(uid, 0)
        return total > self.max_per_user

    def on_evict(self, node: "TreeNode") -> None:
        uid = node.owner
        if not uid:
            return
        num = len(node.value) if node.value is not None else 0
        with self._lock:
            self._user_total[uid] = max(0, self._user_total.get(uid, 0) - num)
            self._user_evictable[uid] = max(
                0, self._user_evictable.get(uid, 0) - num
            )

    # ── lock-ref changes ────────────────────────────────────────

    def on_lock_ref_inc(self, node: "TreeNode") -> None:
        """Node becomes protected: move from evictable to unevictable."""
        uid = node.owner
        if not uid:
            return
        num = len(node.key) if hasattr(node, "key") and node.key is not None else 0
        with self._lock:
            self._user_evictable[uid] = max(
                0, self._user_evictable.get(uid, 0) - num
            )

    def on_lock_ref_dec(self, node: "TreeNode") -> None:
        """Node becomes unprotected: move from unevictable to evictable."""
        uid = node.owner
        if not uid:
            return
        num = len(node.key) if hasattr(node, "key") and node.key is not None else 0
        with self._lock:
            self._user_evictable[uid] = self._user_evictable.get(uid, 0) + num

    # ── decode KV ───────────────────────────────────────────────

    def on_decode_kv_alloc(self, uid: str, num_tokens: int) -> None:
        if not uid or num_tokens <= 0:
            return
        with self._lock:
            self._user_total[uid] = self._user_total.get(uid, 0) + num_tokens

    def on_decode_kv_free(self, uid: str, num_tokens: int) -> None:
        if not uid or num_tokens <= 0:
            return
        with self._lock:
            self._user_total[uid] = max(0, self._user_total.get(uid, 0) - num_tokens)

    # ── admission ───────────────────────────────────────────────

    def _lazy_init_from_cache(self, cache: Optional["RadixCache"] = None) -> None:
        """Lazy-init max_per_user from live tree_cache capacity.

        Called on first admission check so we can use the actual KV pool size
        instead of relying on constructor-time estimates.
        """
        if self._initialized or cache is None:
            return
        allocator = getattr(cache, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return
        size = getattr(allocator, "size", 0)
        page_size = getattr(allocator, "page_size", 1)
        if size <= 0:
            return
        with self._lock:
            self.max_per_user = (size * page_size) // self.n if self.n > 0 else 0
            self._initialized = True

    def can_admit_request(
        self,
        uid: Optional[str],
        num_required_tokens: int,
        *,
        cache: Optional["RadixCache"] = None,
    ) -> bool:
        """Reject if user would exceed their partition.

        Uses unevictable tokens: the prefix cache can be self-evicted.
        Lazily initialises max_per_user from the live KV cache on first call.
        """
        self._lazy_init_from_cache(cache)
        if not uid or self.max_per_user <= 0:
            return True
        with self._lock:
            total = self._user_total.get(uid, 0)
            evictable = self._user_evictable.get(uid, 0)
        unevictable = total - evictable
        return (unevictable + num_required_tokens) <= self.max_per_user

    # ── retraction ──────────────────────────────────────────────

    def get_retract_priority(
        self,
        req: "Req",
        *,
        cache: Optional["RadixCache"] = None,
    ) -> float:
        """Over-quota = retract first (higher -> retracted first)."""
        uid = getattr(req, "uid", None)
        if not uid:
            return 0.0
        with self._lock:
            total = self._user_total.get(uid, 0)
        excess = total - self.max_per_user
        return max(0.0, float(excess))

    # ── queries ─────────────────────────────────────────────────

    def get_user_total_tokens(self, uid: str) -> int:
        with self._lock:
            return self._user_total.get(uid, 0)

    def get_user_evictable_tokens(self, uid: str) -> int:
        with self._lock:
            return self._user_evictable.get(uid, 0)

    def get_all_user_tokens(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._user_total)


class StaticPartitionSchedulingPolicy(NoOpSchedulingPolicy):
    """Scheduler policy for static partition fairness.

    Uses a ``StaticPartitionCachePolicy`` instance for user-token queries
    to make admission and reordering decisions.

    Creates and exposes ``self.cache_policy`` — the scheduler automatically
    wires it into ``tree_cache.cache_hooks_policy`` during init.

    To use, pass the class path via ``--scheduling-policy-path``::

        --scheduling-policy-path sglang.srt.cache_hooks.static_partition_policy.StaticPartitionSchedulingPolicy
    """

    def __init__(self, n: int = 4):
        super().__init__()
        # Cache policy created here; scheduler init wires it to tree_cache.
        # max_per_user is lazily initialised on first can_admit_request call.
        self.cache_policy = StaticPartitionCachePolicy(n=n, total_memory_tokens=0)

    def on_schedule_prefill(
        self,
        waiting_queue: "List[Req]",
        running_batch: "ScheduleBatch",
        prefill_adder: "PrefillAdder",
    ) -> Optional["List[Req]"]:
        """Reorder: over-quota users go last, under-quota users first.

        No-op until the cache policy has been lazily initialized
        (max_per_user is still 0).
        """
        cp = self.cache_policy
        if cp.max_per_user <= 0:
            return None  # not yet initialized — keep default order

        def priority_key(req):
            uid = getattr(req, "uid", None) or ""
            total = cp.get_user_total_tokens(uid)
            allocatable = cp.max_per_user - total
            # higher allocatable space = higher scheduling priority
            return -allocatable

        return sorted(waiting_queue, key=priority_key)

    def on_prefill_vs_decode_decision(
        self,
        waiting_queue: "List[Req]",
        running_batch: "ScheduleBatch",
        new_prefill_batch: Optional["ScheduleBatch"],
    ) -> Optional[str]:
        """Force decode if running users are over quota to free their space.

        No-op until the cache policy has been lazily initialized.
        """
        cp = self.cache_policy
        if cp.max_per_user <= 0:
            return None  # not yet initialized — use default decision
        for req in running_batch.reqs:
            uid = getattr(req, "uid", None)
            if uid:
                total = cp.get_user_total_tokens(uid)
                if total > cp.max_per_user:
                    return "decode"
        return None
