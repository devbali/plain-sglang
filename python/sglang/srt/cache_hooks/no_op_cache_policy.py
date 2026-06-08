from __future__ import annotations

"""Minimal no-op cache hook policy.

Subclass NoOpCachePolicy and override hook methods to inject per-user
tracking or fairness logic into the RadixCache without modifying the
core cache implementation directly.

Design parallels ``scheduling_hooks/no_op_policy.py`` for the scheduler.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


class NoOpCachePolicy:
    """Default pass-through cache policy — all hooks are no-ops.

    Wire one instance into RadixCache via ``cache.cache_hooks_policy``.
    Override specific methods in a subclass; the cache calls each hook at
    the appropriate point in its lifecycle.
    """

    # ── lifecycle ────────────────────────────────────────────────

    def on_cache_reset(self, cache: "RadixCache") -> None:
        """Called when the cache is reset (e.g. server restart)."""

    # ── insert ───────────────────────────────────────────────────

    def on_insert(self, node: "TreeNode", owner_uid: Optional[str]) -> None:
        """Called after a tree node's value is set during insert.

        *node.value* and *node.key* are populated. Use this to track
        per-user token counts via the node's owner and key length.
        """

    # ── eviction ─────────────────────────────────────────────────

    def can_evict_node(self, node: "TreeNode") -> bool:
        """Called for each node *considered* for eviction.

        Return ``False`` to protect the node from being evicted
        (skip to the next candidate). Default always returns ``True``.
        """
        return True

    def on_evict(self, node: "TreeNode") -> None:
        """Called after a node has been evicted (before deletion).

        The node's owner and key length are still available.
        Use this to decrement per-user token counters.
        """

    # ── lock-ref changes ─────────────────────────────────────────

    def on_lock_ref_inc(self, node: "TreeNode") -> None:
        """Called when a node transitions from lock_ref==0 to lock_ref==1.

        The node is becoming *protected* (unevictable). Use this to move
        tokens from the evictable→unevictable per-user bucket.
        """

    def on_lock_ref_dec(self, node: "TreeNode") -> None:
        """Called when a node transitions from lock_ref==1 to lock_ref==0.

        The node is becoming *unprotected* (evictable). Use this to move
        tokens from the unevictable→evictable per-user bucket.
        """

    # ── decode KV ────────────────────────────────────────────────

    def on_decode_kv_alloc(self, uid: str, num_tokens: int) -> None:
        """Called when *num_tokens* decode output KV slots are allocated for *uid*.

        These live outside the radix tree but consume the same memory pool.
        Track them for fairness decisions.
        """

    def on_decode_kv_free(self, uid: str, num_tokens: int) -> None:
        """Called when *num_tokens* decode output KV slots are freed for *uid*."""

    # ── admission checks ─────────────────────────────────────────

    def can_admit_request(
        self,
        uid: Optional[str],
        num_required_tokens: int,
        *,
        cache: Optional["RadixCache"] = None,
    ) -> bool:
        """Called before admitting a new request into the cache/prefill batch.

        *num_required_tokens* is the estimated number of new tokens this
        request will need (excluding already-cached prefix tokens).

        Return ``False`` to reject the request (it stays in the waiting queue).

        Default always returns ``True``.
        """
        return True

    # ── retraction ───────────────────────────────────────────────

    def get_retract_priority(
        self,
        req: "Req",
        *,
        cache: Optional["RadixCache"] = None,
    ) -> float:
        """Called for each decode request being considered for retraction.

        Return a priority value. Higher values are retracted first.

        Used to implement per-user eviction in static partition mode:
        over-quota users get higher retraction priority.

        Default returns 0.0 (all equal priority).
        """
        return 0.0

    # ── queries (for use by scheduler hooks) ─────────────────────

    def get_user_total_tokens(self, uid: str) -> int:
        """Return total (cache + decode) tokens for *uid*, or 0."""
        return 0

    def get_user_evictable_tokens(self, uid: str) -> int:
        """Return evictable cache tokens for *uid*, or 0."""
        return 0

    def get_all_user_tokens(self) -> dict[str, int]:
        """Return {uid: total_tokens} snapshot for all tracked users."""
        return {}
