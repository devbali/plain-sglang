from __future__ import annotations

"""Fairness helpers for static per-user KV cache reservations."""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.global_config import global_config

from .no_fairness_policy import NoFairnessPolicy

logger = logging.getLogger(__name__)

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder


class StaticFairnessPolicy(NoFairnessPolicy):
    """Implements the control flow introduced for static per-user allocations."""

    def _has_static_limit(self, tree_cache: Optional["BasePrefixCache"]) -> bool:
        return tree_cache is not None and getattr(tree_cache, "static_max_per_user", None) is not None

    # ---- Request admission -------------------------------------------------
    def init_next_round_input_control(
        self,
        tree_cache: Optional["BasePrefixCache"],
        req: "Req",
        *,
        fair: bool = False,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        if not self._has_static_limit(tree_cache):
            return None

        if tree_cache.reject_based_on_static_limit(req.uid, req.extend_input_len + extra_tokens):
            return "rejected"
        return None

    def add_prefill_request_control(
        self,
        tree_cache: Optional["BasePrefixCache"],
        req: "Req",
        *,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        if not self._has_static_limit(tree_cache):
            return None

        if tree_cache.reject_based_on_static_limit(req.uid, req.extend_input_len + extra_tokens):
            return "rejected"
        return None

    # ---- Allocation helpers -----------------------------------------------
    def requires_per_user_allocation(self, tree_cache: Optional["BasePrefixCache"]) -> bool:
        return self._has_static_limit(tree_cache)

    def alloc_token_slots(
        self,
        tree_cache: Optional["BasePrefixCache"],
        token_to_kv_pool: "BaseTokenToKVPool",
        num_tokens: int,
        *,
        user_id: Optional[str] = None,
        evict_only_force: bool = False,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> Optional[int]:
        if not self._has_static_limit(tree_cache):
            return super().alloc_token_slots(
                tree_cache,
                token_to_kv_pool,
                num_tokens,
                user_id=user_id,
                evict_only_force=evict_only_force,
                requesting_users=requesting_users,
            )

        out_cache_loc = None if evict_only_force else token_to_kv_pool.alloc(num_tokens)
        logger.debug(
            "StaticFairnessPolicy: alloc %s tokens for %s (force=%s, initial_loc=%s)",
            num_tokens,
            user_id,
            evict_only_force,
            out_cache_loc,
        )

        if evict_only_force or (out_cache_loc is None and not evict_only_force):
            if user_id is None:
                raise RuntimeError("Static fairness allocation requires a user id for evictions.")
            tree_cache.evict_from_user(num_tokens, token_to_kv_pool.free, user_id)
            if not evict_only_force:
                out_cache_loc = token_to_kv_pool.alloc(num_tokens)

        if out_cache_loc is None and not evict_only_force:
            logger.error("Prefill out of memory even after static eviction.")
            raise RuntimeError("Prefill out of memory under static fairness policy.")

        return None if evict_only_force else out_cache_loc

    def prepare_for_extend_allocation(
        self,
        batch: "ScheduleBatch",
        extend_num_tokens: int,
        *,
        running_batch: Optional["ScheduleBatch"] = None,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[int], List["Req"]]:
        if not self.requires_per_user_allocation(batch.tree_cache):
            return super().prepare_for_extend_allocation(
                batch,
                extend_num_tokens,
                running_batch=running_batch,
                requesting_users=requesting_users,
            )

        out_cache_loc = None
        removed_requests: List["Req"] = []

        while out_cache_loc is None:
            eviction_necessary = batch.token_to_kv_pool.available_size() < extend_num_tokens
            if eviction_necessary:
                for req in batch.reqs:
                    num_tokens = len(req.fill_ids[len(req.prefix_indices) :])
                    self.alloc_token_slots(
                        batch.tree_cache,
                        batch.token_to_kv_pool,
                        num_tokens,
                        user_id=req.uid,
                        evict_only_force=True,
                    )

            out_cache_loc = batch.token_to_kv_pool.alloc(extend_num_tokens)
            if out_cache_loc is None and running_batch is not None:
                removed, _ = running_batch.retract_decode(extend_num_tokens)
                removed_requests += removed
            else:
                break

        if out_cache_loc is None:
            raise RuntimeError(
                "Allocation failed after evicting for each user in static fairness policy."
            )

        return out_cache_loc, removed_requests

    # ---- Decode helpers ----------------------------------------------------
    def check_decode_memory(self, batch: "ScheduleBatch") -> bool:
        if not self.requires_per_user_allocation(batch.tree_cache):
            return super().check_decode_memory(batch)

        bs = batch.batch_size()
        tree_cache = batch.tree_cache
        assert tree_cache is not None

        eviction_necessary = any(
            tree_cache.reject_based_on_static_limit(req.uid, 1) for req in batch.reqs
        )

        if batch.token_to_kv_pool.available_size() >= bs and not eviction_necessary:
            return True

        logger.info(
            "StaticFairnessPolicy: decode OOM (available=%s, needed=%s), evicting by user.",
            batch.token_to_kv_pool.available_size(),
            bs,
        )
        for req in batch.reqs:
            tree_cache.evict_from_user(1, batch.token_to_kv_pool.free, req.uid)

        if batch.token_to_kv_pool.available_size() >= bs:
            logger.info(
                "StaticFairnessPolicy: eviction successful (available=%s).",
                batch.token_to_kv_pool.available_size(),
            )
            return True

        return False

    def get_retract_order(self, batch: "ScheduleBatch") -> List[int]:
        if not self.requires_per_user_allocation(batch.tree_cache):
            return super().get_retract_order(batch)

        tree_cache = batch.tree_cache
        assert tree_cache is not None
        prioritized: List[int] = []
        for i, req in enumerate(batch.reqs):
            if tree_cache.reject_based_on_static_limit(req.uid, global_config.retract_decode_steps):
                prioritized.append(i)
        for i in range(len(batch.reqs)):
            if i not in prioritized:
                prioritized.append(i)
        return prioritized

    def alloc_decode_output_slots(self, batch: "ScheduleBatch"):
        if not self.requires_per_user_allocation(batch.tree_cache):
            return super().alloc_decode_output_slots(batch)

        bs = batch.batch_size()
        eviction_necessary = batch.token_to_kv_pool.available_size() < bs
        if eviction_necessary:
            for req in batch.reqs:
                self.alloc_token_slots(
                    batch.tree_cache,
                    batch.token_to_kv_pool,
                    1,
                    user_id=req.uid,
                    evict_only_force=True,
                )

        out_cache_loc = batch.token_to_kv_pool.alloc(bs)
        if out_cache_loc is None:
            raise RuntimeError(
                "Failed to allocate decode slots under static fairness policy."
            )
        return out_cache_loc
