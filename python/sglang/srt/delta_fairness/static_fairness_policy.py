from __future__ import annotations

"""Fairness helpers for static per-user KV cache reservations."""

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .no_fairness_policy import NoFairnessPolicy

logger = logging.getLogger(__name__)
PREFILL_PROTECTED_HEADROOM_FRACTION = 0.9

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder


class StaticFairnessPolicy(NoFairnessPolicy):
    """Implements the control flow introduced for static per-user allocations."""

    def __init__(
        self,
        tree_cache: Optional["BasePrefixCache"] = None,
        *,
        static_reservation_n: Optional[int] = None,
    ):
        super().__init__(tree_cache=tree_cache)
        self.static_reservation_n = static_reservation_n

    def _has_static_limit(self) -> bool:
        tree_cache = self.tree_cache
        return tree_cache is not None and getattr(tree_cache, "static_max_per_user", None) is not None

    def _decode_token_needs_by_user(self, batch: "ScheduleBatch") -> Dict[str, int]:
        return dict(Counter(req.uid for req in batch.reqs))

    def _users_exceeding_static_limit_for_decode(
        self,
        batch: "ScheduleBatch",
    ) -> Dict[str, int]:
        tree_cache = self.tree_cache
        assert tree_cache is not None

        uncached_running_tokens: Dict[str, int] = {}
        if getattr(batch, "seq_lens", None) is not None:
            seq_lens_cpu = batch.seq_lens.cpu().tolist()
            for i, req in enumerate(batch.reqs):
                uncached_running_tokens[req.uid] = uncached_running_tokens.get(req.uid, 0) + max(
                    0, int(seq_lens_cpu[i]) - len(req.prefix_indices)
                )

        overages: Dict[str, int] = {}
        for user_id, needed_tokens in self._decode_token_needs_by_user(batch).items():
            current_tokens = (
                tree_cache.total_user_counters.get_tokens(user_id)
                + uncached_running_tokens.get(user_id, 0)
            )
            overage = current_tokens + needed_tokens - tree_cache.static_max_per_user
            if overage > 0:
                overages[user_id] = overage
        return overages

    def _decode_usage_and_slack_by_user(
        self,
        batch: "ScheduleBatch",
    ) -> Dict[str, Tuple[int, int, int]]:
        tree_cache = self.tree_cache
        assert tree_cache is not None

        uncached_running_tokens: Dict[str, int] = {}
        if getattr(batch, "seq_lens", None) is not None:
            seq_lens_cpu = batch.seq_lens.cpu().tolist()
            for i, req in enumerate(batch.reqs):
                uncached_running_tokens[req.uid] = uncached_running_tokens.get(req.uid, 0) + max(
                    0, int(seq_lens_cpu[i]) - len(req.prefix_indices)
                )

        usage_and_slack: Dict[str, Tuple[int, int, int]] = {}
        for user_id in self._decode_token_needs_by_user(batch):
            cached_total_tokens = tree_cache.total_user_counters.get_tokens(user_id)
            cached_evictable_tokens = tree_cache.evictable_total_user_counters.get_tokens(
                user_id
            )
            current_tokens = cached_total_tokens + uncached_running_tokens.get(user_id, 0)
            unevictable_tokens = (
                cached_total_tokens - cached_evictable_tokens
            ) + uncached_running_tokens.get(user_id, 0)
            slack = tree_cache.static_max_per_user - unevictable_tokens
            usage_and_slack[user_id] = (current_tokens, unevictable_tokens, slack)
        return usage_and_slack

    def _evict_non_batch_users_for_decode(
        self,
        batch: "ScheduleBatch",
        num_tokens: int,
    ) -> int:
        tree_cache = self.tree_cache
        assert tree_cache is not None

        batch_users = {req.uid for req in batch.reqs}
        return tree_cache.evict(
            num_tokens,
            batch.token_to_kv_pool.free,
            evict_condition=lambda node: node.owner not in batch_users,
        )

    def _user_prefill_protected_tokens(
        self,
        user_id: str,
        *,
        running_batch: Optional["ScheduleBatch"],
        pending_prefill_tokens: int = 0,
    ) -> int:
        tree_cache = self.tree_cache
        assert tree_cache is not None

        cached_total_tokens = tree_cache.total_user_counters.get_tokens(user_id)
        cached_evictable_tokens = tree_cache.evictable_total_user_counters.get_tokens(user_id)
        cached_unevictable_tokens = cached_total_tokens - cached_evictable_tokens

        uncached_running_tokens = 0
        if running_batch is not None and getattr(running_batch, "seq_lens", None) is not None:
            seq_lens_cpu = running_batch.seq_lens.cpu().tolist()
            for i, req in enumerate(running_batch.reqs):
                if req.uid != user_id:
                    continue
                uncached_running_tokens += max(
                    0, int(seq_lens_cpu[i]) - len(req.prefix_indices)
                )

        return cached_unevictable_tokens + uncached_running_tokens + pending_prefill_tokens

    def _can_admit_prefill_without_decode_retraction(
        self,
        req: "Req",
        *,
        running_batch: Optional["ScheduleBatch"],
        pending_prefill_tokens: int,
    ) -> bool:
        tree_cache = self.tree_cache
        assert tree_cache is not None
        if tree_cache.static_max_per_user is None:
            return True

        protected_tokens = self._user_prefill_protected_tokens(
            req.uid,
            running_batch=running_batch,
            pending_prefill_tokens=pending_prefill_tokens,
        )
        return protected_tokens + req.extend_input_len <= tree_cache.static_max_per_user

    def _static_prefill_headroom_limit(self) -> Optional[int]:
        tree_cache = self.tree_cache
        if tree_cache is None or tree_cache.static_max_per_user is None:
            return None
        return max(1, int(tree_cache.static_max_per_user * PREFILL_PROTECTED_HEADROOM_FRACTION))

    def _reject_due_to_static_prefill_headroom(
        self,
        req: "Req",
        *,
        extra_tokens: int = 0,
    ) -> bool:
        headroom_limit = self._static_prefill_headroom_limit()
        if headroom_limit is None:
            return False

        tree_cache = self.tree_cache
        assert tree_cache is not None
        protected_tokens = (
            tree_cache.total_user_counters.get_tokens(req.uid)
            - tree_cache.evictable_total_user_counters.get_tokens(req.uid)
        )
        return protected_tokens + req.extend_input_len + extra_tokens > headroom_limit

    # ---- Request admission -------------------------------------------------
    def init_next_round_input_control(
        self,
        req: "Req",
        *,
        fair: bool = False,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        if not self._has_static_limit():
            return None

        tree_cache = self.tree_cache
        assert tree_cache is not None
        if self._reject_due_to_static_prefill_headroom(req, extra_tokens=extra_tokens):
            return "rejected"
        if tree_cache.reject_based_on_static_limit(req.uid, req.extend_input_len + extra_tokens):
            return "rejected"
        return None

    def add_prefill_request_control(
        self,
        req: "Req",
        *,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        if not self._has_static_limit():
            return None

        tree_cache = self.tree_cache
        assert tree_cache is not None
        if self._reject_due_to_static_prefill_headroom(req, extra_tokens=extra_tokens):
            return "rejected"
        if tree_cache.reject_based_on_static_limit(req.uid, req.extend_input_len + extra_tokens):
            return "rejected"
        return None

    # ---- Allocation helpers -----------------------------------------------
    def requires_per_user_allocation(self) -> bool:
        return self._has_static_limit()

    def uses_static_isolated_memory(self) -> bool:
        return self._has_static_limit()

    def ignore_global_prefill_token_budget(self) -> bool:
        return self._has_static_limit()

    def continue_scanning_waiting_queue_on_prefill_block(self) -> bool:
        return self._has_static_limit()

    def deny_prefill_if_decode_retraction_needed(self) -> bool:
        return self._has_static_limit()

    def running_request_partition_size(
        self,
        *,
        max_running_requests: int,
    ) -> Optional[int]:
        if not self._has_static_limit() or not self.static_reservation_n:
            return None
        return max(1, max_running_requests // self.static_reservation_n)

    def can_admit_running_request(
        self,
        req: "Req",
        *,
        running_batch: Optional["ScheduleBatch"],
        token_counters_by_user: Dict[str, List[int]],
        max_running_requests: int,
    ) -> bool:
        partition_size = self.running_request_partition_size(
            max_running_requests=max_running_requests
        )
        if partition_size is None:
            return True

        running_for_user = 0
        if running_batch is not None:
            running_for_user = sum(1 for running_req in running_batch.reqs if running_req.uid == req.uid)

        pending_for_user = len(token_counters_by_user.get(req.uid, []))
        return running_for_user + pending_for_user < partition_size

    def alloc_token_slots(
        self,
        token_to_kv_pool: "BaseTokenToKVPool",
        num_tokens: int,
        *,
        user_id: Optional[str] = None,
        evict_only_force: bool = False,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> Optional[int]:
        if not self._has_static_limit():
            return super().alloc_token_slots(
                token_to_kv_pool,
                num_tokens,
                user_id=user_id,
                evict_only_force=evict_only_force,
                requesting_users=requesting_users,
            )

        tree_cache = self.tree_cache
        assert tree_cache is not None
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
        if not self.requires_per_user_allocation():
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
                        batch.token_to_kv_pool,
                        num_tokens,
                        user_id=req.uid,
                        evict_only_force=True,
                    )

            out_cache_loc = batch.token_to_kv_pool.alloc(extend_num_tokens)
            if out_cache_loc is None and running_batch is not None:
                if self.deny_prefill_if_decode_retraction_needed():
                    logger.info(
                        "StaticFairnessPolicy: denying prefill admission because it would require decode retraction."
                    )
                    break
                removed, _ = running_batch.retract_decode(extend_num_tokens)
                removed_requests += removed
            else:
                break

        if out_cache_loc is None:
            raise RuntimeError(
                "Static prefill admission denied: insufficient user-local capacity without decode retraction."
            )

        return out_cache_loc, removed_requests

    def process_waiting_queue_prefills(
        self,
        waiting_queue: List["Req"],
        *,
        adder: "PrefillAdder",
        token_counters_by_user: Dict[str, List[int]],
        prefix_computed: bool,
        running_batch: Optional["ScheduleBatch"],
        running_batch_size: int,
        max_running_requests: int,
        available_req_slots: int,
        max_input_size: Optional[int],
    ) -> None:
        if not self.requires_per_user_allocation():
            return super().process_waiting_queue_prefills(
                waiting_queue,
                adder=adder,
                token_counters_by_user=token_counters_by_user,
                prefix_computed=prefix_computed,
                running_batch=running_batch,
                running_batch_size=running_batch_size,
                max_running_requests=max_running_requests,
                available_req_slots=available_req_slots,
                max_input_size=max_input_size,
            )

        target_tree_cache = None if prefix_computed else self.tree_cache
        effective_running_limit = min(
            max_running_requests,
            running_batch_size + max(0, available_req_slots),
        )

        if running_batch_size >= effective_running_limit:
            return

        for req in waiting_queue:
            if max_input_size is not None and adder.log_input_tokens > max_input_size:
                break
            elif max_input_size is not None:
                adder.rem_input_tokens = max_input_size - adder.log_input_tokens

            if req in adder.can_run_list:
                continue

            if not self.can_admit_running_request(
                req,
                running_batch=running_batch,
                token_counters_by_user=token_counters_by_user,
                max_running_requests=max_running_requests,
            ):
                continue

            extra_tokens = sum(token_counters_by_user.get(req.uid, []))
            if not self._can_admit_prefill_without_decode_retraction(
                req,
                running_batch=running_batch,
                pending_prefill_tokens=extra_tokens,
            ):
                continue

            res = req.init_next_round_input(
                target_tree_cache,
                fairness_policy=self,
                fair=False,
                extra_tokens=extra_tokens,
            )
            if res == "rejected":
                continue

            add_result = adder.add_one_req(req, extra_tokens=extra_tokens)
            if add_result == "rejected":
                continue

            token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)

            if (
                (
                    not add_result
                    and not self.continue_scanning_waiting_queue_on_prefill_block()
                )
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= effective_running_limit
            ):
                break

    # ---- Decode helpers ----------------------------------------------------
    def check_decode_memory(self, batch: "ScheduleBatch") -> bool:
        if not self.requires_per_user_allocation():
            return super().check_decode_memory(batch)

        bs = batch.batch_size()
        tree_cache = self.tree_cache
        assert tree_cache is not None
        overages = self._users_exceeding_static_limit_for_decode(batch)

        if batch.token_to_kv_pool.available_size() >= bs and not overages:
            return True

        logger.info(
            "StaticFairnessPolicy: decode pressure (available=%s, needed=%s, overages=%s), evicting by user.",
            batch.token_to_kv_pool.available_size(),
            bs,
            overages,
        )
        if not overages and batch.token_to_kv_pool.available_size() < bs:
            evicted = self._evict_non_batch_users_for_decode(
                batch, bs - batch.token_to_kv_pool.available_size()
            )
            logger.info(
                "StaticFairnessPolicy: evicted %s tokens from non-batch users for decode pressure.",
                evicted,
            )

        for user_id, overage in overages.items():
            tree_cache.evict_from_user(overage, batch.token_to_kv_pool.free, user_id)

        remaining_overages = self._users_exceeding_static_limit_for_decode(batch)
        if batch.token_to_kv_pool.available_size() >= bs and not remaining_overages:
            logger.info(
                "StaticFairnessPolicy: eviction successful (available=%s).",
                batch.token_to_kv_pool.available_size(),
            )
            return True

        return False

    def get_retract_order(self, batch: "ScheduleBatch") -> List[int]:
        if not self.requires_per_user_allocation():
            return super().get_retract_order(batch)

        overages = self._users_exceeding_static_limit_for_decode(batch)
        violating_indices = [i for i, req in enumerate(batch.reqs) if req.uid in overages]

        if not violating_indices and batch.token_to_kv_pool.available_size() < batch.batch_size():
            usage_and_slack = self._decode_usage_and_slack_by_user(batch)
            logger.info(
                "StaticFairnessPolicy: decode retraction fallback with usage/slack=%s",
                {
                    user_id: {
                        "usage": usage,
                        "unevictable": unevictable,
                        "slack": slack,
                    }
                    for user_id, (usage, unevictable, slack) in usage_and_slack.items()
                },
            )
            violating_indices = list(range(len(batch.reqs)))
            violating_indices.sort(
                key=lambda i: (
                    usage_and_slack[batch.reqs[i].uid][1],
                    usage_and_slack[batch.reqs[i].uid][0],
                    -usage_and_slack[batch.reqs[i].uid][2],
                    -len(batch.reqs[i].origin_input_ids),
                    -len(batch.reqs[i].output_ids),
                ),
                reverse=True,
            )
            return violating_indices

        violating_indices.sort(
            key=lambda i: (
                overages[batch.reqs[i].uid],
                -len(batch.reqs[i].origin_input_ids),
                -len(batch.reqs[i].output_ids),
            ),
            reverse=True,
        )
        return violating_indices

    def alloc_decode_output_slots(self, batch: "ScheduleBatch"):
        if not self.requires_per_user_allocation():
            return super().alloc_decode_output_slots(batch)

        bs = batch.batch_size()
        overages = self._users_exceeding_static_limit_for_decode(batch)
        if not overages and batch.token_to_kv_pool.available_size() < bs:
            self._evict_non_batch_users_for_decode(
                batch, bs - batch.token_to_kv_pool.available_size()
            )
        for user_id, overage in overages.items():
            self.alloc_token_slots(
                batch.token_to_kv_pool,
                overage,
                user_id=user_id,
                evict_only_force=True,
            )

        out_cache_loc = batch.token_to_kv_pool.alloc(bs)
        if out_cache_loc is None:
            raise RuntimeError(
                "Failed to allocate decode slots under static fairness policy."
            )
        return out_cache_loc
