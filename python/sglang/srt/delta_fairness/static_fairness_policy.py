from __future__ import annotations

"""Fairness helpers for static per-user KV cache reservations."""

import logging
import os
import time
from collections import Counter
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .no_fairness_policy import NoFairnessPolicy

logger = logging.getLogger(__name__)
CLIP_MAX_NEW_TOKENS = int(os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS", "4096"))

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
        self.static_debug_admission = os.environ.get(
            "SGLANG_STATIC_DEBUG_ADMISSION", ""
        ).lower() in ("1", "true", "yes", "on")
        self._last_static_debug_log_ts = 0.0

    def _maybe_log_static_prefill_summary(
        self,
        *,
        summaries_by_user: Dict[str, Dict[str, int]],
        details_by_user: Dict[str, Dict[str, int]],
    ) -> None:
        if not self.static_debug_admission or not summaries_by_user:
            return
        now = time.time()
        if now - self._last_static_debug_log_ts < 5.0:
            return
        self._last_static_debug_log_ts = now

        parts = []
        for user_id in sorted(summaries_by_user):
            summary = summaries_by_user[user_id]
            detail = details_by_user.get(user_id, {})
            parts.append(
                (
                    f"user={user_id} "
                    f"admitted={summary.get('admitted', 0)} "
                    f"blocked_partition={summary.get('blocked_partition', 0)} "
                    f"blocked_protected={summary.get('blocked_protected', 0)} "
                    f"rejected_init={summary.get('rejected_init', 0)} "
                    f"rejected_adder={summary.get('rejected_adder', 0)} "
                    f"running={detail.get('running_for_user', -1)}/{detail.get('partition_size', -1)} "
                    f"pending={detail.get('pending_for_user', -1)} "
                    f"protected={detail.get('protected_tokens', -1)}/{detail.get('static_limit', -1)} "
                    f"req_extend={detail.get('req_extend_input_len', -1)}"
                )
            )
        logger.info("StaticFairnessPolicy prefill gate summary: %s", " | ".join(parts))

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
        if tree_cache.static_max_per_user is None:
            return {}

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
            overage = (
                current_tokens
                + needed_tokens
                - tree_cache.static_max_per_user
            )
            if overage > 0:
                overages[user_id] = overage

        if not overages:
            deficit = batch.batch_size() - batch.token_to_kv_pool.available_size()
            if deficit > 0:
                usage_by_user: Dict[str, int] = {}
                for user_id, needed_tokens in self._decode_token_needs_by_user(batch).items():
                    usage_by_user[user_id] = (
                        tree_cache.total_user_counters.get_tokens(user_id)
                        + uncached_running_tokens.get(user_id, 0)
                        + needed_tokens
                    )
                if usage_by_user:
                    fallback_user = max(usage_by_user, key=usage_by_user.get)
                    overages[fallback_user] = deficit
        return overages

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

    def _user_running_decode_liability_tokens(
        self,
        user_id: str,
        *,
        running_batch: Optional["ScheduleBatch"],
    ) -> int:
        if running_batch is None:
            return 0

        liability = 0
        for req in running_batch.reqs:
            if req.uid != user_id:
                continue
            liability += max(
                0,
                min(
                    req.sampling_params.max_new_tokens - len(req.output_ids),
                    CLIP_MAX_NEW_TOKENS,
                ),
            )
        return liability

    def _prefill_total_tokens_for_req(self, req: "Req") -> int:
        return req.extend_input_len + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)

    def _reject_due_to_static_partition_total(
        self,
        req: "Req",
        *,
        pending_total_tokens: int = 0,
    ) -> bool:
        tree_cache = self.tree_cache
        if tree_cache is None or tree_cache.static_max_per_user is None:
            return False
        protected_tokens = self._user_prefill_protected_tokens(
            req.uid,
            running_batch=getattr(self, "_pass_running_batch", None),
            pending_prefill_tokens=0,
        )
        running_decode_liability = self._user_running_decode_liability_tokens(
            req.uid,
            running_batch=getattr(self, "_pass_running_batch", None),
        )
        return (
            protected_tokens
            + running_decode_liability
            + pending_total_tokens
            + self._prefill_total_tokens_for_req(req)
            > tree_cache.static_max_per_user
        )

    def _reject_due_to_static_immediate_extend_capacity(
        self,
        req: "Req",
        *,
        pending_global_extend_tokens: int = 0,
    ) -> bool:
        tree_cache = self.tree_cache
        if tree_cache is None:
            return False

        available_now = max(
            0,
            tree_cache.token_to_kv_pool.available_size() - pending_global_extend_tokens,
        )
        user_evictable = tree_cache.evictable_total_user_counters.get_tokens(req.uid)
        overlimit_evictable = 0
        if tree_cache.static_max_per_user is not None:
            total_snapshot = tree_cache.total_user_counters.snapshot()
            evictable_snapshot = tree_cache.evictable_total_user_counters.snapshot()
            for user_id, total_tokens in total_snapshot.items():
                if user_id == req.uid:
                    continue
                if total_tokens > tree_cache.static_max_per_user:
                    overlimit_evictable += evictable_snapshot.get(user_id, 0)

        return req.extend_input_len > available_now + user_evictable + overlimit_evictable

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

        if self._reject_due_to_static_partition_total(
            req,
            pending_total_tokens=extra_tokens,
        ):
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

        if self._reject_due_to_static_partition_total(
            req,
            pending_total_tokens=extra_tokens,
        ):
            return "rejected"
        return None

    # ---- Allocation helpers -----------------------------------------------
    def requires_per_user_allocation(self) -> bool:
        return self._has_static_limit()

    def uses_static_isolated_memory(self) -> bool:
        return self._has_static_limit()

    def ignore_global_prefill_token_budget(self) -> bool:
        return False

    def continue_scanning_waiting_queue_on_prefill_block(self) -> bool:
        return self._has_static_limit()

    def deny_prefill_if_decode_retraction_needed(self) -> bool:
        return False

    def running_request_partition_size(
        self,
        *,
        max_running_requests: int,
    ) -> Optional[int]:
        del max_running_requests
        return None

    def can_admit_running_request(
        self,
        req: "Req",
        *,
        running_batch: Optional["ScheduleBatch"],
        token_counters_by_user: Dict[str, List[int]],
        max_running_requests: int,
    ) -> bool:
        del req, running_batch, token_counters_by_user, max_running_requests
        return True

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

        if (
            out_cache_loc is None
            and not evict_only_force
            and tree_cache.static_max_per_user is not None
        ):
            # If the requesting user is still within its partition but the pooled KV
            # space is tight, reclaim only evictable cache from users currently above
            # their static partition before giving up.
            tree_cache.evict(
                num_tokens,
                token_to_kv_pool.free,
                evict_condition=lambda node: (
                    node.owner != user_id
                    and tree_cache.total_user_counters.get_tokens(node.owner)
                    > tree_cache.static_max_per_user
                ),
            )
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
        out_cache_loc = self.alloc_token_slots(
            batch.token_to_kv_pool,
            extend_num_tokens,
            user_id=batch.reqs[0].uid if batch.reqs else None,
        )
        return out_cache_loc, []

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
        partition_size = self.running_request_partition_size(
            max_running_requests=max_running_requests
        )
        pending_global_extend_tokens = 0
        summaries_by_user: Dict[str, Dict[str, int]] = {}
        details_by_user: Dict[str, Dict[str, int]] = {}

        if running_batch_size >= effective_running_limit:
            return

        ordered_waiting_queue = waiting_queue
        if (
            self.tree_cache is not None
            and getattr(self.tree_cache, "static_max_per_user", None) is not None
        ):
            static_limit = self.tree_cache.static_max_per_user
            queue_by_user = defaultdict(list)
            for idx, req in enumerate(waiting_queue):
                queue_by_user[req.uid].append((idx, req))

            user_order = sorted(
                queue_by_user,
                key=lambda user_id: (
                    (
                        self._user_prefill_protected_tokens(
                            user_id,
                            running_batch=running_batch,
                            pending_prefill_tokens=sum(
                                token_counters_by_user.get(user_id, [])
                            ),
                        )
                        + self._user_running_decode_liability_tokens(
                            user_id,
                            running_batch=running_batch,
                        )
                    )
                    / max(1, static_limit),
                    queue_by_user[user_id][0][0],
                ),
            )

            ordered_waiting_queue = []
            exhausted = False
            round_index = 0
            while not exhausted:
                exhausted = True
                for user_id in user_order:
                    user_queue = queue_by_user[user_id]
                    if round_index < len(user_queue):
                        ordered_waiting_queue.append(user_queue[round_index][1])
                        exhausted = False
                round_index += 1

        for req in ordered_waiting_queue:
            if max_input_size is not None and adder.log_input_tokens > max_input_size:
                break
            elif max_input_size is not None:
                adder.rem_input_tokens = max_input_size - adder.log_input_tokens

            if req in adder.can_run_list:
                continue

            running_for_user = 0
            if running_batch is not None:
                running_for_user = sum(
                    1 for running_req in running_batch.reqs if running_req.uid == req.uid
                )
            pending_for_user = len(token_counters_by_user.get(req.uid, []))
            protected_tokens = self._user_prefill_protected_tokens(
                req.uid,
                running_batch=running_batch,
                pending_prefill_tokens=sum(token_counters_by_user.get(req.uid, [])),
            )
            static_limit = (
                self.tree_cache.static_max_per_user
                if self.tree_cache is not None
                and getattr(self.tree_cache, "static_max_per_user", None) is not None
                else -1
            )
            summary = summaries_by_user.setdefault(req.uid, {})
            detail = details_by_user.setdefault(
                req.uid,
                {
                    "running_for_user": running_for_user,
                    "pending_for_user": pending_for_user,
                    "partition_size": partition_size if partition_size is not None else -1,
                    "protected_tokens": protected_tokens,
                    "static_limit": static_limit,
                    "req_extend_input_len": req.extend_input_len,
                },
            )
            detail.update(
                {
                    "running_for_user": running_for_user,
                    "pending_for_user": pending_for_user,
                    "partition_size": partition_size if partition_size is not None else -1,
                    "protected_tokens": protected_tokens,
                    "static_limit": static_limit,
                    "req_extend_input_len": req.extend_input_len,
                }
            )

            if self._reject_due_to_static_immediate_extend_capacity(
                req,
                pending_global_extend_tokens=pending_global_extend_tokens,
            ):
                summary["blocked_immediate_capacity"] = (
                    summary.get("blocked_immediate_capacity", 0) + 1
                )
                continue

            can_admit_running = self.can_admit_running_request(
                req,
                running_batch=running_batch,
                token_counters_by_user=token_counters_by_user,
                max_running_requests=max_running_requests,
            )
            res = req.init_next_round_input(
                target_tree_cache,
                fairness_policy=self,
                fair=False,
                extra_tokens=sum(token_counters_by_user.get(req.uid, [])),
            )
            if res == "rejected":
                summary["rejected_init"] = summary.get("rejected_init", 0) + 1
                continue

            add_result = adder.add_one_req(
                req, extra_tokens=sum(token_counters_by_user.get(req.uid, []))
            )
            if add_result == "rejected":
                summary["rejected_adder"] = summary.get("rejected_adder", 0) + 1
                continue

            summary["admitted"] = summary.get("admitted", 0) + 1
            token_counters_by_user.setdefault(req.uid, []).append(
                self._prefill_total_tokens_for_req(req)
            )
            pending_global_extend_tokens += req.extend_input_len

            if (
                (
                    not add_result
                    and not self.continue_scanning_waiting_queue_on_prefill_block()
                )
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= effective_running_limit
            ):
                break

        self._maybe_log_static_prefill_summary(
            summaries_by_user=summaries_by_user,
            details_by_user=details_by_user,
        )

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

        if not violating_indices:
            return []

        violating_indices.sort(
            key=lambda i: (
                -overages[batch.reqs[i].uid],
                -len(batch.reqs[i].origin_input_ids),
                len(batch.reqs[i].output_ids),
            ),
            reverse=True,
        )
        return violating_indices

    def alloc_decode_output_slots(self, batch: "ScheduleBatch"):
        if not self.requires_per_user_allocation():
            return super().alloc_decode_output_slots(batch)

        overages = self._users_exceeding_static_limit_for_decode(batch)
        for user_id, overage in overages.items():
            self.alloc_token_slots(
                batch.token_to_kv_pool,
                overage,
                user_id=user_id,
                evict_only_force=True,
            )

        out_cache_loc = batch.token_to_kv_pool.alloc(batch.batch_size())
        if out_cache_loc is None:
            raise RuntimeError(
                "Failed to allocate decode slots under static fairness policy."
            )
        return out_cache_loc
