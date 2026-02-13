from __future__ import annotations

"""Fairness control helpers for the baseline (no-op) policy.

This module mirrors the control-flow that existed before any fairness logic
was introduced. Concrete fairness policies subclass this class and override
the individual hooks with the more complex logic extracted from the manager
diffs.
"""

from typing import Dict, List, Optional, Sequence, Tuple

if False:  # pragma: no cover - imported only for type checkers
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
    from sglang.srt.managers.policy_scheduler import PrefillAdder


class NoFairnessPolicy:
    """Encapsulates the original scheduling behaviour without fairness checks."""

    def init_next_round_input_control(
        self,
        tree_cache: Optional["BasePrefixCache"],
        req: "Req",
        *,
        fair: bool = False,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        """Baseline implementation never rejects a request."""

        return None

    def add_prefill_request_control(
        self,
        tree_cache: Optional["BasePrefixCache"],
        req: "Req",
        *,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        """Baseline implementation never rejects a request."""

        return None

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
        """Replicates the original best-effort allocation logic."""

        out_cache_loc = None if evict_only_force else token_to_kv_pool.alloc(num_tokens)

        if evict_only_force or (out_cache_loc is None and not evict_only_force):
            if tree_cache is not None:
                tree_cache.evict(num_tokens, token_to_kv_pool.free)
                if not evict_only_force:
                    out_cache_loc = token_to_kv_pool.alloc(num_tokens)

        if out_cache_loc is None and not evict_only_force:
            raise RuntimeError(
                "Prefill out of memory. Try lowering the batch size for baseline policy."
            )

        return None if evict_only_force else out_cache_loc

    def requires_per_user_allocation(self, tree_cache: Optional["BasePrefixCache"]) -> bool:
        return False

    def handle_prefill_eviction(
        self,
        batch: "ScheduleBatch",
        extend_num_tokens: int,
        *,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> None:
        if batch.tree_cache is not None:
            batch.tree_cache.evict(extend_num_tokens, batch.token_to_kv_pool.free)

    def prepare_for_extend_allocation(
        self,
        batch: "ScheduleBatch",
        extend_num_tokens: int,
        *,
        running_batch: Optional["ScheduleBatch"] = None,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> tuple[int, List["Req"]]:
        out_cache_loc = self.alloc_token_slots(
            batch.tree_cache, batch.token_to_kv_pool, extend_num_tokens
        )
        return out_cache_loc, []

    def check_decode_memory(self, batch: "ScheduleBatch") -> bool:
        bs = batch.batch_size()
        if batch.token_to_kv_pool.available_size() >= bs:
            return True

        if batch.tree_cache is not None:
            batch.tree_cache.evict(bs, batch.token_to_kv_pool.free)

        return batch.token_to_kv_pool.available_size() >= bs

    def get_retract_order(self, batch: "ScheduleBatch") -> List[int]:
        sorted_indices = [i for i in range(len(batch.reqs))]
        sorted_indices.sort(
            key=lambda i: (
                len(batch.reqs[i].output_ids),
                -len(batch.reqs[i].origin_input_ids),
            ),
            reverse=True,
        )
        return sorted_indices

    def alloc_decode_output_slots(self, batch: "ScheduleBatch"):
        return self.alloc_token_slots(batch.tree_cache, batch.token_to_kv_pool, batch.batch_size())

    # ---- Delta fairness hooks (no-op defaults) ----
    def fairinf_force_decode(
        self,
        running_batch: Optional["ScheduleBatch"],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        tree_cache: Optional["BasePrefixCache"] = None,
        delta_fairness_n: Optional[int] = None,
        max_running_requests: Optional[int] = None,
        decode_time_us: int = 20000,
    ) -> Tuple[bool, Optional[int]]:
        return False, None

    def fairinf_force_prefill(
        self,
        req: "Req",
        token_counters_by_user: Dict[str, List[int]],
        *,
        tree_cache: Optional["BasePrefixCache"],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        delta_fairness_n: Optional[int] = None,
        running_batch: Optional["ScheduleBatch"] = None,
        max_running_requests: Optional[int] = None,
        decode_time_us: int = 20000,
    ) -> bool:
        return False

    def fairinf_force_prefill_any_waiting(
        self,
        waiting_queue: List["Req"],
        *,
        tree_cache: Optional["BasePrefixCache"],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        delta_fairness_n: Optional[int] = None,
        running_batch: Optional["ScheduleBatch"] = None,
        max_running_requests: Optional[int] = None,
    ) -> bool:
        return False

    def user_is_fair_prefill(
        self,
        user_id: str,
        *,
        tree_cache: Optional["BasePrefixCache"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_n: Optional[int],
        max_running_requests: Optional[int],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        return False

    def req_is_fair_prefill(
        self,
        req: "Req",
        *,
        tree_cache: Optional["BasePrefixCache"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_n: Optional[int],
        max_running_requests: Optional[int],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        return False

    def req_is_fair_decode(
        self,
        req: "Req",
        *,
        tree_cache: Optional["BasePrefixCache"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_n: Optional[int],
        max_running_requests: Optional[int],
    ) -> bool:
        return False

    def force_prefill_reservations(
        self,
        waiting_queue: List["Req"],
        *,
        token_counters_by_user: Dict[str, List[int]],
        adder: "PrefillAdder",
        tree_cache: Optional["BasePrefixCache"],
        token_to_kv_pool: Optional["BaseTokenToKVPool"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        delta_fairness_n: Optional[int] = None,
        max_running_requests: Optional[int] = None,
        prefix_computed: bool = False,
    ) -> Tuple[int, Optional[List["Req"]]]:
        return 0, None

    def process_waiting_queue_prefills(
        self,
        waiting_queue: List["Req"],
        *,
        adder: "PrefillAdder",
        token_counters_by_user: Dict[str, List[int]],
        prefix_computed: bool,
        tree_cache: Optional["BasePrefixCache"],
        running_batch_size: int,
        max_running_requests: int,
        max_input_size: Optional[int],
    ) -> None:
        target_tree_cache = None if prefix_computed else tree_cache

        for req in waiting_queue:
            if max_input_size is not None and adder.log_input_tokens > max_input_size:
                break
            elif max_input_size is not None:
                adder.rem_input_tokens = max_input_size - adder.log_input_tokens

            if req in adder.can_run_list:
                continue

            extra_tokens = sum(token_counters_by_user.get(req.uid, []))
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
                not add_result
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= max_running_requests
            ):
                break
