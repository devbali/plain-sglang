from __future__ import annotations

"""Delta fairness control logic extracted from the manager diffs."""

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .static_fairness_policy import StaticFairnessPolicy

logger = logging.getLogger(__name__)

CLIP_MAX_NEW_TOKENS = int(os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS", "4096"))
DECODE_TIME_US = 20000
PREFILL_TOKEN_PER_DECODE = 250

# pragma: no cover - imported only for type checkers
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool
from sglang.srt.managers.policy_scheduler import PrefillAdder


class DeltaFairnessPolicy(StaticFairnessPolicy):
    """Encapsulates delta fairness scheduling decisions."""

    def __init__(
        self,
        *,
        tree_cache: Optional["BasePrefixCache"] = None,
        delta_fairness_n: Optional[int] = None,
        max_running_requests: Optional[int] = None,
    ):
        super().__init__(tree_cache=tree_cache)
        self.delta_fairness_n = delta_fairness_n
        self.max_running_requests = max_running_requests

    def _has_delta_limit(self) -> bool:
        tree_cache = self.tree_cache
        return tree_cache is not None and getattr(tree_cache, "fairinf_max_per_user", None) is not None

    # ---- Request admission -------------------------------------------------
    def init_next_round_input_control(
        self,
        req: Req,
        *,
        fair: bool = False,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        """
        This is called for an inflight prefill request being init-ed into the next prefill branch
        Return the string rejected here if it is to not be next prefill
        """
        
        rejected = super().init_next_round_input_control(
            req, fair=fair, extra_tokens=extra_tokens
        )
        if rejected is not None or not self._has_delta_limit():
            return rejected

        tree_cache = self.tree_cache
        assert tree_cache is not None
        if fair:
            if tree_cache.reject_based_on_computed_fair_limit(req.uid, req.extend_input_len + extra_tokens):
                return "rejected"
        else:
            if tree_cache.reject_based_on_computed_fair_limit_unfair(
                req.uid, req.extend_input_len + extra_tokens
            ):
                return "rejected"
        return None

    def add_prefill_request_control(
        self,
        req: "Req",
        *,
        extra_tokens: int = 0,
    ) -> Optional[str]:
        """
        Only called from within this class indirectly
        We are deciding whether to add this prefill request that has already been approved by the other methods of this
        class, via add_one_req being called
        """
        rejected = super().add_prefill_request_control(
            req, extra_tokens=extra_tokens
        )
        if rejected is not None or not self._has_delta_limit():
            return rejected

        tree_cache = self.tree_cache
        assert tree_cache is not None
        if tree_cache.reject_based_on_computed_fair_limit(req.uid, req.extend_input_len + extra_tokens):
            return "rejected"
        return None

    # ---- Allocation helpers -----------------------------------------------
    def alloc_token_slots(
        self,
        token_to_kv_pool: "BaseTokenToKVPool",
        num_tokens: int,
        *,
        user_id: Optional[str] = None,
        evict_only_force: bool = False,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> Optional[int]:
        """
        Only called from within this class, directly as a helper function that
        allocates token slots for a request.
        This is not supposed to evict running requests if necessary
        Return None if not possible, and if evict_only_force is not set then raise an error
        """
        if not self._has_delta_limit():
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
            "DeltaFairnessPolicy: alloc %s tokens (force=%s, initial_loc=%s, users=%s)",
            num_tokens,
            evict_only_force,
            out_cache_loc,
            requesting_users,
        )

        if evict_only_force or (out_cache_loc is None and not evict_only_force):
            if requesting_users is None:
                requesting_users = []
            tree_cache.evict_delta_fair(num_tokens, token_to_kv_pool.free, requesting_users)
            if not evict_only_force:
                out_cache_loc = token_to_kv_pool.alloc(num_tokens)

        if out_cache_loc is None and not evict_only_force:
            logger.error("Prefill out of memory under delta fairness policy.")
            raise RuntimeError("Prefill out of memory under delta fairness policy.")

        return None if evict_only_force else out_cache_loc

    def prepare_for_extend_allocation(
        self,
        batch: "ScheduleBatch",
        extend_num_tokens: int,
        *,
        running_batch: Optional["ScheduleBatch"] = None,
        requesting_users: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[int], List["Req"]]:
        """ 
        Extending the allocation for a particular request
        Evict running requests to make space
        At this point the request can not be rejected, should have been done at earlier steps
        """
        if not self._has_delta_limit():
            return super().prepare_for_extend_allocation(
                batch,
                extend_num_tokens,
                running_batch=running_batch,
                requesting_users=requesting_users,
            )

        out_cache_loc = None
        removed_requests: List["Req"] = []
        requesting_users = list(requesting_users or [])
        alloc_start = time.perf_counter()
        retract_count = 0

        while out_cache_loc is None:
            loop_start = time.perf_counter()
            eviction_necessary = batch.token_to_kv_pool.available_size() < extend_num_tokens
            if eviction_necessary:
                self.alloc_token_slots(
                    batch.token_to_kv_pool,
                    extend_num_tokens,
                    evict_only_force=True,
                    requesting_users=requesting_users,
                )

            out_cache_loc = batch.token_to_kv_pool.alloc(extend_num_tokens)
            if out_cache_loc is None and running_batch is not None:
                retract_start = time.perf_counter()
                removed, _ = running_batch.retract_decode(extend_num_tokens)
                retract_count += 1
                removed_requests += removed
                logger.info(
                    "DeltaFairness prepare_for_extend_allocation retract_iter=%s removed=%s loop_ms=%.3f retract_ms=%.3f available_after=%s need=%s",
                    retract_count,
                    len(removed),
                    (time.perf_counter() - loop_start) * 1000.0,
                    (time.perf_counter() - retract_start) * 1000.0,
                    batch.token_to_kv_pool.available_size(),
                    extend_num_tokens,
                )
            else:
                break

        if out_cache_loc is None:
            raise RuntimeError(
                "Allocation failed after evicting for delta fairness reservations."
            )

        logger.info(
            "DeltaFairness prepare_for_extend_allocation total_ms=%.3f retract_iters=%s removed_total=%s extend_tokens=%s available_final=%s",
            (time.perf_counter() - alloc_start) * 1000.0,
            retract_count,
            len(removed_requests),
            extend_num_tokens,
            batch.token_to_kv_pool.available_size(),
        )

        return out_cache_loc, removed_requests

    # ---- Decode helpers ----------------------------------------------------
    def check_decode_memory(self, batch: "ScheduleBatch") -> bool:
        if not self._has_delta_limit():
            return super().check_decode_memory(batch)

        bs = batch.batch_size()
        if batch.token_to_kv_pool.available_size() >= bs:
            return True

        logger.info(
            "DeltaFairnessPolicy: decode OOM (available=%s vs %s needed), evicting delta fair.",
            batch.token_to_kv_pool.available_size(),
            bs,
        )
        assert self.tree_cache is not None
        self.tree_cache.evict_delta_fair(bs, batch.token_to_kv_pool.free, None)

        if batch.token_to_kv_pool.available_size() >= bs:
            logger.info(
                "DeltaFairnessPolicy: eviction successful (available=%s).",
                batch.token_to_kv_pool.available_size(),
            )
            return True

        return False

    def get_retract_order(self, batch: "ScheduleBatch") -> List[int]:
        """
        When retraction is being done for decode to free up memory, this decides the ordering
        Returns a list of indices of requests in the batch
        """
        if not self._has_delta_limit():
            return super().get_retract_order(batch)

        tree_cache = self.tree_cache
        assert tree_cache is not None
        retractable: List[int] = []
        for i, req in enumerate(batch.reqs):
            if not self.req_is_fair_decode(
                req,
                running_batch=batch,
            ):
                retractable.append(i)

        for i, req in enumerate(batch.reqs):
            if i not in retractable:
                retractable.append(i)
        return retractable

    def alloc_decode_output_slots(self, batch: "ScheduleBatch"):
        """
        Allocate output slots for decoding. Going to call alloc_token_slots helper probably.
        """
        if not self._has_delta_limit():
            return super().alloc_decode_output_slots(batch)

        bs = batch.batch_size()
        eviction_necessary = batch.token_to_kv_pool.available_size() < bs
        if eviction_necessary:
            for _ in range(bs):
                self.alloc_token_slots(
                    batch.token_to_kv_pool,
                    1,
                    evict_only_force=True,
                    requesting_users=None,
                )

        out_cache_loc = batch.token_to_kv_pool.alloc(bs)
        if out_cache_loc is None:
            raise RuntimeError(
                "Failed to allocate decode slots under delta fairness policy."
            )
        return out_cache_loc

    # ---- Delta fairness specific helpers ----------------------------------
    def fairinf_force_decode(
        self,
        running_batch: Optional["ScheduleBatch"],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        decode_time_us: int = DECODE_TIME_US,
    ) -> Tuple[bool, Optional[int]]:
        """ 
        Determine whether a decode is to be forced
        Can either return True along with an ignored int, 
            or False along with a maximum amount of tokens that the next prefill batch can process
        """
        if (
            not self.delta_fairness_n
            or running_batch is None
            or delta_fairness_deltas_microseconds is None
        ):
            return False, None

        decode_wait_in_decodes = (
            delta_fairness_deltas_microseconds.get("decode_running_batch", 0) // decode_time_us
            if decode_time_us > 0
            else 0
        )
        max_decode_wait: Optional[int] = None
        force_decode = False

        for req in running_batch.reqs:
            if self.req_is_fair_decode(
                req,
                running_batch=running_batch,
            ):
                if req.waiting_time_in_decodes + 1 >= decode_wait_in_decodes:
                    force_decode = True
                    break
                if max_decode_wait is None or req.waiting_time_in_decodes > max_decode_wait:
                    max_decode_wait = req.waiting_time_in_decodes

        if force_decode:
            return True, 0

        if max_decode_wait is None or decode_wait_in_decodes <= max_decode_wait:
            return False, None

        max_prefill_tokens = (decode_wait_in_decodes - max_decode_wait) * PREFILL_TOKEN_PER_DECODE
        return False, max_prefill_tokens

    def fairinf_force_prefill(
        self,
        req: "Req",
        token_counters_by_user: Dict[str, List[int]],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional["ScheduleBatch"] = None,
        decode_time_us: int = DECODE_TIME_US,
    ) -> bool:
        """ 
        Returns a boolean for whether or not the next batch will be a forced prefill batch because of a particular request
        """
        if (
            not self.delta_fairness_n
            or delta_fairness_deltas_microseconds is None
            #or not req.first_time_in_waiting_queue
        ):
            return False

        prefill_wait_in_decodes = (
            delta_fairness_deltas_microseconds.get("prefill_running_batch", 0) // decode_time_us
            if decode_time_us > 0
            else 0
        )
        this_users_extras = token_counters_by_user.get(req.uid, [])
        extra_sum = sum(this_users_extras)

        if self.req_is_fair_prefill(
            req,
            running_batch=running_batch,
            this_user_len=len(this_users_extras),
            this_user_sum=extra_sum,
        ):
            if req.waiting_time_in_decodes + 1 >= prefill_wait_in_decodes:
                return True

        return False

    def fairinf_force_prefill_any_waiting(
        self,
        waiting_queue: List["Req"],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional["ScheduleBatch"] = None,
    ) -> bool:
        """ 
        True if any of the waiting requests demand a forced prefill
        """
        if not self.delta_fairness_n:
            return False

        dummy_counters: Dict[str, List[int]] = {}
        for req in waiting_queue:
            if self.fairinf_force_prefill(
                req,
                dummy_counters,
                delta_fairness_deltas_microseconds=delta_fairness_deltas_microseconds,
                running_batch=running_batch,
            ):
                return True
        return False

    def user_is_fair_prefill(
        self,
        user_id: str,
        *,
        running_batch: Optional["ScheduleBatch"],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        """ 
        Determine if a user is delta fair when looking at the prefill running batch 
        resource
        """
        if not self.delta_fairness_n or running_batch is None:
            return False

        tree_cache = self.tree_cache
        if tree_cache is not None and not tree_cache.user_unevictable_kv_is_under_fair_share_reservation(
            user_id, this_user_sum
        ):
            return False

        total = self.max_running_requests
        if total is None:
            raise ValueError("Can not run delta fairness without max running requests parameter")
        for running_req in running_batch.reqs:
            if running_req.uid == user_id:
                this_user_len += 1

        if this_user_len == 0:
            return True

        return self.delta_fairness_n < total / this_user_len

    def req_is_fair_prefill(
        self,
        req: "Req",
        *,
        running_batch: Optional["ScheduleBatch"],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        """ 
        Determine whether a request is fair (belonging to a delta fair user) when looking
        at the criteria for prefill running batch
        """
        return self.user_is_fair_prefill(
            req.uid,
            running_batch=running_batch,
            this_user_len=this_user_len,
            this_user_sum=req.get_estimated_prefill_impact() + this_user_sum,
        )

    def req_is_fair_decode(
        self,
        req: "Req",
        *,
        running_batch: Optional["ScheduleBatch"],
    ) -> bool:
        """ 
        Determine whether a request is fair (belonging to a delta fair user) when looking
        at the criteria for decode running batch
        """
        return self.req_is_fair_prefill(
            req,
            running_batch=running_batch,
        )

    def force_prefill_reservations(
        self,
        waiting_queue: Sequence["Req"],
        *,
        token_counters_by_user: Dict[str, List[int]],
        adder: "PrefillAdder",
        token_to_kv_pool: Optional["BaseTokenToKVPool"],
        running_batch: Optional["ScheduleBatch"],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        max_input_size: Optional[int] = None,
        prefix_computed: bool = False,
    ) -> Tuple[int, Optional[List["Req"]]]:
        """ 
        Make space for the forced prefill requests
        """
        tree_cache = self.tree_cache
        if (
            not self._has_delta_limit()
            or not self.delta_fairness_n
            or tree_cache is None
            or token_to_kv_pool is None
        ):
            return 0, None

        extra_space = 0
        last_evicted: Optional[List["Req"]] = None
        waiting_snapshot = list(waiting_queue)

        for req in waiting_snapshot:
            if not self.fairinf_force_prefill(
                req,
                token_counters_by_user,
                delta_fairness_deltas_microseconds=delta_fairness_deltas_microseconds,
                running_batch=running_batch,
            ):
                continue

            logger.info("Prefill request forced in by user %s", req.uid)
            total_tokens = len(req.origin_input_ids) + min(
                req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
            )
            extra_space += total_tokens
            sz = new_sz = token_to_kv_pool.available_size() + tree_cache.evictable_size()
            while adder.rem_total_tokens < total_tokens:
                if running_batch is None:
                    raise RuntimeError(
                        "Delta fairness reservation requires a running batch to retract."
                    )
                try:
                    last_evicted, _ = running_batch.retract_decode(
                        extra_space + sz - adder.rem_total_tokens
                    )
                except RuntimeError as exc:
                    if "Delta fairness retraction blocked" in str(exc):
                        logger.info(
                            "Forced prefill skipped for uid=%s rid=%s: %s",
                            req.uid,
                            req.rid,
                            exc,
                        )
                        last_evicted = []
                    else:
                        raise
                if not last_evicted:
                    logger.info(
                        "Unable to retract decode requests for forced reservation; "
                        "skipping forced prefill for uid=%s rid=%s.",
                        req.uid,
                        req.rid,
                    )
                    break
                waiting_queue.extend(last_evicted)
                new_sz = token_to_kv_pool.available_size() + tree_cache.evictable_size()
                adder.expand_capacity(new_sz - sz)
                if max_input_size is not None:
                    adder.rem_input_tokens = min(
                        adder.rem_input_tokens,
                        max(0, max_input_size - adder.log_input_tokens),
                    )
                sz = new_sz
                logger.info(
                    "Evicting tokens for fair reservation. Available=%s Needed=%s",
                    adder.rem_total_tokens,
                    total_tokens,
                )

            # if adder.rem_total_tokens < total_tokens:
            #     continue

            user_tokens = token_counters_by_user.setdefault(req.uid, [])
            extra_for_user = sum(user_tokens)
            res = req.init_next_round_input(
                None if prefix_computed else tree_cache,
                fairness_policy=self,
                fair=True,
                extra_tokens=extra_for_user,
            )
            if res == "rejected":
                raise RuntimeError("Fair reservation request rejected unexpectedly.")

            if max_input_size is not None:
                remaining_input_budget = max(0, max_input_size - adder.log_input_tokens)
                if req.extend_input_len > remaining_input_budget:
                    logger.info(
                        "Forced prefill skipped for uid=%s rid=%s: "
                        "extend_input_len=%s exceeds remaining input budget=%s",
                        req.uid,
                        req.rid,
                        req.extend_input_len,
                        remaining_input_budget,
                    )
                    continue

            user_tokens.append(req.extend_input_len)
            add_res = adder.add_one_req(req, sum(user_tokens))
            if add_res == "rejected":
                raise RuntimeError("Fair reservation request rejected unexpectedly.")

        if extra_space > 0 and last_evicted:
            logger.info(
                "Delta fairness forced %s tokens to stay free (available=%s, evictable=%s).",
                extra_space,
                token_to_kv_pool.available_size(),
                tree_cache.evictable_size(),
            )

        return extra_space, last_evicted

    def sorted_waiting_queue(self, waiting_queue: List["Req"]):
        return waiting_queue

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
        """
        Process waiting queue prefill requests. Will add requests to the adder if we want to run prefill.
        """
        effective_max_running_requests = (
            self.max_running_requests
            if self.max_running_requests is not None
            else max_running_requests
        )
        if effective_max_running_requests is None:
            raise ValueError("max_running_requests must be set for waiting-queue processing.")
        if not self._has_delta_limit():
            return super().process_waiting_queue_prefills(
                waiting_queue,
                adder=adder,
                token_counters_by_user=token_counters_by_user,
                prefix_computed=prefix_computed,
                running_batch=running_batch,
                running_batch_size=running_batch_size,
                max_running_requests=effective_max_running_requests,
                available_req_slots=available_req_slots,
                max_input_size=max_input_size,
            )

        effective_running_limit = min(
            effective_max_running_requests,
            running_batch_size + max(0, available_req_slots),
        )
        if running_batch_size >= effective_running_limit:
            return

        tree_cache = self.tree_cache
        target_tree_cache = None if prefix_computed else tree_cache

        for req in self.sorted_waiting_queue(waiting_queue):
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
                logger.info(
                    "Prefill request uid=%s rid=%s rejected due to KV cache user limit.",
                    req.uid,
                    req.rid,
                )
                continue

            add_res = adder.add_one_req(req, extra_tokens)
            if add_res == "rejected":
                logger.info(
                    "Prefill request uid=%s rid=%s rejected during admission.",
                    req.uid,
                    req.rid,
                )
                continue
            # if add_res is False:
            #     logger.info(
            #         "Prefill request uid=%s rid=%s skipped this round due to "
            #         "remaining token budget; trying next queued request.",
            #         req.uid,
            #         req.rid,
            #     )
            #     continue

            token_counters_by_user.setdefault(req.uid, []).append(req.extend_input_len)

            if (
                not add_res
                or adder.no_remaining_tokens()
                or running_batch_size + len(adder.can_run_list) >= effective_running_limit
            ):
                break
