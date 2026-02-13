from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import csv
import heapq
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional, Dict

import torch

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
from sglang.srt.metrics.prefix_match import flush_prefix_match_metrics

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import logging
logger = logging.getLogger(__name__)
DO_CACHE_DEBUG_LOGS = False

CACHE_LOG_FILE = "sglang_cache.csv"
_CACHE_LOG_LOCK = threading.Lock()

@dataclass
class EvictionData:
    input_ids: list
    evicted_ids: list

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.owner = None
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

class UserCounters:
    def __init__ (self):
        self.user_to_token_map : Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def __iter__ (self):
        with self._lock:
            items = list(self.user_to_token_map.items())
        return iter(items)
    
    def add_tokens(self, user_id: str, num_tokens: int):
        with self._lock:
            if user_id not in self.user_to_token_map:
                self.user_to_token_map[user_id] = 0
            self.user_to_token_map[user_id] += num_tokens
    
    def remove_tokens(self, user_id: str, num_tokens: int):
        with self._lock:
            if user_id in self.user_to_token_map:
                self.user_to_token_map[user_id] -= num_tokens
                if self.user_to_token_map[user_id] < 0:
                    self.user_to_token_map[user_id] = 0
    
    def get_tokens(self, user_id: str) -> int:
        with self._lock:
            return self.user_to_token_map.get(user_id, 0)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.user_to_token_map)

class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
        static_max_per_user: int | None = None,
        fairinf_n: int | None = None,
        fairinf_max_per_user: int | None = None,
        fairinf_deltas_microseconds: dict | None = None
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.capacity = token_to_kv_pool.can_use_mem_size
        self.disable = disable
        self.static_max_per_user = static_max_per_user
        self.fairinf_n = fairinf_n
        self.fairinf_max_per_user = fairinf_max_per_user
        if self.fairinf_n is not None and self.fairinf_max_per_user is None:
            self.fairinf_max_per_user = token_to_kv_pool.can_use_mem_size // fairinf_n
        
        if self.fairinf_n is not None:
            self.fairinf_delta_evictable = fairinf_deltas_microseconds.get("prefix_cache", 0)
            self.fairinf_delta_unevictable = fairinf_deltas_microseconds.get("kv_cache", 0)
            logger.info(f"Cache init, maximum cache size is {token_to_kv_pool.can_use_mem_size}. Total reservation is {self.calculate_delta_fair_reservation_size(self.fairinf_delta_evictable)}")


        self.cache_log_path = os.path.abspath(CACHE_LOG_FILE)
        self.reset()
        self._start_cache_usage_thread()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.total_user_counters = UserCounters()
        self.evictable_total_user_counters = UserCounters()

    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node, user=kwargs.get("user"))
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return value, last_node[0]

    def insert(self, key: List, value=None, owner=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, owner)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.disable:
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone(), req.uid)
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone(), req.uid)
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids, user=req.uid)
        assert len(new_indices) == len(token_ids)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, len(req.prefix_indices) : len(new_indices)
        ] = new_indices[len(req.prefix_indices) :]

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict_from_user (self, num_tokens: int, evict_callback: Callable, user_id: str):
        """Evict tokens from a specific user. Would be useful for static partition enforcement."""
        self.evict(
            num_tokens,
            evict_callback,
            evict_condition=lambda x: x.owner == user_id
            )

    def reject_based_on_static_limit (self, user_id: str, num_tokens: int):
        """Check if a new request from user_id with num_tokens would exceed the static limit."""
        if self.static_max_per_user is None:
            return False
        current_tokens = self.total_user_counters.get_tokens(user_id)
        evictable_tokens = self.evictable_total_user_counters.get_tokens(user_id)
        if DO_CACHE_DEBUG_LOGS: logger.debug(f"[FairInf Cache] User {user_id} has {current_tokens} current tokens, {evictable_tokens} evictable tokens, static max {self.static_max_per_user}, requesting {num_tokens} new tokens.")
        if current_tokens - evictable_tokens + num_tokens > self.static_max_per_user:
            return True
        return False
    
    def reject_based_on_computed_fair_limit (self, user_id: str, num_tokens:int):
        if self.fairinf_n is None:
            return False
        REAL_EXPANDABLE_SIZE = self.calculate_real_expandable_size_for_user_fairinf (user_id, [])
        logger.info(f"IN REJECT BASED ON COMPUTED FAIR LIMIT: num_tokens: {num_tokens}, user_id: {user_id}, real expandable size: {REAL_EXPANDABLE_SIZE}")
        return REAL_EXPANDABLE_SIZE  < num_tokens
    
    def reject_based_on_computed_fair_limit_unfair (self, user_id, num_tokens):
        if self.fairinf_n is None:
            return False
        REAL_EXPANDABLE_SIZE = self.calculate_real_expandable_size_for_user_fairinf (user_id, [], self_unfair=True)
        logger.info(f"IN REJECT BASED ON COMPUTED FAIR LIMIT UNFAIR: num_tokens: {num_tokens}, user_id: {user_id}, real expandable size: {REAL_EXPANDABLE_SIZE}")
        return REAL_EXPANDABLE_SIZE  < num_tokens
    
    def calculate_delta_fair_reservation_size (self, time_microseconds):
        TOKENS_PER_MICROSECOND = 1 / 60
        RESERVED_SIZE = min(int(self.fairinf_max_per_user * 0.9), int(self.fairinf_max_per_user - time_microseconds * TOKENS_PER_MICROSECOND))
        return max(0, RESERVED_SIZE)

    def calculate_real_expandable_size_for_user_fairinf (self, user_id, fair_users, self_unfair=False):
        # Exclude reservations for others
        # Exclude unevictables for your own self
        
        RESERVATION_SIZE = self.calculate_delta_fair_reservation_size(self.fairinf_delta_evictable)
        expandable_size = self.capacity
        for user, token_count in self.total_user_counters:
            unevictable = (token_count - self.evictable_total_user_counters.get_tokens(user))
            if user == user_id:
                expandable_size -= unevictable
            else:
                reserved_size = min(token_count, RESERVATION_SIZE)
                if user in fair_users or self_unfair:
                    # You can not evict this user's running requests, so count either reservation size, or unevictable whichever is larger
                    expandable_size -= max(reserved_size, unevictable)
                else:
                    expandable_size -= reserved_size

        return expandable_size

    def evict_delta_fair (self, num_tokens: int, evict_callback: Callable, requesting_users):
        # This method is only called on evictable nodes
        if self.fairinf_max_per_user is None:
            return self.evict(num_tokens, evict_callback)
        
        logger.info(f"Evict delta fair num_tokens: {num_tokens}, requesting_users: {requesting_users}")
        
        RESERVATION_SIZE = self.calculate_delta_fair_reservation_size(self.fairinf_delta_evictable)
        if requesting_users is None:
            requesting_users = []

        def evict_condition (node):
            # if this returns false this node can not be evicted
            return not self.user_total_is_under_fair_share_reservation(
                node.owner, 
                fair_share_override=RESERVATION_SIZE
            ) or "user_" not in node.owner

        def evict_condition_2 (node):
            # if this returns false this node can not be evicted
            return not self.user_total_is_under_fair_share_reservation(
                node.owner, 
                fair_share_override=RESERVATION_SIZE
            ) or node.owner in requesting_users
        
        
        num_evicted = self.evict(
            num_tokens,
            evict_callback,
            evict_condition=evict_condition
        )

        if num_evicted < num_tokens and requesting_users:
            logger.info(f"Tried to evict from {requesting_users}, could not evict just from above fair share, evicting from their quota")
            num_evicted += self.evict(
                num_tokens - num_evicted,
                evict_callback,
                evict_condition=evict_condition_2
            )
        
        if num_evicted < num_tokens:
            return None
        return num_tokens
        
        # If we reach here, it should be that running decodes of the running users should be evicted. But for now let's just evict normally
        logger.info(f"Evicting normally for users: {requesting_users}. Should not happen, temporary fix")
        self.evict(
            num_tokens - num_evicted,
            evict_callback,
        )

    def evict_from_users_exceeding_limit(self, num_tokens: int, evict_callback: Callable, limit: int | None = None):
        """Evict tokens from users whose token count exceeds the limit. Would be useful for Delta Fairness based eviction."""
        if limit is None and self.fairinf_max_per_user is not None:
            limit = self.fairinf_max_per_user

        self.evict(
            num_tokens,
            evict_callback,
            evict_condition=lambda x: self.total_user_counters.get_tokens(x.owner) + self.evictable_total_user_counters.get_tokens(x.owner) > limit
        )


    def evict(self, num_tokens: int, evict_callback: Callable, evict_condition : Optional[Callable] = None):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                logger.warning("[Fairinf KV Cache] Reached root node during eviction. No more evictable nodes.")
                break

            if evict_condition is not None and not evict_condition(x):
                logger.debug(f"[Fairinf KV Cache] Skipping eviction of node {x} belonging to {x.owner} due to condition.")
                continue

            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)
        return num_evicted

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.evictable_total_user_counters.remove_tokens(node.owner, len(node.value))
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.evictable_total_user_counters.add_tokens(node.owner, len(node.value))
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def user_unevictable_kv_is_under_fair_share_reservation (self, user_id: str, extra_tokens = 0) -> bool:
        if self.fairinf_n is None:
            return False
        MAX = self.calculate_delta_fair_reservation_size(self.fairinf_delta_unevictable)
        return self.total_user_counters.get_tokens(user_id) - self.evictable_total_user_counters.get_tokens(user_id) + extra_tokens < MAX

    def user_total_is_under_fair_share_reservation (self, user_id: str, extra_tokens = 0, fair_share_override=None) -> bool:
        if self.fairinf_n is None:
            return False
        if fair_share_override is not None:
            MAX = self.fairinf_max_per_user
        else:
            MAX = fair_share_override
        return self.total_user_counters.get_tokens(user_id) + extra_tokens < MAX

    def fairinf_evictable_size_pooled (self):
        # Returns "evictable" size for Fair Inference, i.e., evictable tokens from users
        #   that are exceeding their reserved share. This is only to be used for users
        #   that are also above the reserved size, otherwise evictions should be forced
        if self.fairinf_max_per_user is None:
            return self.evictable_size()

        size = 0
        for user_id, token_count in self.total_user_counters:
            if token_count > self.fairinf_max_per_user:
                # Tokens exceeding the fair share are evictable
                evictable_count = self.evictable_total_user_counters.get_tokens(user_id)
                evictable_count = min(evictable_count, token_count - self.fairinf_max_per_user)
                size += evictable_count
        return size

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####

    def _start_cache_usage_thread(self):
        if hasattr(self, "_cache_logger_thread"):
            return

        self._cache_logger_thread = threading.Thread(
            target=self._cache_usage_logger_loop, daemon=True
        )
        self._cache_logger_thread.start()

    def _cache_usage_logger_loop(self):
        while True:
            try:
                self._log_current_cache_usage()
                flush_prefix_match_metrics()
            except Exception:
                logger.exception("[Fairinf KV Cache] Failed to log cache usage.")
            time.sleep(1)

    def _log_current_cache_usage(self):
        total_snapshot = self.total_user_counters.snapshot()
        evictable_snapshot = self.evictable_total_user_counters.snapshot()

        if not total_snapshot and not evictable_snapshot:
            return

        timestamp = datetime.utcnow().isoformat()
        rows = []
        all_users = set(total_snapshot.keys()) | set(evictable_snapshot.keys())
        for user_id in all_users:
            total_tokens = total_snapshot.get(user_id, 0)
            evictable_tokens = evictable_snapshot.get(user_id, 0)
            unevictable_tokens = max(total_tokens - evictable_tokens, 0)
            rows.append([timestamp, "evictable", user_id, evictable_tokens])
            rows.append([timestamp, "unevictable", user_id, unevictable_tokens])

        if rows:
            self._append_cache_usage_rows(rows)

    def _append_cache_usage_rows(self, rows):
        dir_name = os.path.dirname(self.cache_log_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with _CACHE_LOG_LOCK:
            file_exists = os.path.exists(self.cache_log_path)
            with open(self.cache_log_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                if not file_exists:
                    writer.writerow(["timestamp", "kind", "user", "bytes"])
                writer.writerows(rows)

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode, user=None
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len, owner_parent=user)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node, user=user)

    def _split_node(self, key, child: TreeNode, split_len: int, owner_parent=None, owner_new=None):
        # new_node -> child
        if DO_CACHE_DEBUG_LOGS: logger.debug(f"[Fairinf KV Cache] Splitting node at length {split_len}, child: owner={child.owner}, length={len(child.value)}={len(child.key)}. Owner parent = {owner_parent} Owner new = {owner_new}")
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.owner = child.owner

        if owner_parent is not None and new_node.owner != owner_parent:
            # Parent owner changes to the parent user
            self.total_user_counters.remove_tokens(new_node.owner, split_len)
            self.total_user_counters.add_tokens(owner_parent, split_len)
            new_node.owner = owner_parent
            
        if owner_new is not None and child.owner != owner_new:
            length = len(child.value)-split_len
            self.total_user_counters.remove_tokens(child.owner, length)
            self.total_user_counters.add_tokens(owner_new, length)
            child.owner = owner_new

        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        if child.owner is None:
            child.owner = new_node.owner
            self.total_user_counters.add_tokens(child.owner, len(child.key))
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value, owner):
        if DO_CACHE_DEBUG_LOGS: logger.debug(f"[Fairinf KV Cache] Inserting key of length {len(key)} under owner {owner}")
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                # Change ownership of child to this new owner
                if child.owner != owner:
                    self.total_user_counters.remove_tokens(child.owner, prefix_len)
                    self.total_user_counters.add_tokens(owner, prefix_len)
                    child.owner = owner

                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value, owner)

            if DO_CACHE_DEBUG_LOGS: logger.debug(f"[Fairinf KV Cache] Calling split node from insert, prefix_len={prefix_len}, child key length={len(child.key)}")
            new_node = self._split_node(child.key, child, prefix_len, owner_parent=owner)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:], owner
            )

        if len(key):
            new_node = TreeNode()
            new_node.owner = owner
            self.total_user_counters.add_tokens(owner, len(value))
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
            self.evictable_total_user_counters.add_tokens(owner, len(value))
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        self.total_user_counters.remove_tokens(node.owner, len(node.key))
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)
        self.evictable_total_user_counters.remove_tokens(node.owner, len(node.key))

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
