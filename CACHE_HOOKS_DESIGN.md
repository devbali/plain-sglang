# Cache Hooks & Static Partition Policy

**Date:** 2026-05-07
**Branch:** `verity/minimal-scheduling-hooks`

## Summary

Added a **cache hooks layer** (parallel to the existing scheduler hooks) and implemented the **Static Partition fairness policy** that works across both the cache and scheduler.

---

## New Files

### `python/sglang/srt/cache_hooks/`

| File | Purpose |
|------|---------|
| `__init__.py` | Exports `NoOpCachePolicy`, `StaticPartitionCachePolicy`, `StaticPartitionSchedulingPolicy` |
| `no_op_cache_policy.py` | Base class — all hooks are no-ops; subclass to override |
| `static_partition_policy.py` | Static partition implementations for cache + scheduler |

### `NoOpCachePolicy` Hook Methods

| Method | Called when | Purpose |
|--------|------------|---------|
| `on_cache_reset(cache)` | Cache reset | Reset per-user tracking |
| `on_insert(node, owner_uid)` | Node created in radix tree | Track per-user token counts |
| `can_evict_node(node)` | Node considered for eviction | Return `False` to protect |
| `on_evict(node)` | Node evicted (before deletion) | Decrement per-user counters |
| `on_lock_ref_inc(node)` | Node becomes protected (0→1) | Move tokens to unevictable |
| `on_lock_ref_dec(node)` | Node becomes unprotected (1→0) | Move tokens to evictable |
| `on_decode_kv_alloc(uid, n)` | Decode KV slots allocated | Track decode memory outside tree |
| `on_decode_kv_free(uid, n)` | Decode KV slots freed | Decrement decode tracking |
| `can_admit_request(uid, tokens)` | Before request admitted | Return `False` to reject |
| `get_retract_priority(req)` | Decode retraction candidate | Higher value → retracted first |
| `get_user_total_tokens(uid)` | Query | Total tokens per user |
| `get_user_evictable_tokens(uid)` | Query | Evictable tokens per user |
| `get_all_user_tokens()` | Query | Full snapshot |

---

## Wiring Changes

### `radix_cache.py`
- Added `owner: Optional[str]` to `TreeNode`
- Added `cache_hooks_policy: NoOpCachePolicy` attribute to `RadixCache`
- `evict()`: calls `can_evict_node()` before free, `on_evict()` after
- `inc_lock_ref()`: calls `on_lock_ref_inc()` on 0→1 transition
- `dec_lock_ref()`: calls `on_lock_ref_dec()` on 1→0 transition
- `reset()`: calls `on_cache_reset()`
- `_insert_helper()`: accepts `owner`, sets it on new nodes, calls `on_insert()`
- `_split_node()`: inherits `owner` from child node
- `cache_finished_req()` / `cache_unfinished_req()`: pass `req.uid` as owner in `InsertParams`

### `base_prefix_cache.py`
- Added `owner: Optional[str] = None` to `InsertParams`

### `schedule_batch.py`
- `retract_decode()`: sorts by `get_retract_priority()` from cache hooks

### `scheduler.py`
- `_add_requests_to_batch()`: calls `can_admit_request()` before adding each req
- `init_schedule_policy()`: auto-wires `scheduling_hooks_policy.cache_policy` into `tree_cache.cache_hooks_policy`

---

## Static Partition Policy

### `StaticPartitionCachePolicy(n, max_per_user)`

**Enforces:**
- Each of N users gets `max_per_user` tokens
- A node can only be evicted if its owner is **over** quota
- A new request for user U is rejected if U's **unevictable** tokens + request tokens > max
- Decode retraction prioritizes over-quota users (highest excess first)

**Tracking uses `Threading.Lock`** for thread safety. Token counters track:
- `_user_total[uid]` — all tokens (cache + decode)
- `_user_evictable[uid]` — evictable cache tokens only

### `StaticPartitionSchedulingPolicy(n)`

**Creates and exposes `self.cache_policy`** (auto-wired to tree_cache by scheduler init).

| Hook | Behavior |
|------|----------|
| `on_schedule_prefill()` | Sort queue: users with most available quota first |
| `on_prefill_vs_decode_decision()` | Return `"decode"` if any running user is over quota |

**Usage:**
```bash
--scheduling-policy-path sglang.srt.cache_hooks.static_partition_policy.StaticPartitionSchedulingPolicy
```

### Policy Semantics

| Scenario | Static Partition Behavior |
|----------|--------------------------|
| User A at 40% of quota, wants prefill | ✅ Accepted, high priority |
| User B at 90% of quota, wants prefill | ✅ Accepted, lower priority |
| User C at 110% of quota, wants prefill | ❌ Rejected, stays in waiting queue |
| User C at 110% of quota, node eviction needed | Only User C's nodes targeted for eviction |
| User D at 80% of quota, node eviction needed | User D's nodes **protected** from eviction |
| User C at 110% of quota, decode retraction | User C's requests retracted first |
| User C at 110% of quota, running decode | Scheduler forces decode to free C's space |

---

## Design Principles

1. **Minimal hooks, maximum power:** 14 hook methods cover all cache lifecycle events
2. **Separation of concerns:** Cache policy handles memory tracking; scheduling policy handles queue decisions
3. **Thin coupling:** Hooks receive simple parameters (node, uid, req); no monolithic state objects
4. **Default no-op:** `NoOpCachePolicy` returns pass-through defaults — no overhead for non-fairness use
5. **Thread safety:** Static partition tracking uses a single lock covering all counters

## Comparison with Old (fairinf-sglang) Implementation

| Aspect | Old (`radix_cache.py`) | New (hooks) |
|--------|----------------------|-------------|
| User tracking | Mixed into `RadixCache.__init__` and methods | Separate `NoOpCachePolicy` subclass |
| Eviction control | `evict_from_user()`, `evict_condition` in evict | `can_evict_node()` hook |
| Admission | `reject_based_on_static_limit()` in cache | `can_admit_request()` hook |
| Thread safety | Multiple `UserCounters` with separate locks | Single lock, same counters |
| Policy switching | Code change in radix_cache.py | Replace `cache_hooks_policy` instance |
| Configurability | Constructor params on `RadixCache` | Policy-level constructor params |
