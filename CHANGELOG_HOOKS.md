# Changelog: Scheduling Hook Control Flow

**Date:** 2026-05-07  
**Branch:** `verity/minimal-scheduling-hooks`

## Summary

Added **two control hooks** and refactored scheduler helpers to make the main scheduling pass clean and readable:

1. **`on_prefill_vs_decode_decision`** — Control whether to run prefill or decode work
2. **`on_schedule_prefill`** — Control which waiting requests get prefilled

## Changes

### 1. New Control Hook: `on_prefill_vs_decode_decision`

**File:** `python/sglang/srt/scheduling_hooks/no_op_policy.py`

```python
def on_prefill_vs_decode_decision(
    self,
    waiting_queue: List[Req],
    running_batch: ScheduleBatch,
    new_prefill_batch: Optional[ScheduleBatch],
) -> Optional[str]:
    """Return 'prefill', 'decode', or None for default logic."""
    return None
```

**When:** After `get_new_batch_prefill()`, before deciding whether to run prefill or decode

**Purpose:**
- ✅ Force decode even when prefill work is available
- ✅ Implement time-slicing between prefill and decode
- ✅ Prevent decode starvation from continuous prefill
- ✅ Balance new requests (prefill) vs. running requests (decode)

**Return values:**
- `None` → default (prefill-first if available)
- `'prefill'` → run prefill
- `'decode'` → run decode (even if prefill available)

### 2. New Control Hook: `on_schedule_prefill`

**File:** `python/sglang/srt/scheduling_hooks/no_op_policy.py`

```python
def on_schedule_prefill(
    self,
    waiting_queue: List[Req],
    running_batch: ScheduleBatch,
    prefill_adder: PrefillAdder,
) -> Optional[List[Req]]:
    """Called at prefill scheduling time with full scheduler context.
    
    Return None to proceed with default queue, or a filtered/reordered
    list to override the scheduler's selection.
    """
    return None  # baseline: no override
```

**When:** After `calc_priority()` sorts the queue, before requests are added to the batch

**Purpose:** 
- ✅ Reorder queue (fairness policies)
- ✅ Filter requests (per-user throttling, quotas)
- ✅ Replace queue entirely (custom scheduling)
- ✅ Inspect resource constraints via `prefill_adder`

**Contrast with existing hooks:**
- `on_new_request` — observes arrivals (can't control scheduling)
- `on_prefill_decision` — observes final selection (too late to change)
- **`on_schedule_prefill`** — **controls** which requests get selected

### 2. Scheduler Refactoring

**File:** `python/sglang/srt/managers/scheduler.py`

Extracted helper methods from the monolithic `_get_new_batch_prefill_raw`:

| Helper | Lines | Purpose |
|--------|-------|---------|
| `_prepare_prefill_context()` | ~15 | Grammar + HiCache setup |
| `_should_skip_prefill()` | ~25 | Early exit checks |
| `_get_chunked_prefill_size()` | ~10 | Dynamic chunking logic |
| `_create_prefill_adder()` | ~30 | PrefillAdder construction |
| `_process_chunked_request()` | ~12 | Chunked prefill handling |
| `_add_requests_to_batch()` | ~50 | Main request-adding loop |
| `_check_can_add_request()` | ~12 | Memory/batch size checks |
| `_check_hicache_ready()` | ~8 | HiCache prefetch status |
| `_handle_add_request_failure()` | ~20 | OOM cleanup |
| `_finalize_prefill_batch()` | ~90 | Queue updates, batch creation |

**New main flow** (~40 lines, was ~250):

```python
def _get_new_batch_prefill_raw(...):
    # 1. Prepare grammar + cache
    self._prepare_prefill_context()
    
    # 2. Early exit if nothing to schedule
    if self._should_skip_prefill():
        return None
    
    # 3. Apply upstream policy
    self.policy.calc_priority(self.waiting_queue, self.running_batch)
    
    # 4. Build resource manager
    adder = self._create_prefill_adder(...)
    
    # 5. 🪝 Hook override point
    queue_to_schedule = self.scheduling_hooks_policy.on_schedule_prefill(
        self.waiting_queue, self.running_batch, adder
    )
    if queue_to_schedule is None:
        queue_to_schedule = self.waiting_queue
    
    # 6. Process chunked request
    self._process_chunked_request(adder)
    
    # 7. Add requests to batch
    self._add_requests_to_batch(adder, queue_to_schedule)
    
    # 8. Finalize batch
    return self._finalize_prefill_batch(adder)
```

## Why This Design?

### Comparison with fairinf fork approach

**fairinf approach (multi-hook):**
- `start_of_pass()` — observes state
- `process_waiting_queue_prefills()` — iterates queue, calls per-request gates
- `can_admit_running_request()` — per-request admission control
- Multiple scattered decision points

**Our approach (single control hook):**
- `on_schedule_prefill()` — one hook with full context
- Hook returns filtered/reordered queue
- Scheduler walks the returned queue
- Single clear decision point

### Benefits

1. **Simpler mental model:** "What should we schedule?" vs. scattered per-request callbacks
2. **Full visibility:** Hook sees waiting queue + running state + resource constraints
3. **Flexible:** Can reorder, filter, or replace queue
4. **Minimal invasiveness:** One insertion point vs. multiple call sites
5. **Readable main logic:** Helper methods hide complexity

### Trade-offs

- Hook needs to understand `PrefillAdder` structure (but only high-level APIs)
- Slightly more method calls (negligible vs. GPU work)

## Usage Examples

See `SCHEDULING_HOOK_DESIGN.md` and `examples/fairness_policy_example.py` for detailed examples:

**Prefill vs. Decode Control:**
- Time-slicing policy (alternate between prefill/decode windows)
- Starvation prevention (force decode when running requests starved)

**Prefill Queue Control:**
- Fairness policy (deprioritize users already running)
- Per-user throttling (max concurrent requests)
- Custom prioritization logic

**Combined:**
- Hybrid policies using both hooks together

## Next Steps

Potential future additions:
- Helper methods on `PrefillAdder` for common queries (`can_fit_tokens`, `remaining_slots`, etc.)
- Similar refactoring for decode path if needed
- Additional hooks for request completion, retraction (see `VERITY_PORT_NOTES.md` "Future Hooks" section)
