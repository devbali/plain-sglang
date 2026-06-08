# Scheduling Hook Design

## Overview

Added **two control-flow hooks** for scheduling decisions, plus helper method refactoring to keep the main scheduler pass clean and readable:

1. **`on_prefill_vs_decode_decision`** — Control whether to run prefill or decode work
2. **`on_schedule_prefill`** — Control which waiting requests get prefilled

## Changes

### 1. New Hook: `on_prefill_vs_decode_decision`

**Location:** `python/sglang/srt/scheduling_hooks/no_op_policy.py`

```python
def on_prefill_vs_decode_decision(
    self,
    waiting_queue: List[Req],
    running_batch: ScheduleBatch,
    new_prefill_batch: Optional[ScheduleBatch],
) -> Optional[str]:
    """Return 'prefill', 'decode', or None for default logic."""
    return None  # baseline: prefill-first if available
```

**When it's called:**
- After `get_new_batch_prefill()` has created a potential prefill batch
- Before the scheduler decides whether to run prefill or decode
- On every scheduler pass (even when idle)

**What it can do:**
- ✅ Force decode even when prefill work is available
- ✅ Force prefill (though default does this already)
- ✅ Implement time-slicing between prefill and decode
- ✅ Prevent decode starvation from continuous prefill pressure
- ✅ Balance work between new requests (prefill) and running requests (decode)

**Return values:**
- `None` → default behavior (prefill-first if available, else decode)
- `'prefill'` → run prefill (only if `new_prefill_batch` is not None)
- `'decode'` → run decode (even if prefill work is available)

**Helper information available:**
- `len(waiting_queue)` — pending prefill work
- `len(running_batch.reqs)` — active decode work  
- `new_prefill_batch is not None` — whether prefill work is ready
- `req.uid` on `running_batch.reqs` — identify users needing decode
- Scheduler methods (via `scheduler` reference):
  - `scheduler.get_num_allocatable_reqs(running_bs)` — batch size headroom
  - `scheduler.running_batch.batch_is_full` — at capacity?
  - `scheduler._should_skip_prefill()` — early exit checks

### 2. New Hook: `on_schedule_prefill`

**Location:** `python/sglang/srt/scheduling_hooks/no_op_policy.py`

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

**When it's called:**
- After upstream `calc_priority()` has sorted the waiting queue
- Before requests are walked and added to the batch
- With full visibility into waiting queue, running state, and resource constraints

**What it can do:**
- ✅ Reorder the waiting queue (fairness-based prioritization)
- ✅ Filter out requests (enforce per-user quotas, throttling)
- ✅ Completely replace the queue (custom scheduling logic)
- ✅ Inspect `prefill_adder` for memory/token budget info

**What it can't do:**
- ❌ Modify running batch directly
- ❌ Allocate memory (PrefillAdder handles that)

### 2. Scheduler Refactoring

**Location:** `python/sglang/srt/managers/scheduler.py`

Extracted helper methods to make `_get_new_batch_prefill_raw()` readable:

| Helper Method | Purpose |
|---------------|---------|
| `_prepare_prefill_context()` | Grammar manager + HiCache setup |
| `_should_skip_prefill()` | Early exit checks (batch full, empty queue, etc.) |
| `_get_chunked_prefill_size()` | Dynamic chunking logic |
| `_create_prefill_adder()` | PrefillAdder construction |
| `_process_chunked_request()` | Handle chunked prefill continuation |
| `_add_requests_to_batch()` | Main loop: walk queue, add requests |
| `_check_can_add_request()` | Memory + batch size limit checks |
| `_check_hicache_ready()` | HiCache prefetch status |
| `_handle_add_request_failure()` | OOM/failure cleanup |
| `_finalize_prefill_batch()` | Queue updates, batch creation, stats |

**New main flow in `get_next_batch_to_run`:**

```python
def get_next_batch_to_run(...):
    # ... filter running batch ...
    
    # Try to create prefill batch
    new_batch = self.get_new_batch_prefill()
    
    # 🪝 Hook 1: Decide prefill vs. decode
    decision = self.scheduling_hooks_policy.on_prefill_vs_decode_decision(
        self.waiting_queue, self.running_batch, new_batch
    )
    
    should_run_prefill = (
        (decision == 'prefill') or
        (decision is None and new_batch is not None)
    )
    should_run_decode = (
        (decision == 'decode') or
        (decision is None and new_batch is None)
    )
    
    if should_run_prefill and new_batch is not None:
        ret = new_batch
        self.scheduling_hooks_policy.on_prefill_decision(ret)
    elif should_run_decode:
        ret = self.running_batch
        self.scheduling_hooks_policy.on_decode_decision(ret)
    
    # ... finalize and return ...
```

**Prefill batch creation flow in `_get_new_batch_prefill_raw`:**

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
    
    # 5. 🪝 Hook 2: Override queue selection
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

## How to Use

### Example 1: Time-Slicing (prefill vs. decode)

```python
from sglang.srt.scheduling_hooks import NoOpSchedulingPolicy
import time

class TimeSlicingPolicy(NoOpSchedulingPolicy):
    def __init__(self, prefill_window_ms=100, decode_window_ms=100):
        super().__init__()
        self.prefill_window_ms = prefill_window_ms
        self.decode_window_ms = decode_window_ms
        self.last_switch_time = 0
        self.current_mode = 'prefill'
    
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        # No work available -> use default
        if new_prefill_batch is None or running_batch.is_empty():
            return None
        
        current_time = time.time() * 1000
        time_in_mode = current_time - self.last_switch_time
        
        # Switch modes based on time windows
        if self.current_mode == 'prefill' and time_in_mode >= self.prefill_window_ms:
            self.current_mode = 'decode'
            self.last_switch_time = current_time
            return 'decode'
        elif self.current_mode == 'decode' and time_in_mode >= self.decode_window_ms:
            self.current_mode = 'prefill'
            self.last_switch_time = current_time
            return 'prefill'
        
        return self.current_mode
```

### Example 2: Starvation Prevention

```python
class StarvationPreventionPolicy(NoOpSchedulingPolicy):
    def __init__(self, max_steps_without_progress=10):
        super().__init__()
        self.max_steps = max_steps_without_progress
        self.request_last_step = {}
        self.global_step = 0
    
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        self.global_step += 1
        
        if running_batch.is_empty():
            return None
        
        # Check if any running request has been starved
        max_starvation = 0
        for req in running_batch.reqs:
            last_step = self.request_last_step.get(req.rid, self.global_step)
            starvation = self.global_step - last_step
            max_starvation = max(max_starvation, starvation)
        
        # Force decode if any request is starved
        if max_starvation >= self.max_steps:
            return 'decode'
        
        return None
    
    def on_decode_decision(self, batch):
        for req in batch.reqs:
            self.request_last_step[req.rid] = self.global_step
```

### Example 3: Fairness Policy (queue reordering)

```python
from sglang.srt.scheduling_hooks import NoOpSchedulingPolicy

class FairnessPolicy(NoOpSchedulingPolicy):
    def __init__(self):
        super().__init__()
        self.user_tokens = {}  # uid -> tokens served
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        # Track running users
        running_users = {req.uid for req in running_batch.reqs}
        
        # Deprioritize users already running
        def priority_key(req):
            if req.uid in running_users:
                return (1, self.user_tokens.get(req.uid, 0))  # lower priority
            else:
                return (0, self.user_tokens.get(req.uid, 0))  # higher priority
        
        # Return reordered queue
        return sorted(waiting_queue, key=priority_key)
```

### Example: Per-User Throttling

```python
class ThrottlingPolicy(NoOpSchedulingPolicy):
    def __init__(self, max_concurrent_per_user=2):
        super().__init__()
        self.max_concurrent = max_concurrent_per_user
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        # Count running requests per user
        running_counts = {}
        for req in running_batch.reqs:
            running_counts[req.uid] = running_counts.get(req.uid, 0) + 1
        
        # Filter out over-quota users
        filtered = [
            req for req in waiting_queue
            if running_counts.get(req.uid, 0) < self.max_concurrent
        ]
        
        return filtered
```

## Benefits

1. **Two clean decision points:** 
   - Prefill vs. decode (higher-level scheduling strategy)
   - Which requests to prefill (lower-level queue management)
2. **Full visibility:** Hooks see waiting queue, running state, and resource constraints
3. **Clean main logic:** Helper methods hide complexity, main flow is readable pseudocode
4. **Minimal changes:** Upstream scheduler logic preserved, hooks are clean insertion points
5. **Flexible:** 
   - Control prefill/decode balance (time-slicing, starvation prevention)
   - Reorder, filter, or replace prefill queue (fairness, throttling)
6. **Complementary hooks:** Both can be used together for sophisticated policies

## Trade-offs

- Hook needs to understand `PrefillAdder` (but only high-level: resource budgets, not KV cache internals)
- Slightly more method calls (but negligible overhead vs. GPU work)

## Next Steps

- Add similar helper extraction for decode path if needed
- Add `PrefillAdder` helper methods for common queries:
  - `adder.can_fit_tokens(num_tokens)` → memory check
  - `adder.remaining_slots()` → batch size headroom
  - etc.
