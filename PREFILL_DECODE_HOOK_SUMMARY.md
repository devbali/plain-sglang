# Prefill vs. Decode Hook Summary

**Date:** 2026-05-07  
**Branch:** `verity/minimal-scheduling-hooks`

## What Was Added

A new high-level scheduling hook that controls **whether to run prefill or decode work**, complementing the existing `on_schedule_prefill` hook that controls **which requests** to prefill.

## The Hook

### `on_prefill_vs_decode_decision`

**Location:** `python/sglang/srt/scheduling_hooks/no_op_policy.py`

```python
def on_prefill_vs_decode_decision(
    self,
    waiting_queue: List[Req],
    running_batch: ScheduleBatch,
    new_prefill_batch: Optional[ScheduleBatch],
) -> Optional[str]:
    """Decide whether to run prefill or decode work.
    
    Returns:
        None -> default (prefill-first if available)
        'prefill' -> run prefill
        'decode' -> run decode (even if prefill available)
    """
    return None
```

**When it's called:**
- After `get_new_batch_prefill()` has created a potential prefill batch
- Before the scheduler decides prefill vs. decode
- Every scheduler pass (even when idle)

**What it can do:**
- ✅ Force decode even when prefill work is available
- ✅ Implement time-slicing between prefill and decode
- ✅ Prevent decode starvation from continuous prefill
- ✅ Balance new requests (prefill) vs. running requests (decode)

## Why This Matters

**Before:** Prefill always wins if available. Decode only runs when no prefill work exists.

**After:** Policy can override this decision to:
- Implement fairness between prefill and decode
- Prevent continuous prefill from starving decode requests
- Balance resource allocation between new and running users

## Helper Methods for Decision-Making

The hook can use scheduler helpers (via `scheduler` reference) to make informed decisions:

```python
# Batch size constraints
scheduler.get_num_allocatable_reqs(running_bs)  # -> slots available
scheduler.running_batch.batch_is_full           # -> at capacity?

# Early exit checks
scheduler._should_skip_prefill()                # -> should we skip prefill?

# Queue state
len(waiting_queue)                              # -> pending prefill work
len(running_batch.reqs)                         # -> active decode work
new_prefill_batch is not None                   # -> prefill work ready?

# Per-user tracking
for req in running_batch.reqs:
    req.uid                                      # -> identify users
```

**Note:** These helpers are available via comments in the hook docstring. For actual usage, the policy needs a `scheduler` reference (see `ResourceAwareFairnessPolicy` pattern in examples).

## Example Use Cases

### 1. Time-Slicing

Alternate between prefill and decode windows to ensure both get fair execution time:

```python
class TimeSlicingPolicy(NoOpSchedulingPolicy):
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        # Alternate every 100ms
        if self.time_in_prefill >= 100:
            return 'decode'
        elif self.time_in_decode >= 100:
            return 'prefill'
        return self.current_mode
```

### 2. Starvation Prevention

Force decode when running requests haven't progressed in N steps:

```python
class StarvationPreventionPolicy(NoOpSchedulingPolicy):
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        # Check if any running request is starved
        if self.max_starvation >= 10:
            return 'decode'
        return None
```

### 3. Resource-Aware Scheduling

Use scheduler helpers to make informed decisions:

```python
class ResourceAwarePolicy(NoOpSchedulingPolicy):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        # Force decode if we're at capacity and decode work exists
        running_bs = len(running_batch.reqs)
        if self.scheduler.get_num_allocatable_reqs(running_bs) <= 0:
            return 'decode'
        return None
```

## How It Works

### Scheduler Flow

```python
def get_next_batch_to_run():
    # 1. Try to create prefill batch
    new_batch = self.get_new_batch_prefill()
    
    # 2. 🪝 Hook decides prefill vs. decode
    decision = self.scheduling_hooks_policy.on_prefill_vs_decode_decision(
        self.waiting_queue,
        self.running_batch,
        new_batch,
    )
    
    # 3. Apply decision
    if decision == 'prefill' or (decision is None and new_batch is not None):
        return new_batch  # Run prefill
    elif decision == 'decode' or (decision is None and new_batch is None):
        return self.running_batch  # Run decode
```

### Decision Logic

| Hook Returns | `new_batch` | Result |
|-------------|-------------|--------|
| `None` | Not `None` | **Prefill** (default: prefill-first) |
| `None` | `None` | **Decode** (default: no prefill available) |
| `'prefill'` | Not `None` | **Prefill** (forced) |
| `'prefill'` | `None` | **None** (no prefill available) |
| `'decode'` | Any | **Decode** (even if prefill available) |

## Two Hooks Working Together

### Hook 1: High-Level Strategy
**`on_prefill_vs_decode_decision`**
- "Should we do prefill or decode work?"
- Controls balance between new requests and running requests
- Time-slicing, starvation prevention

### Hook 2: Low-Level Queue Management
**`on_schedule_prefill`**
- "Which waiting requests should we prefill?"
- Controls queue ordering and filtering
- Fairness, throttling, prioritization

### Example: Hybrid Policy

```python
class HybridFairnessPolicy(NoOpSchedulingPolicy):
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        # High-level: prevent decode starvation
        if self.decode_starved():
            return 'decode'
        return None
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        # Low-level: fairness within prefill
        return self.reorder_by_fairness(waiting_queue, running_batch)
```

## Files Changed

1. **`python/sglang/srt/scheduling_hooks/no_op_policy.py`**
   - Added `on_prefill_vs_decode_decision` method

2. **`python/sglang/srt/managers/scheduler.py`**
   - Wired hook into `get_next_batch_to_run`
   - Added decision logic based on hook return value

3. **`examples/fairness_policy_example.py`**
   - Added `TimeSlicingPolicy` example
   - Added `StarvationPreventionPolicy` example

4. **Documentation**
   - Updated `SCHEDULING_HOOK_DESIGN.md`
   - Updated `CHANGELOG_HOOKS.md`
   - Created this summary

## Testing

Syntax validated:
```bash
python3 -m py_compile python/sglang/srt/scheduling_hooks/no_op_policy.py  # ✓
python3 -m py_compile python/sglang/srt/managers/scheduler.py             # ✓
python3 -m py_compile examples/fairness_policy_example.py                 # ✓
```

## Next Steps

1. Test with real workloads to validate hook behavior
2. Add metrics/logging for hook decisions (prefill vs. decode counts)
3. Consider additional helper methods on `PrefillAdder` for resource queries
4. Potentially expose scheduler reference more cleanly for advanced policies
