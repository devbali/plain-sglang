# Per-Token Deadline Model (from SOSP 2026 FairInference paper)

## Overview

FairInference computes per-token deadlines to enforce **τ-token fairness**:
a well-behaved client's token that takes `d` time units in isolation must complete
within `d + τ` time units in multi-tenant execution.

## Delay Tolerance Breakdown

The single configured parameter `τ` is split into components:

| Component | Symbol | Default share | Applies to |
|-----------|--------|---------------|------------|
| Caching delay | `τ_cache` | 80% of τ | First token (KV cache restoration) |
| Prefill scheduling delay | `τ_prefill` | 20% of τ | First token (prefill batch scheduling) |
| Decode scheduling delay | `τ_decode` | fixed (e.g. 80 ms per batch) | Intermediate and last tokens |

For a batch *B*:
```
τ_0 = τ_cache + τ_prefill_B          ∀ k = 0 (first token)
τ_k = τ_decode_B                     ∀ k > 0 (intermediate / last tokens)
```

## Per-Token Deadline Formula

Let `T_ISO_k` be the completion time of the *k*-th token under isolated execution.

**First token:**
```
d(r)_0 = T_ISO_0 + τ_prefill + τ_cache
```

**Subsequent tokens:**
```
d(r)_k = T_ISO_k + τ_decode    ∀ k > 0
```

Where:
```
T_ISO_0     = Δprefill_B                           (prefill batch execution)
T_ISO_k     = T_ISO_0 + Σ Δdecode_B                (sum over previous decode batches)
```

## Relationship to Hook Methods

| Hook | Related Deadline Concept |
|------|-------------------------|
| `on_prefill_vs_decode_decision()` | Must return `"decode"` when the next decode deadline is at risk of being missed due to prefill admission |
| `on_schedule_prefill()` | Should return a queue sorted by earliest deadline first, filtering out clients whose admission would violate running decode deadlines |
| `on_prefill_decision()` | Records that admitted prefills are consuming part of the available headroom |
| `on_decode_decision()` | Records that decode steps are advancing toward deadlines |

## Implementation Notes

- τ is a single configurable parameter (e.g., `--tau-ms 1000`)
- For first token: 80% → τ_cache, 20% → τ_prefill
- τ_decode is independently configured as max acceptable per-batch decode delay
- If the simulator produces deadlines earlier than current time, deadlines are raised to current time (starvation safeguard)
