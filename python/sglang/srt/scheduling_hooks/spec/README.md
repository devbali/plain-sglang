# Scheduling Hooks — Policy Spec

This directory contains extracted policy specifications from the
**SOSP 2026 FairInference** paper (Paper #965).

| File | Content |
|------|---------|
| `deadline_model.md` | Per-token deadline computation (τ-fairness formulae) |
| `tau_fair_scheduler.md` | τ-Token Fair Scheduler algorithm and hook mapping |
| `scheduling_policy_api.md` | Complete API specification for each scheduling hook |

## Core Fairness Concepts

- **τ-token fairness**: A well-behaved client's token generated in `d` time units in isolation must complete within `d + τ` in multi-tenant execution.
- **τ** is a single configurable delay tolerance, split 80/20 into caching and prefill scheduling components for the first token.
- **Deadline** = isolated latency + tolerable delay.
- **Fair share**: Each client gets 1/N of total GPU resources.
- **Well-behaved client**: Usage ≤ fair share.
