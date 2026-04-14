# Delta-Fair Inference: Design Document

This document describes the architecture and implementation of the Doc Policy (deadline-ordered, completion-fair scheduling) used in the fairinf-sglang inference server.

---
### Scheduling

let there be a waiting queue Q for prefills. let there be a running batch B full of decode events.

let Bp be a prefill batch, initialized to empty
while True: # while there is there is space (excluding reservations). This is what the "can fit" is
- if there is a prefill from a user under fair share that is the earliest start deadline event, and it can fit in Bp, add to Bp
- if there is a prefill from a user (checked in deadline order) that can fit before the earliest decode deadline, and it can fit in Bp, add to Bp
- if no such prefills can be found, break

if there is Bp, do prefill
otherwise, do decode

----- 
claude ne bola

Key divergences:
Pseudocode	Code
Single loop with two conditions per iteration	Two separate phases: fairinf_force_decode (prefill vs decode) + process_waiting_queue_prefills (build Bp)
Condition 1 (fair + earliest deadline) and condition 2 (any user + fits before decode) checked on every iteration of the same loop	Condition 1 checked only for the single earliest-deadline request; condition 2 pre-computed offline into forced_prefill_queue and max_safe_prefill_tokens
"can fit in Bp" = space remaining in batch	"can fit" = two separate things: KV space (_reject_based_on_computed_fair_limit) and token budget (rem_total_tokens/max_safe_prefill_tokens)
Unfair users can satisfy condition 2	Unfair users are blocked by PREFILL_PRIORITIZE_FAIR in both the snapshot loop and fairinf_force_decode fallback
Conditions 1 and 2 alternate on each iteration	Condition 1 (override) applies to a single request; condition 2 applies to all _safe_waiting_queue in bulk
The code's logic is more conservative and offline than the pseudocode implies — the "fits before decode deadline" decision (_compute_safe_prefix_state) is made during GPU execution for the next pass, not inline during batch construction.

-----
eval todos
cache difference not showing delta 0 as low
delta violations % not working
try more aggressive policy?
