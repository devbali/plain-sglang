## The Doc Policy: `DocPolicy`

`DocPolicy` extends `DeltaFairnessPolicy` and is the main scheduling policy class. It owns both a **main-thread simulator** (used for hot-path operations) and a **`_DocPolicyPrepareWorker`** (for async pre-computation).

### Key state

- `simulator: AlternateHistorySimulator` — the main-thread simulator, always current
- `_deadline_queue` — list of `DeadlineCandidate` (from last consumed snapshot or live build)
- `_safe_waiting_queue`, `_safe_waiting_rids` — EDF-ordered waiting queue
- `_forced_prefill_queue`, `_forced_prefill_rids` — requests that must be prefilled immediately
- `_max_safe_prefill_tokens` — total safe prefill budget
- `_has_decode_deadline`, `_earliest_decode_start_deadline` — whether a decode is overdue
- `_last_consumed_prepare_snapshot_seq` — seq of last consumed snapshot (prevents re-consuming)

### `start_of_pass(running_batch, waiting_queue, ...)`

Called at the beginning of each scheduler pass.

1. Call `super().start_of_pass(...)` for base fairness bookkeeping.
2. Get new pass state from `_consume_prepared_pass_state(waiting_queue, running_batch)`, this is blocking

### `prepare_during_gpu_execution(event_type, running_batch, waiting_queue, ...)`

Called while the GPU is executing (between `start_of_pass` and the next pass). Enqueues async work so the next `start_of_pass` can consume a fresh snapshot.

For **decode** events:
1. Enqueue a `logical_decode_update` mutation.

For **prefill** events:
1. Enqueue a mutation: `event_type="prefill"`, `prepare_pass_state=True`, including the scheduled batch so the worker can predict the post-prefill state.

### `finished_prefill(batch)`
Called after a real prefill completes. Advances the **main-thread simulator**:
- Enqueues a `finished_prefill` mutation (no-op in worker, handled via task overlay instead).

### `finished_decode(batch)`
Called after a real decode completes. Minimal work — the logical decode events are already applied in `prepare_during_gpu_execution`.

### `process_new_request(req)`
Called when a new request enters the waiting queue:
Enqueue a `process_new_request` mutation with `(req_snapshot, deltas_us, arrival_ts)`.

### `_consume_prepared_pass_state(waiting_queue, running_batch)`
Attempts to fetch the most recent snapshot from the worker thread. Blocks till it is ready.

### Hot-path decode: `refresh_decode_hot_path_state`

Called during the decode hot path (between decode steps) to pick up a fresher snapshot without blocking. Only calls `_consume_prepared_pass_state`.

### `sorted_waiting_queue(waiting_queue)`

Returns the waiting queue sorted by EDF priority: requests in `_safe_waiting_queue` come first (in their precomputed order), followed by any new requests not yet in the snapshot.

### `fairinf_force_decode(running_batch, ...)`

Returns `(True, 0)` (force decode, no prefill budget) if there's an overdue decode deadline and no safe prefill tokens. Otherwise returns `(False, _max_safe_prefill_tokens)`.

### `fairinf_force_prefill(req, ...)`

Returns True if `req.rid in _forced_prefill_rids` and the user still satisfies fairness constraints.

### `force_prefill_reservations(...)`

Orchestrates forced prefills for overdue requests: evicts decode requests from the running batch if needed to create KV memory space, then returns the list of forced-prefill requests.


## Other functions
There may be other functions that are called directly from the main sglang files such as tp_worker.py
Leave those unchanged
