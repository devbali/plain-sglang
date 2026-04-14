## The Prepare Worker: `_DocPolicyPrepareWorker`

`_DocPolicyPrepareWorker` owns the single `AlternateHistorySimulator` instance and runs all heavyweight computation off the scheduler's critical path. It has two input queues and one output slot.

---

### Queues and Output

- **`_mutation_queue`** — lightweight state updates. Applied immediately when seen, always before any task. Each entry is a `(kind, payload)` tuple.
- **`_task_queue`** — snapshot build requests. Each entry is a 5-field tuple: `(waiting_reqs, running_batch, frozen_cache_state, frozen_inputs, requested_mutation_seq)`. Every task always produces a snapshot.
- **`_published_snapshot`** — the latest `_PreparedSnapshot`, protected by `_published_snapshot_lock`. Written by the worker thread only; read non-blocking by the main thread via `latest_snapshot()`.

---

### Mutations

Mutations update the simulator's ground-truth state. They are enqueued by the main thread and applied by the worker before processing any task. All payloads use `_PrepareReq` objects (plain-int snapshots, no torch tensors).

| Kind | Sent from | Payload | What the worker does |
|------|-----------|---------|----------------------|
| `process_new_request` | `DocPolicy.process_new_request` | `(req, deltas_us, arrival_ts)` | Calls `simulator.process_new_request(req, deltas_us, arrival_timestamp=arrival_ts)` — seeds the request's isolated timeline with a `RequestStartEvent` |
| `note_retracted_reqs` | `DocPolicy.note_retracted_reqs` | `([reqs], deltas_us)` | Re-registers each retracted request via `simulator.process_new_request` — resets its timeline as if it just arrived |
| `logical_decode_update` | `DocPolicy.prepare_during_gpu_execution` (decode path) | `(running_batch, decode_steps)` | Calls `simulator.finished_decode(running_batch)` **N times** then `rebuild_from_real_state()` once on every user timeline — advances the real decode count N steps and refreshes anticipated events in a single mutation |
| `finished_prefill` | `DocPolicy.finished_prefill` (via mutation) | `(batch,)` | Calls `simulator.finished_prefill(batch)` — commits a `RequestPrefillEvent` to each request's history and seeds the first anticipated decode |
| `mark_request_finished` | `DocPolicy.mark_request_finished` | `(req, pass_id)` | Calls `simulator.mark_request_finished(req)` — moves the request to `finished_request_timelines` and removes it from live tracking |
| `note_scheduled_prefill_batch` | `DocPolicy.note_scheduled_prefill_batch` | `[reqs]` | No-op |
| `finished_decode` | (legacy) | — | No-op |

---

### Tasks

A task is enqueued exactly **twice per scheduler pass**: once at the start of the decode epoch, and once during the prefill forward. Both always produce a snapshot (`prepare_pass_state` is no longer a field — every task builds a snapshot).

| Sent from | When | What the worker does |
|-----------|------|----------------------|
| `DocPolicy.prepare_during_gpu_execution` (decode path) | **Once, at the start of the N-decode hot-path loop** in `tp_worker.py`, before any decode step runs | Drains pending mutations (including the N `logical_decode_update` mutations enqueued just before), then runs `_build_prepare_snapshot` |
| `DocPolicy.prepare_during_gpu_execution` (prefill path) | **During the GPU prefill forward** (between `model_forward_start.record()` and `synchronize()`) | Same — drains any pending mutations (including `finished_prefill`), then runs `_build_prepare_snapshot` |

**Task tuple fields** (5 fields):
1. `waiting_reqs` — `List[_PrepareReq]`, one per waiting request
2. `running_batch` — `SimpleNamespace(reqs=[_PrepareReq, ...])` or `None`
3. `frozen_cache_state` — `_FrozenPrepareCacheState` (KV token counters snapshot for fairness checks)
4. `frozen_inputs` — `_FrozenPrepareInputs` (deltas, no-retraction cap, token ratio, fairinf_n)
5. `requested_mutation_seq` — the mutation seq at enqueue time (informational)

---

### Worker Loop `_worker_loop`

Runs on a daemon thread with `torch.set_grad_enabled(False)` and `torch.inference_mode()`:

1. **Drain mutations**: while `_mutation_queue` is non-empty, pop and `_apply_mutation`. This always runs before any task.
2. **Pop a task**: dequeue `(task_seq, task)` from `_task_queue`.
3. **Drain remaining mutations** that arrived after the task was enqueued.
4. **Build snapshot**: call `_build_prepare_snapshot(waiting_queue, running_batch, ...)` and publish via `_publish_snapshot`.

There is no "skip snapshot" path — every task produces a snapshot.

---

### Snapshot Builder `_build_prepare_snapshot`

Called for every task:

1. **`simulator.get_live_users(running_batch, waiting_queue, ...)`** — syncs which requests are tracked. Creates `TrackedRequest`s for new ones, drops stale ones, updates `requests_real` from timeline history.
2. **`user_timeline.rebuild_from_real_state()`** for every user — runs the per-user isolated scheduler forward (up to 200 steps) to produce `next_anticipated_event` for each live request.
3. **`simulator.build_deadline_candidates(...)`** — scans all tracked requests, generates `DeadlineCandidate` objects, sorts by `(start_deadline, event_type_priority, deadline, arrival_timestamp)`, returns `(deadline_queue, waiting_prefill_deadline_by_rid, ordered_waiting_queue)`.
4. **`owner._compute_safe_prefix_state(...)`** — walks `ordered_waiting_queue` in EDF order, greedily selects prefills that complete before the earliest decode deadline.
5. Packages everything into a `_PreparedSnapshot` and calls `_publish_snapshot`.

---

### `_PrepareReq` — Torch Safety

All payloads passed to the worker use `_PrepareReq` instead of live `Req` objects. `_PrepareReq` stores only plain Python `int` values via `_LenOnlySeq` — it never holds references to token lists or tensors. This ensures the worker thread cannot accidentally touch PyTorch's autograd graph.

Constructed directly on the main thread via `_PrepareReq.from_req(req)` — a few `int(len(...))` copies, no dict serialization roundtrip.

---

### `_PreparedSnapshot` Fields

Frozen dataclass written by the worker, consumed by the main thread. Holds `_PrepareReq` objects (not live `Req`s) — the main thread remaps by rid on consume.

| Field | Meaning |
|-------|---------|
| `task_seq` | Monotonic ID of the task that produced this snapshot |
| `mutation_seq` | How many mutations had been applied when snapshot was built |
| `waiting_sig`, `running_sig` | Tuple of rids — used for staleness detection on consume |
| `deadline_queue` | Tuple of `DeadlineCandidate`, sorted by EDF priority |
| `waiting_prefill_deadlines` | `rid → start_deadline` for waiting requests |
| `safe_waiting_queue` | EDF-ordered waiting requests that fit before decode deadline |
| `safe_waiting_rids` | frozenset of the above |
| `forced_prefill_queue` | Subset of safe queue that must be prefilled now |
| `forced_prefill_rids` | frozenset of the above |
| `max_safe_prefill_tokens` | Total prompt tokens of forced prefills |
| `has_fair_waiting` | True if any request is in `safe_waiting_queue` |
| `has_decode_deadline` | True if any decode deadline exists |
| `earliest_decode_start_deadline` | Wall-clock time by which a decode must start |
| `earliest_decode_rid`, `earliest_decode_uid` | Request/user with the soonest decode deadline |
| `safe_prefix_now` | Timestamp at which the safe-prefix scan was run |
| `breakdown_items` | Tuple of `(metric_name, ms)` for latency accounting |

---

### Snapshot Consumption (`DocPolicy._consume_prepared_pass_state`)

Called at `start_of_pass` (blocking wait if needed) and opportunistically during `refresh_decode_hot_path_state`:

1. Rejects if: no snapshot exists; `task_seq <= last_consumed`; `task_seq < _last_prepare_task_seq`.
2. Remaps `candidate.req` from `_PrepareReq` to live `Req` by rid lookup.
3. **If `waiting_sig` matches** (common): re-runs `_compute_safe_prefix_state` with live objects so forced-prefill headroom checks use real `ScheduleBatch` data. Deadlines come from the snapshot.
4. **If `waiting_sig` doesn't match** (queue changed mid-flight): uses snapshot deadline values directly, re-sorts waiting queue by deadline, remaps forced prefills by rid.
5. Updates `_safe_waiting_*`, `_forced_prefill_*`, `_max_safe_prefill_tokens`, `_has_decode_deadline`, etc. on `DocPolicy`.

---

### Backpressure and Flow Control

- Both queues cap at 100 entries: `_maybe_wait_for_queue_capacity` busy-waits on the main thread if a queue is full before enqueuing.
- `wait_for_mutation_queue_below_limit(100)` is called at `start_of_pass` to stall if mutations are piling up faster than the worker drains them.
- `wait_for_snapshot(min_task_seq, timeout_s)` — busy-waits with 1 ms sleep until the published snapshot's `task_seq >= min_task_seq`. Used in `start_of_pass` to block on the snapshot from the previous GPU execution phase.

---

### Threading Invariants

- **Only the worker thread** writes to `_simulator`. The main thread never calls simulator methods directly.
- **`_published_snapshot`** is guarded by `_published_snapshot_lock`. The worker writes; the main thread reads via `latest_snapshot()`.
- Worker runs with `torch.set_grad_enabled(False)` and all task/mutation processing inside `torch.inference_mode()`, ensuring no gradient tracking ever touches the worker thread.
