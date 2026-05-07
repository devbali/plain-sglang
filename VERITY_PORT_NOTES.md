# Verity Port Notes — uid Plumbing & Scheduling Hooks

Branch: `verity/minimal-scheduling-hooks`
Base: upstream SGLang `main` (this repo is a clean upstream clone with no fairinf internals)

---

## Context

This branch ports two minimal, non-invasive additions from the `fairinf-sglang` fork into the upstream repo:

1. **uid plumbing** — a user-identity field threaded from request input objects all the way into the scheduler's `Req` object.
2. **Scheduling hooks** — a thin policy interface that lets external code observe scheduling events without touching scheduler internals.

The reference implementations in `fairinf-sglang` (`delta_fairness/no_fairness_policy.py`, `delta_fairness/doc_policy.py`, `managers/policy_scheduler.py`) were used for structural guidance only. No fairness logic was ported.

---

## uid Plumbing

`uid` is a nullable `str` that identifies the end user making a request. It currently flows through both the generate and embedding request paths as follows:

| Layer | File | Change |
|-------|------|--------|
| API input dataclass | `python/sglang/srt/managers/io_struct.py` | Added `uid: Optional[str] = None` to `GenerateReqInput` and `EmbeddingReqInput` |
| Tokenized request dataclass | `python/sglang/srt/managers/io_struct.py` | Added `uid: Optional[str] = None` to `TokenizedGenerateReqInput` and `TokenizedEmbeddingReqInput` |
| Batch-to-tokenized conversion | `python/sglang/srt/managers/tokenizer_manager.py` | Included `uid=obj.uid` when constructing tokenized generate and embedding requests |
| Batch splitting (`__getitem__`) | `python/sglang/srt/managers/io_struct.py` | Included `uid=self.uid` when splitting batched generate and embedding requests |
| `Req` constructor | `python/sglang/srt/managers/schedule_batch.py` | Added `uid: Optional[str] = None` parameter; stored as `self.uid` |
| Scheduler `Req` construction | `python/sglang/srt/managers/scheduler.py` | Passed `uid=recv_req.uid` when constructing `Req` in `handle_generate_request` and `handle_embedding_request` |

**Design note:** `uid` is a scalar field (not expanded per batch item). For a batch request, all sub-requests share the same `uid` since they originate from the same caller. The `__getitem__` split propagates the scalar directly.

---

## Scheduling Hooks

### New directory: `python/sglang/srt/scheduling_hooks/`

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports `NoOpSchedulingPolicy` |
| `no_op_policy.py` | Base class — all hooks are no-ops; subclass to override |

### Hook methods on `NoOpSchedulingPolicy`

| Method | Called from | When |
|--------|-------------|------|
| `on_new_request(req)` | `scheduler.py` `_add_request_to_queue` | After a request joins `waiting_queue`; `req.uid` and `req.rid` are available |
| `on_prefill_decision(batch)` | `scheduler.py` `get_next_batch_to_run` | Just before a new prefill batch is dispatched to the GPU |
| `on_decode_decision(batch)` | `scheduler.py` `get_next_batch_to_run` | Just before the running decode batch is dispatched |
| `on_end_of_scheduler_pass(batch)` | `scheduler.py` `get_next_batch_to_run` | At the end of every scheduling pass (`batch` is `None` when idle) |

### Wiring in `Scheduler.__init__`

```python
self.scheduling_hooks_policy: NoOpSchedulingPolicy = NoOpSchedulingPolicy()
```

Replace this instance with a subclass to inject custom logic.

---

## Future Hooks (suggested by DocPolicy, not implemented)

The fairinf `DocPolicy` and `NoFairnessPolicy` use additional hooks that were intentionally omitted from this minimal port. They are listed here for reference when extending:

| Hook | DocPolicy source | Purpose |
|------|-----------------|---------|
| `mark_request_finished(req)` | `no_fairness_policy.py:195`, `doc_policy.py:922` | Called when a request completes; useful for per-user completion tracking and simulator updates |
| `note_scheduled_prefill_batch(batch)` | `no_fairness_policy.py:177`, `doc_policy.py:906` | Called after a prefill batch is committed; used to update prepare-thread simulator |
| `note_retracted_reqs(reqs)` | `no_fairness_policy.py:180`, `doc_policy.py:912` | Called when decode requests are retracted to free slots for prefill |
| `start_of_pass(running_batch, waiting_queue, ...)` | `no_fairness_policy.py:183` | Called at the top of each scheduling pass (vs. our end-of-pass hook) |
| `can_admit_running_request(req, ...)` | `no_fairness_policy.py:107` | Per-request admission gate for the running batch |
| `get_retract_order(batch)` | `no_fairness_policy.py:150` | Controls eviction order when retracting decode requests |
| `alloc_token_slots(...)` | `no_fairness_policy.py:55` | Memory allocation hook for per-user KV budget enforcement |
| `check_decode_memory(batch)` | `no_fairness_policy.py:140` | Memory sufficiency check before a decode step |
| `prepare_during_gpu_execution(...)` | `doc_policy.py:787` | Async prepare-thread hook for background scheduler computations |
| `process_waiting_queue_prefills(...)` | `no_fairness_policy.py:339` | Full control over which waiting requests are admitted to prefill |

---

## Extending Later

1. **Subclass `NoOpSchedulingPolicy`** and override the hooks you need. Pass an instance to `scheduler.scheduling_hooks_policy` before the scheduler starts (or expose it via server args).

2. **Add hooks for request completion**: Wire `on_request_finished(req)` into `Scheduler._finish_request` (or equivalent) alongside `mark_request_finished` in the fairinf fork.

3. **Add a `on_retract(reqs)` hook**: The retract path in `schedule_batch.py` (`retract_decode`) is a natural extension point.

4. **Per-user KV accounting**: The fairinf `alloc_token_slots` hook is the right place to enforce per-user memory budgets. The upstream `BaseTokenToKVPool.alloc` call sites in `schedule_batch.py` would need wrapping.

5. **`uid` in OpenAI-compatible adapters**: The internal request structs now support `uid`, but the OpenAI-compatible entrypoints do not yet map an external field into it. Wire that adapter layer if you want callers on those surfaces to set `uid` directly.
