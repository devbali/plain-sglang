from __future__ import annotations

"""Minimal no-op scheduling hook policy.

Subclass NoOpSchedulingPolicy and override individual hook methods to inject
custom logic at scheduler decision points without modifying scheduler.py.

Structural model: mirrors delta_fairness/no_fairness_policy.py from the
fairinf fork, but stripped to only the four scheduler hook points needed for
external policy integration.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder


class NoOpSchedulingPolicy:
    """Default pass-through policy — all hooks are no-ops.

    Wire one instance into Scheduler via scheduler.scheduling_hooks_policy.
    Override specific methods in a subclass; the scheduler calls each hook at
    the appropriate point in the scheduling pass.
    """

    def on_new_request(self, req: "Req") -> None:
        """Called when a new request is enqueued to the waiting queue.

        req.uid and req.rid are available. Use this to initialize per-user
        state, record arrival timestamps, etc.
        """

    def on_prefill_vs_decode_decision(
        self,
        waiting_queue: List["Req"],
        running_batch: "ScheduleBatch",
        new_prefill_batch: Optional["ScheduleBatch"],
    ) -> Optional[str]:
        """Called to decide whether to run prefill or decode work.

        Args:
            waiting_queue: Requests waiting to be scheduled
            running_batch: Currently running decode batch
            new_prefill_batch: Prefill batch that's ready to run (or None)

        Returns:
            None to use default logic (prefill-first if available),
            'prefill' to force prefill,
            'decode' to force decode (even if prefill work is available)

        Use this to implement fairness policies that balance prefill vs. decode work.
        
        Examples:
        - Force decode when running batch has starved users
        - Implement time-slicing between prefill and decode
        - Skip prefill when decode queue has urgent requests

        Helper information available via scheduler methods:
        - scheduler.get_num_allocatable_reqs(running_bs) → batch size headroom
        - scheduler.running_batch.batch_is_full → whether at capacity
        - scheduler._should_skip_prefill() → early exit checks
        - len(waiting_queue) → pending prefill work
        - len(running_batch.reqs) → active decode work
        - Check req.uid on running_batch.reqs → identify starved users
        """
        return None

    def on_schedule_prefill(
        self,
        waiting_queue: List["Req"],
        running_batch: "ScheduleBatch",
        prefill_adder: "PrefillAdder",
    ) -> Optional[List["Req"]]:
        """Called at prefill scheduling time with full scheduler context.

        Args:
            waiting_queue: Requests waiting to be scheduled (already sorted by policy)
            running_batch: Currently running batch
            prefill_adder: Resource manager with memory/token budget info

        Returns:
            None to proceed with the default queue, or a filtered/reordered
            list of requests to override the scheduler's selection.

        Use this to implement fairness policies, custom prioritization, or
        resource-aware scheduling decisions.
        """
        return None

    def on_prefill_decision(self, batch: "ScheduleBatch") -> None:
        """Called just before a prefill batch is dispatched to the GPU.

        batch.reqs contains the requests selected for prefill. Use this to
        record which requests were admitted, update per-user counters, etc.
        """

    def on_decode_decision(self, batch: "ScheduleBatch") -> None:
        """Called just before a decode batch is dispatched to the GPU.

        batch.reqs contains the currently running decode requests. Use this
        to record decode-step events, update deadlines, etc.
        """

    def on_end_of_scheduler_pass(self, batch: Optional["ScheduleBatch"]) -> None:
        """Called at the end of every scheduling pass (get_next_batch_to_run).

        batch is the batch that will be run (None if the scheduler is idle).
        Use this for per-pass housekeeping, logging, etc.
        """
