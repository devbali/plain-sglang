from __future__ import annotations

"""Minimal no-op scheduling hook policy.

Subclass NoOpSchedulingPolicy and override individual hook methods to inject
custom logic at scheduler decision points without modifying scheduler.py.

Structural model: mirrors delta_fairness/no_fairness_policy.py from the
fairinf fork, but stripped to only the four scheduler hook points needed for
external policy integration.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


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
