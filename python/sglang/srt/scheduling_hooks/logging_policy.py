"""Logging scheduling policy — logs every hook call for testing."""

import logging
from typing import List, Optional

from sglang.srt.scheduling_hooks.no_op_policy import NoOpSchedulingPolicy
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


class LoggingSchedulingPolicy(NoOpSchedulingPolicy):
    """A policy that logs every hook invocation with request info.

    Used for testing that all four scheduling hook points fire correctly.
    """

    def __init__(self):
        super().__init__()
        self._call_log: List[str] = []
        self._seen_uids: set = set()
        self.hook_counts = {"on_new_request": 0, "on_prefill_decision": 0, "on_decode_decision": 0, "on_end_of_scheduler_pass": 0}

    def _log(self, msg: str):
        logger.info(f"[HOOK] {msg}")
        self._call_log.append(msg)
        print(f"[HOOK] {msg}", flush=True)

    def on_new_request(self, req: Req):
        self.hook_counts["on_new_request"] += 1
        uid = getattr(req, "uid", None) or "(no uid)"
        self._seen_uids.add(uid)
        self._log(f"on_new_request: rid={req.rid} uid={uid} input_len={len(req.origin_input_text) if req.origin_input_text else 0}")

    def on_prefill_decision(self, batch: Optional[ScheduleBatch]):
        self.hook_counts["on_prefill_decision"] += 1
        if batch is not None:
            reqs = getattr(batch, 'reqs', [])
            uids = [getattr(r, 'uid', '(none)') for r in reqs]
            self._log(f"on_prefill_decision: {len(reqs)} reqs, uids={uids}")
        else:
            self._log("on_prefill_decision: no batch (empty)")

    def on_decode_decision(self, batch: Optional[ScheduleBatch]):
        self.hook_counts["on_decode_decision"] += 1
        if batch is not None:
            reqs = getattr(batch, 'reqs', [])
            uids = [getattr(r, 'uid', '(none)') for r in reqs]
            self._log(f"on_decode_decision: {len(reqs)} reqs, uids={uids}")
        else:
            self._log("on_decode_decision: no batch (empty)")

    def on_end_of_scheduler_pass(self, batch: Optional[ScheduleBatch]):
        self.hook_counts["on_end_of_scheduler_pass"] += 1
        batch_desc = f"{len(batch.reqs) if batch and hasattr(batch, 'reqs') else 0} reqs" if batch else "idle"
        self._log(f"on_end_of_scheduler_pass: {batch_desc}")

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "HOOK CALL SUMMARY",
            f"  on_new_request:           {self.hook_counts['on_new_request']}",
            f"  on_prefill_decision:      {self.hook_counts['on_prefill_decision']}",
            f"  on_decode_decision:       {self.hook_counts['on_decode_decision']}",
            f"  on_end_of_scheduler_pass: {self.hook_counts['on_end_of_scheduler_pass']}",
            f"  Unique UIDs seen:         {sorted(self._seen_uids)}",
            "=" * 50,
        ]
        return "\n".join(lines)
