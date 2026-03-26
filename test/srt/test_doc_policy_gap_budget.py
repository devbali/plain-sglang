import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.delta_fairness.delta_fairness_policy import DeltaFairnessPolicy
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.delta_fairness.doc_policy_simulator import (
    RequestDecodeEvent,
    RequestPrefillEvent,
    RequestStartEvent,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


SNAPSHOT_PATH = Path(
    "/home/devbali/fairinf/delta-fair-inference/experiments/3-22-doc-policy/"
    "completion_difference_4/fairinf_delta_1s/doc_policy_pass_snapshots.jsonl"
)


def _mk_req_from_snapshot(row: dict) -> Req:
    req = Req(
        uid=row["uid"],
        rid=row["rid"],
        origin_input_text="",
        origin_input_ids=[1] * int(row["prompt_tokens"]),
    )
    req.output_ids = list(range(int(row["output_tokens"])))
    fill_tokens = row.get("fill_tokens")
    req.fill_ids = None if fill_tokens is None else [1] * int(fill_tokens)
    req.sampling_params = SamplingParams(
        max_new_tokens=int(row.get("max_new_tokens") or 0),
        min_new_tokens=0,
    )
    req.extend_input_len = len(req.origin_input_ids)
    req.waiting_time_in_decodes = row.get("waiting_time_in_decodes")
    return req


def _real_event_from_snapshot(row: dict):
    if row is None:
        return None
    event_type = row["type"]
    kwargs = {
        "req_id": row.get("req_id", ""),
        "end_timestamp": float(row["end_timestamp"]),
    }
    if event_type == "RequestStartEvent":
        return RequestStartEvent(**kwargs)
    if event_type == "RequestPrefillEvent":
        return RequestPrefillEvent(**kwargs)
    if event_type == "RequestDecodeEvent":
        return RequestDecodeEvent(
            req_id=kwargs["req_id"],
            end_timestamp=kwargs["end_timestamp"],
            completion_number=int(row.get("completion_number") or 0),
        )
    raise ValueError(f"Unknown event type: {event_type}")


class TestDocPolicyGapBudget(unittest.TestCase):
    def _load_snapshots(self):
        self.assertTrue(SNAPSHOT_PATH.exists(), f"missing snapshot file: {SNAPSHOT_PATH}")
        return [json.loads(line) for line in SNAPSHOT_PATH.read_text().splitlines() if line.strip()]

    def _new_policy(self):
        return DocPolicy(delta_fairness_n=4, max_running_requests=256)

    def _hydrate_policy_state(self, policy: DocPolicy, snapshot: dict):
        snapshot_state = snapshot["pre"] if "pre" in snapshot else snapshot
        running_reqs = [_mk_req_from_snapshot(row) for row in snapshot_state["running_reqs"]]
        waiting_reqs = [_mk_req_from_snapshot(row) for row in snapshot_state["waiting_reqs"]]
        reqs_by_rid = {req.rid: req for req in running_reqs + waiting_reqs}

        for rid, tracked_row in snapshot_state["tracked_requests"].items():
            req = reqs_by_rid.get(rid)
            if req is None:
                req = Req(
                    uid=tracked_row["uid"],
                    rid=rid,
                    origin_input_text="",
                    origin_input_ids=[],
                )
                req.output_ids = []
                req.fill_ids = None
                req.sampling_params = SamplingParams(max_new_tokens=0, min_new_tokens=0)
                req.extend_input_len = 0
            if rid not in policy.simulator.requests:
                policy.process_new_request(req)
            tracked = policy.simulator.requests[rid]
            tracked.arrival_timestamp = float(tracked_row["arrival_timestamp"])

        for rid, event_row in snapshot_state["most_recent_event_real"].items():
            event = _real_event_from_snapshot(event_row)
            if event is None:
                continue
            if not getattr(event, "req_id", None):
                event.req_id = rid
            policy.simulator.most_recent_event_real[rid] = event

        return snapshot_state, SimpleNamespace(reqs=running_reqs), waiting_reqs

    def _build_policy_with_prehistory(self, snapshots: list[dict], target_index: int):
        policy = self._new_policy()
        warmup_start = max(0, target_index - 3)
        for bootstrap_snapshot in snapshots[warmup_start:target_index]:
            _, bootstrap_running_batch, bootstrap_waiting_reqs = self._hydrate_policy_state(
                policy, bootstrap_snapshot
            )
            self._exercise_policy_api(policy, bootstrap_running_batch, bootstrap_waiting_reqs)
        snapshot_state, running_batch, waiting_reqs = self._hydrate_policy_state(
            policy, snapshots[target_index]
        )
        return policy, snapshot_state, running_batch, waiting_reqs

    def _policy_summary(self, policy: DocPolicy):
        return {
            "has_decode_deadline": policy._has_decode_deadline,
            "max_safe_prefill_tokens": policy._max_safe_prefill_tokens,
            "forced_prefill_rids": tuple(sorted(policy._forced_prefill_rids)),
            "safe_waiting_rids": tuple(req.rid for req in policy._safe_waiting_queue),
            "deadline_rows": tuple(
                (
                    candidate.req.rid,
                    candidate.event_type,
                    round(candidate.start_deadline, 6),
                    round(candidate.deadline, 6),
                )
                for candidate in policy._deadline_queue
            ),
        }

    def _exercise_policy_api(self, policy: DocPolicy, running_batch, waiting_reqs):
        policy.start_of_pass(running_batch, list(waiting_reqs))
        force_prefill_any = policy.fairinf_force_prefill_any_waiting(
            list(waiting_reqs),
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
        )
        force_decode, fair_cap = policy.fairinf_force_decode(
            running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
        )
        sorted_waiting = policy.sorted_waiting_queue(list(waiting_reqs))
        sample_force_prefill = (
            policy.fairinf_force_prefill(
                sorted_waiting[0],
                {},
                delta_fairness_deltas_microseconds=policy._deltas_us,
                running_batch=running_batch,
            )
            if sorted_waiting
            else False
        )
        with patch.object(
            DeltaFairnessPolicy,
            "force_prefill_reservations",
            return_value=(123, ["sentinel"]),
        ) as mock_force:
            reservation_result = policy.force_prefill_reservations(
                list(waiting_reqs),
                token_counters_by_user={},
                adder=SimpleNamespace(can_run_list=[]),
                token_to_kv_pool=None,
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
                max_input_size=None,
                prefix_computed=False,
                max_running_requests=256,
            )
            called_waiting = (
                tuple(req.rid for req in mock_force.call_args[0][0])
                if mock_force.call_args is not None
                else None
            )
        return {
            "summary": self._policy_summary(policy),
            "force_prefill_any": force_prefill_any,
            "force_decode": force_decode,
            "fair_cap": fair_cap,
            "sorted_waiting_rids": tuple(req.rid for req in sorted_waiting),
            "sample_force_prefill": sample_force_prefill,
            "reservation_result": reservation_result,
            "reservation_waiting_rids": called_waiting,
        }

    def test_doc_policy_pass_budget_should_stay_under_five_ms(self):
        snapshots = self._load_snapshots()
        pass_times_ms = []

        for target_index, snapshot in enumerate(snapshots):
            policy, snapshot_state, running_batch, waiting_reqs = (
                self._build_policy_with_prehistory(snapshots, target_index)
            )
            self.assertEqual(len(running_batch.reqs), snapshot_state["running_batch_size"])
            self.assertEqual(len(waiting_reqs), snapshot_state["waiting_queue_size"])

            start = time.perf_counter()
            first_result = self._exercise_policy_api(policy, running_batch, waiting_reqs)
            pass_times_ms.append((time.perf_counter() - start) * 1000.0)

            summary = first_result["summary"]
            self.assertEqual(
                [row[2] for row in summary["deadline_rows"]],
                sorted(row[2] for row in summary["deadline_rows"]),
            )
            decode_deadline_rows = [
                row for row in summary["deadline_rows"] if row[1] == "decode"
            ]
            self.assertEqual(summary["has_decode_deadline"], bool(decode_deadline_rows))
            waiting_rids = {req.rid for req in waiting_reqs}
            self.assertTrue(set(summary["forced_prefill_rids"]).issubset(waiting_rids))
            if summary["safe_waiting_rids"]:
                safe_waiting_deadlines = [
                    policy._waiting_prefill_start_deadline_by_rid.get(rid, float("inf"))
                    for rid in summary["safe_waiting_rids"]
                ]
                self.assertEqual(safe_waiting_deadlines, sorted(safe_waiting_deadlines))
            if summary["max_safe_prefill_tokens"] is not None:
                self.assertGreaterEqual(summary["max_safe_prefill_tokens"], 0)
            if summary["forced_prefill_rids"]:
                self.assertEqual(first_result["reservation_result"], (123, ["sentinel"]))
                self.assertEqual(
                    first_result["reservation_waiting_rids"],
                    tuple(req.rid for req in policy.sorted_waiting_queue(list(waiting_reqs))),
                )
            else:
                self.assertEqual(first_result["reservation_result"], (0, None))
                self.assertIsNone(first_result["reservation_waiting_rids"])

            second_result = self._exercise_policy_api(policy, running_batch, waiting_reqs)
            self.assertEqual(first_result, second_result)

        self.assertTrue(
            all(elapsed_ms < 5.0 for elapsed_ms in pass_times_ms),
            f"doc_policy API replay exceeded 5ms budget on recorded snapshots: {pass_times_ms}",
        )


if __name__ == "__main__":
    unittest.main()
