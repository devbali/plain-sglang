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
    maxDiff = None

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

    def _clone_req(self, row: dict, *, rid: str) -> Req:
        clone = _mk_req_from_snapshot({**row, "rid": rid})
        return clone

    def _clone_event(self, event_row: dict | None, *, rid: str, ts_bump: float = 0.0):
        if event_row is None:
            return None
        clone = dict(event_row)
        clone["req_id"] = rid
        clone["end_timestamp"] = float(clone["end_timestamp"]) + ts_bump
        return _real_event_from_snapshot(clone)

    def _inflate_policy_state(
        self,
        policy: DocPolicy,
        snapshot_state: dict,
        running_batch,
        waiting_reqs,
        *,
        target_running: int,
        target_waiting: int,
    ):
        running_rows = list(snapshot_state["running_reqs"])
        waiting_rows = list(snapshot_state["waiting_reqs"])

        running_reqs = list(running_batch.reqs)
        waiting_reqs = list(waiting_reqs)

        if running_rows:
            idx = 0
            while len(running_reqs) < target_running:
                src = running_rows[idx % len(running_rows)]
                new_rid = f"{src['rid']}__runclone_{len(running_reqs)}"
                req = self._clone_req(src, rid=new_rid)
                running_reqs.append(req)
                policy.process_new_request(req)
                tracked = policy.simulator.requests[new_rid]
                tracked.arrival_timestamp = float(
                    snapshot_state["tracked_requests"][src["rid"]]["arrival_timestamp"]
                ) + (idx + 1) * 1e-6
                event = self._clone_event(
                    snapshot_state["most_recent_event_real"].get(src["rid"]),
                    rid=new_rid,
                    ts_bump=(idx + 1) * 1e-6,
                )
                if event is not None:
                    policy.simulator.most_recent_event_real[new_rid] = event
                idx += 1

        if waiting_rows:
            idx = 0
            while len(waiting_reqs) < target_waiting:
                src = waiting_rows[idx % len(waiting_rows)]
                new_rid = f"{src['rid']}__waitclone_{len(waiting_reqs)}"
                req = self._clone_req(src, rid=new_rid)
                waiting_reqs.append(req)
                policy.process_new_request(req)
                tracked = policy.simulator.requests[new_rid]
                tracked.arrival_timestamp = float(
                    snapshot_state["tracked_requests"][src["rid"]]["arrival_timestamp"]
                ) + (idx + 1) * 1e-6
                event = self._clone_event(
                    snapshot_state["most_recent_event_real"].get(src["rid"]),
                    rid=new_rid,
                    ts_bump=(idx + 1) * 1e-6,
                )
                if event is not None:
                    policy.simulator.most_recent_event_real[new_rid] = event
                idx += 1

        return SimpleNamespace(reqs=running_reqs), waiting_reqs

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
        timings_ms = {}

        start = time.perf_counter()
        policy.start_of_pass(running_batch, list(waiting_reqs))
        timings_ms["start_of_pass_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        force_prefill_any = policy.fairinf_force_prefill_any_waiting(
            list(waiting_reqs),
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
        )
        timings_ms["force_prefill_any_waiting_ms"] = (
            time.perf_counter() - start
        ) * 1000.0

        start = time.perf_counter()
        force_decode, fair_cap = policy.fairinf_force_decode(
            running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
        )
        timings_ms["force_decode_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        sorted_waiting = policy.sorted_waiting_queue(list(waiting_reqs))
        timings_ms["sorted_waiting_queue_ms"] = (
            time.perf_counter() - start
        ) * 1000.0

        start = time.perf_counter()
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
        timings_ms["force_prefill_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
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
        timings_ms["force_prefill_reservations_ms"] = (
            time.perf_counter() - start
        ) * 1000.0

        timings_ms["start_of_pass_breakdown_ms"] = dict(
            getattr(policy, "_last_pass_breakdown_ms", {})
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
            "timings_ms": timings_ms,
        }

    def _exercise_prepare_and_policy_api(self, policy: DocPolicy, running_batch, waiting_reqs):
        prepare_start = time.perf_counter()
        policy.prepare_during_gpu_execution(
            event_type="decode",
            running_batch=running_batch,
            waiting_queue=list(waiting_reqs),
            scheduled_batch=None,
            selected_rids={req.rid for req in running_batch.reqs},
        )
        prepare_elapsed_ms = (time.perf_counter() - prepare_start) * 1000.0
        result = self._exercise_policy_api(policy, running_batch, waiting_reqs)
        result["timings_ms"]["prepare_during_gpu_execution_ms"] = prepare_elapsed_ms
        result["timings_ms"]["prepare_breakdown_ms"] = dict(
            getattr(policy, "_last_prepare_breakdown_ms", {})
        )
        return result

    def _exercise_decode_prepare_epoch(self, policy: DocPolicy, running_batch, waiting_reqs):
        step_times_ms = []
        step_breakdowns = []
        for step_idx in range(10):
            prepare_start = time.perf_counter()
            policy.prepare_during_gpu_execution(
                event_type="decode",
                running_batch=running_batch,
                waiting_queue=list(waiting_reqs),
                scheduled_batch=None,
                selected_rids={req.rid for req in running_batch.reqs},
                prepare_pass_state=(step_idx == 9),
            )
            step_times_ms.append((time.perf_counter() - prepare_start) * 1000.0)
            step_breakdowns.append(dict(getattr(policy, "_last_prepare_breakdown_ms", {})))
            for req in running_batch.reqs:
                req.output_ids.append(0)

        post_prepare_result = self._exercise_policy_api(policy, running_batch, waiting_reqs)
        return {
            "prepare_step_times_ms": tuple(step_times_ms),
            "prepare_epoch_total_ms": sum(step_times_ms),
            "prepare_breakdowns": tuple(step_breakdowns),
            "post_prepare_result": post_prepare_result,
        }

    def test_doc_policy_pass_budget_should_stay_under_five_ms(self):
        snapshots = self._load_snapshots()
        critical_path_times_ms = []
        prepare_times_ms = []

        for target_index, snapshot in enumerate(snapshots):
            policy, snapshot_state, running_batch, waiting_reqs = (
                self._build_policy_with_prehistory(snapshots, target_index)
            )
            self.assertEqual(len(running_batch.reqs), snapshot_state["running_batch_size"])
            self.assertEqual(len(waiting_reqs), snapshot_state["waiting_queue_size"])

            first_result = self._exercise_prepare_and_policy_api(
                policy, running_batch, waiting_reqs
            )
            total_elapsed_ms = (
                first_result["timings_ms"]["prepare_during_gpu_execution_ms"]
                + first_result["timings_ms"]["start_of_pass_ms"]
                + first_result["timings_ms"]["force_prefill_any_waiting_ms"]
                + first_result["timings_ms"]["force_decode_ms"]
                + first_result["timings_ms"]["sorted_waiting_queue_ms"]
                + first_result["timings_ms"]["force_prefill_ms"]
                + first_result["timings_ms"]["force_prefill_reservations_ms"]
            )
            critical_elapsed_ms = (
                first_result["timings_ms"]["start_of_pass_ms"]
                + first_result["timings_ms"]["force_prefill_any_waiting_ms"]
                + first_result["timings_ms"]["force_decode_ms"]
                + first_result["timings_ms"]["sorted_waiting_queue_ms"]
                + first_result["timings_ms"]["force_prefill_ms"]
                + first_result["timings_ms"]["force_prefill_reservations_ms"]
            )
            critical_path_times_ms.append(critical_elapsed_ms)
            prepare_times_ms.append(
                first_result["timings_ms"]["prepare_during_gpu_execution_ms"]
            )
            print(
                f"snapshot={target_index} total_ms={total_elapsed_ms:.3f} "
                f"prepare_ms={first_result['timings_ms']['prepare_during_gpu_execution_ms']:.3f} "
                f"critical_path_ms={critical_elapsed_ms:.3f} "
                f"timings={first_result['timings_ms']}",
                flush=True,
            )

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

            policy2, _, running_batch2, waiting_reqs2 = self._build_policy_with_prehistory(
                snapshots, target_index
            )
            second_result = self._exercise_prepare_and_policy_api(
                policy2, running_batch2, waiting_reqs2
            )
            comparable_first = dict(first_result)
            comparable_second = dict(second_result)
            comparable_first.pop("timings_ms", None)
            comparable_second.pop("timings_ms", None)
            if comparable_first != comparable_second:
                print(f"replay_mismatch snapshot={target_index}", flush=True)
                print(
                    f"first_summary={first_result['summary']}",
                    flush=True,
                )
                print(
                    f"second_summary={second_result['summary']}",
                    flush=True,
                )
                print(
                    f"first_timings={first_result['timings_ms']}",
                    flush=True,
                )
                print(
                    f"second_timings={second_result['timings_ms']}",
                    flush=True,
                )
                first_rows = first_result["summary"]["deadline_rows"]
                second_rows = second_result["summary"]["deadline_rows"]
                for idx, (row1, row2) in enumerate(zip(first_rows, second_rows)):
                    if row1 != row2:
                        print(
                            f"deadline_row_diff index={idx} first={row1} second={row2}",
                            flush=True,
                        )
                        break
                if len(first_rows) != len(second_rows):
                    print(
                        f"deadline_row_count first={len(first_rows)} second={len(second_rows)}",
                        flush=True,
                    )
            self.assertEqual(comparable_first, comparable_second)

        self.assertTrue(
            all(elapsed_ms < 5.0 for elapsed_ms in critical_path_times_ms[1:]),
            "doc_policy API replay critical path exceeded 5ms budget on recorded "
            f"snapshots: {critical_path_times_ms}",
        )
        self.assertTrue(
            all(elapsed_ms < 20.0 for elapsed_ms in prepare_times_ms[1:]),
            "doc_policy prepare_during_gpu_execution exceeded 20ms budget on recorded "
            f"snapshots: {prepare_times_ms}",
        )

    def test_doc_policy_prepare_decode_epoch_should_stay_under_two_hundred_ms(self):
        snapshots = self._load_snapshots()
        epoch_totals_ms = []

        for target_index, snapshot in enumerate(snapshots):
            policy, snapshot_state, running_batch, waiting_reqs = (
                self._build_policy_with_prehistory(snapshots, target_index)
            )
            self.assertEqual(len(running_batch.reqs), snapshot_state["running_batch_size"])
            self.assertEqual(len(waiting_reqs), snapshot_state["waiting_queue_size"])

            epoch_result = self._exercise_decode_prepare_epoch(
                policy, running_batch, waiting_reqs
            )
            epoch_totals_ms.append(epoch_result["prepare_epoch_total_ms"])
            print(
                f"prepare_epoch snapshot={target_index} "
                f"total_ms={epoch_result['prepare_epoch_total_ms']:.3f} "
                f"step_times_ms={epoch_result['prepare_step_times_ms']}",
                flush=True,
            )
            max_step_index, max_step_ms = max(
                enumerate(epoch_result["prepare_step_times_ms"]),
                key=lambda item: item[1],
            )
            print(
                f"prepare_epoch snapshot={target_index} "
                f"worst_step={max_step_index} worst_step_ms={max_step_ms:.3f} "
                f"worst_breakdown={epoch_result['prepare_breakdowns'][max_step_index]} "
                f"last_breakdown={epoch_result['prepare_breakdowns'][-1]} "
                f"post_start_of_pass_ms={epoch_result['post_prepare_result']['timings_ms']['start_of_pass_ms']:.3f}",
                flush=True,
            )
            print(
                f"prepare_epoch snapshot={target_index} "
                f"post_prepare_summary={epoch_result['post_prepare_result']['summary']}",
                flush=True,
            )

        self.assertTrue(
            all(elapsed_ms < 200.0 for elapsed_ms in epoch_totals_ms),
            "doc_policy 10-step decode prepare epoch exceeded 200ms budget on recorded "
            f"snapshots: {epoch_totals_ms}",
        )

    def test_doc_policy_prepare_decode_stress_replay(self):
        snapshots = self._load_snapshots()
        target_index = 4
        policy, snapshot_state, running_batch, waiting_reqs = self._build_policy_with_prehistory(
            snapshots, target_index
        )
        running_batch = SimpleNamespace(reqs=list(running_batch.reqs[:4]))
        running_batch, waiting_reqs = self._inflate_policy_state(
            policy,
            snapshot_state,
            running_batch,
            waiting_reqs,
            target_running=4,
            target_waiting=705,
        )
        prepare_start = time.perf_counter()
        policy.prepare_during_gpu_execution(
            event_type="decode",
            running_batch=running_batch,
            waiting_queue=list(waiting_reqs),
            scheduled_batch=None,
            selected_rids={req.rid for req in running_batch.reqs},
        )
        prepare_elapsed_ms = (time.perf_counter() - prepare_start) * 1000.0
        print(
            "stress_prepare "
            f"running={len(running_batch.reqs)} waiting={len(waiting_reqs)} "
            f"tracked={len(policy.simulator.requests)} "
            f"prepare_ms={prepare_elapsed_ms:.3f} "
            f"breakdown={getattr(policy, '_last_prepare_breakdown_ms', {})}",
            flush=True,
        )
        self.assertGreater(len(waiting_reqs), 700)


if __name__ == "__main__":
    unittest.main()
