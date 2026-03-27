import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
from sglang.srt.delta_fairness.doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
)
from sglang.srt.managers.schedule_batch import Req


def _mk_req(uid: str, rid: str, n_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * n_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    return req


class TestDocPolicySimulatorUnit(unittest.TestCase):
    def test_grouped_finished_decode_replays_each_decode_round(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time):
            with patch.object(
                sim_mod, "isolated_prefill_time_estimation", return_value=2.0
            ), patch.object(
                sim_mod, "isolated_decode_time_estimation", return_value=3.0
            ):
                simulator = AlternateHistorySimulator(
                    max_kv_tokens_per_user=100,
                    fairinf_n=2,
                )

                req = _mk_req("user_19", "rid_running", 4)
                simulator.process_new_request(req)
                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req]))
                req.output_ids = list(range(10))
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req]), decode_rounds=10)

                now["t"] = 100.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req]),
                    [],
                )

                candidates, _ = simulator.build_deadline_candidates(
                    [],
                    SimpleNamespace(reqs=[req]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                tracked = simulator.requests[req.rid]
                decode_events = [
                    event
                    for event in tracked.alternate_history_timeline.history
                    if isinstance(event, RequestDecodeEvent)
                ]
                anticipated_decode_events = [
                    event
                    for event in tracked.alternate_history_timeline.anticipated_future_events
                    if isinstance(event, RequestDecodeEvent)
                ]
                self.assertEqual(
                    [event.completion_number for event in decode_events],
                    list(range(1, 11)),
                )
                self.assertEqual(
                    [event.end_timestamp for event in decode_events],
                    [15.0 + 3.0 * i for i in range(10)],
                )
                self.assertEqual(
                    [event.completion_number for event in anticipated_decode_events],
                    [11],
                )
                decode_candidates = [
                    candidate for candidate in candidates if candidate.event_type == "decode"
                ]
                self.assertEqual(len(decode_candidates), 1)
                self.assertEqual(decode_candidates[0].event.completion_number, 11)
                self.assertEqual(decode_candidates[0].deadline, 45.0)

    def test_waiting_prefill_deadlines_follow_isolated_arrival_order_not_pass_time(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time):
            with patch.object(
                sim_mod, "isolated_prefill_time_estimation", return_value=2.0
            ), patch.object(
                sim_mod, "isolated_decode_time_estimation", return_value=3.0
            ):
                simulator = AlternateHistorySimulator(
                    max_kv_tokens_per_user=100,
                    fairinf_n=2,
                )

                req1 = _mk_req("user_1", "rid_1", 4)
                req2 = _mk_req("user_1", "rid_2", 4)

                simulator.process_new_request(req1)
                now["t"] = 11.0
                simulator.process_new_request(req2)

                now["t"] = 100.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[]),
                    [req1, req2],
                )

                candidates, waiting_deadlines = simulator.build_deadline_candidates(
                    [req1, req2],
                    SimpleNamespace(reqs=[]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                self.assertEqual(waiting_deadlines[req1.rid], 10.0)
                self.assertEqual(waiting_deadlines[req2.rid], 12.0)
                self.assertEqual(
                    [candidate.req.rid for candidate in candidates if candidate.event_type == "prefill"],
                    [req1.rid, req2.rid],
                )

                req1_prefill = [
                    event
                    for event in simulator.requests[req1.rid].alternate_history_timeline.anticipated_future_events
                    if isinstance(event, RequestPrefillEvent)
                ]
                req2_prefill = [
                    event
                    for event in simulator.requests[req2.rid].alternate_history_timeline.anticipated_future_events
                    if isinstance(event, RequestPrefillEvent)
                ]
                self.assertEqual(req1_prefill[0].end_timestamp, 12.0)
                self.assertEqual(req2_prefill[0].end_timestamp, 14.0)

    def test_running_decode_deadline_comes_from_isolated_sequence_not_wall_clock(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time):
            with patch.object(
                sim_mod, "isolated_prefill_time_estimation", return_value=2.0
            ), patch.object(
                sim_mod, "isolated_decode_time_estimation", return_value=3.0
            ):
                simulator = AlternateHistorySimulator(
                    max_kv_tokens_per_user=100,
                    fairinf_n=2,
                )

                req = _mk_req("user_19", "rid_running", 4)
                simulator.process_new_request(req)
                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req]))
                req.output_ids = [42]
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req]))

                now["t"] = 100.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req]),
                    [],
                )

                candidates, _ = simulator.build_deadline_candidates(
                    [],
                    SimpleNamespace(reqs=[req]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                decode_candidates = [
                    candidate for candidate in candidates if candidate.event_type == "decode"
                ]
                self.assertEqual(len(decode_candidates), 1)
                self.assertIsInstance(decode_candidates[0].event, RequestDecodeEvent)
                self.assertEqual(decode_candidates[0].event.completion_number, 2)
                self.assertEqual(decode_candidates[0].deadline, 18.0)

                tracked = simulator.requests[req.rid]
                decode_events = [
                    event
                    for event in tracked.alternate_history_timeline.history
                    if isinstance(event, RequestDecodeEvent)
                ]
                anticipated_decode_events = [
                    event
                    for event in tracked.alternate_history_timeline.anticipated_future_events
                    if isinstance(event, RequestDecodeEvent)
                ]
                self.assertEqual(
                    [event.completion_number for event in decode_events],
                    [1],
                )
                self.assertEqual(
                    [event.completion_number for event in anticipated_decode_events],
                    [2],
                )
                self.assertEqual(
                    [event.end_timestamp for event in decode_events],
                    [15.0],
                )
                self.assertEqual(
                    [event.end_timestamp for event in anticipated_decode_events],
                    [18.0],
                )


if __name__ == "__main__":
    unittest.main()
