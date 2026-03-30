import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
from sglang.srt.delta_fairness.doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
    RequestStartEvent,
    RequestStatusReal,
    RequestTimeline,
    TrackedRequest,
    UserTimeline,
)
from sglang.srt.managers.schedule_batch import Req


def _mk_req(uid: str, rid: str, n_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * n_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    return req


class TestDocPolicySimulatorUnit(unittest.TestCase):
    def test_process_new_request_logs_isolated_start_from_simulation_time(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_start"
        ) as mark_start:
            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=100,
                fairinf_n=2,
            )

            req = _mk_req("user_19", "rid_running", 4)
            simulator.process_new_request(req)

            mark_start.assert_called_once_with(
                req.rid,
                req.uid,
                timestamp_iso="1970-01-01T00:00:10.000+00:00",
            )

    def test_finished_events_log_isolated_timestamps_from_simulation_time(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done"
        ) as mark_prefill, patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"
        ) as mark_decode:
            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=100,
                fairinf_n=2,
            )

            req = _mk_req("user_19", "rid_running", 4)
            simulator.process_new_request(req)

            now["t"] = 11.0
            simulator.finished_prefill(SimpleNamespace(reqs=[req]))

            req.output_ids = [42, 43]
            now["t"] = 12.0
            simulator.finished_decode(SimpleNamespace(reqs=[req]), decode_rounds=2)

            mark_prefill.assert_called_once_with(
                req.rid,
                req.uid,
                timestamp_iso="1970-01-01T00:00:12.000+00:00",
            )
            self.assertEqual(mark_decode.call_count, 2)
            self.assertEqual(
                [call.kwargs["timestamp_iso"] for call in mark_decode.call_args_list],
                [
                    "1970-01-01T00:00:15.000+00:00",
                    "1970-01-01T00:00:18.000+00:00",
                ],
            )
            self.assertEqual(
                [call.kwargs["completion_number"] for call in mark_decode.call_args_list],
                [1, 2],
            )

    def test_mark_request_finished_logs_isolated_completion_from_simulation_time(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_completed"
        ) as mark_completed:
            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=100,
                fairinf_n=2,
            )

            req = _mk_req("user_19", "rid_running", 4)
            simulator.process_new_request(req)

            now["t"] = 11.0
            simulator.finished_prefill(SimpleNamespace(reqs=[req]))

            req.output_ids = [42, 43]
            now["t"] = 12.0
            simulator.finished_decode(SimpleNamespace(reqs=[req]), decode_rounds=2)

            now["t"] = 13.0
            simulator.mark_request_finished(req)

            mark_completed.assert_called_once_with(
                req.rid,
                req.uid,
                timestamp_iso="1970-01-01T00:00:18.000+00:00",
            )

    def test_mark_request_finished_backfills_missing_isolated_decode_completion(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"
        ) as mark_decode, patch.object(
            sim_mod.TIMELINE_WRITER, "mark_isolated_completed"
        ) as mark_completed:
            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=100,
                fairinf_n=2,
            )

            req = _mk_req("user_19", "rid_running", 4)
            simulator.process_new_request(req)

            now["t"] = 11.0
            simulator.finished_prefill(SimpleNamespace(reqs=[req]))

            req.output_ids = [42, 43]
            now["t"] = 13.0
            simulator.mark_request_finished(req)

            self.assertEqual(
                [call.kwargs["completion_number"] for call in mark_decode.call_args_list],
                [1, 2],
            )
            mark_completed.assert_called_once_with(
                req.rid,
                req.uid,
                timestamp_iso="1970-01-01T00:00:18.000+00:00",
            )

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
                history_decode_events = [
                    e for e in tracked.timeline.history if isinstance(e, RequestDecodeEvent)
                ]
                self.assertEqual(
                    [e.completion_number for e in history_decode_events],
                    list(range(1, 11)),
                )
                self.assertEqual(
                    [e.end_timestamp for e in history_decode_events],
                    [15.0 + 3.0 * i for i in range(10)],
                )
                self.assertIsInstance(tracked.timeline.next_anticipated_event, RequestDecodeEvent)
                self.assertEqual(tracked.timeline.next_anticipated_event.completion_number, 11)
                decode_candidates = [
                    candidate for candidate in candidates if candidate.event_type == "decode"
                ]
                self.assertEqual(len(decode_candidates), 1)
                self.assertEqual(decode_candidates[0].event.completion_number, 11)
                self.assertEqual(decode_candidates[0].deadline, 45.0)

    def test_finished_prefill_after_long_queue_delay_should_not_leave_first_decode_in_the_past(self):
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

                req = _mk_req("user_19", "rid_delayed", 4)
                simulator.process_new_request(req)

                now["t"] = 20.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req]))

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
                self.assertEqual(decode_candidates[0].event.completion_number, 1)
                self.assertGreaterEqual(
                    decode_candidates[0].event.end_timestamp,
                    now["t"],
                )

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

                req1_anticipated = simulator.requests[req1.rid].timeline.next_anticipated_event
                req2_anticipated = simulator.requests[req2.rid].timeline.next_anticipated_event
                self.assertIsInstance(req1_anticipated, RequestPrefillEvent)
                self.assertIsInstance(req2_anticipated, RequestPrefillEvent)
                self.assertEqual(req1_anticipated.end_timestamp, 12.0)
                self.assertEqual(req2_anticipated.end_timestamp, 14.0)

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
                history_decode_events = [
                    e for e in tracked.timeline.history if isinstance(e, RequestDecodeEvent)
                ]
                self.assertEqual([e.completion_number for e in history_decode_events], [1])
                self.assertEqual([e.end_timestamp for e in history_decode_events], [15.0])
                self.assertIsInstance(tracked.timeline.next_anticipated_event, RequestDecodeEvent)
                self.assertEqual(tracked.timeline.next_anticipated_event.completion_number, 2)
                self.assertEqual(tracked.timeline.next_anticipated_event.end_timestamp, 18.0)

    def test_retracted_bad_request_becomes_waiting_prefill_candidate_again(self):
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
                    max_kv_tokens_per_user=8,
                    fairinf_n=2,
                )

                req1 = _mk_req("user_19", "rid_bad_1", 4)
                req2 = _mk_req("user_19", "rid_bad_2", 4)

                simulator.process_new_request(req1)
                simulator.process_new_request(req2)

                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req1, req2]))

                req1.output_ids = [1]
                req2.output_ids = [1]
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req1, req2]))

                now["t"] = 100.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req1]),
                    [req2],
                )

                candidates, waiting_deadlines = simulator.build_deadline_candidates(
                    [req2],
                    SimpleNamespace(reqs=[req1]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                decode_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.event_type == "decode"
                ]
                prefill_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.event_type == "prefill"
                ]

                self.assertEqual(
                    [candidate.req.rid for candidate in decode_candidates],
                    [req1.rid],
                )
                self.assertEqual(
                    [candidate.req.rid for candidate in prefill_candidates],
                    [req2.rid],
                )
                self.assertIn(req2.rid, waiting_deadlines)

    def test_waiting_request_deadline_is_delayed_by_existing_running_decode_queue(self):
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

                req1 = _mk_req("user_19", "rid_running", 4)
                req2 = _mk_req("user_19", "rid_waiting", 4)

                simulator.process_new_request(req1)
                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req1]))

                req1.output_ids = [1, 2, 3, 4, 5]
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req1]), decode_rounds=5)

                now["t"] = 13.0
                simulator.process_new_request(req2)

                now["t"] = 100.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req1]),
                    [req2],
                )

                candidates, waiting_deadlines, _ = simulator.build_deadline_candidates(
                    [req2],
                    SimpleNamespace(reqs=[req1]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                    include_ordered_waiting_queue=True,
                )

                # In the new isolation model, all requests (including running ones)
                # start from their arrival time in the isolation queue. req2 arrives at t=13,
                # waits for req1 to be prefilled (t=12), then gets prefilled at t=13 (fits
                # alongside req1). Its anticipated prefill ends at t=15 (2s prefill duration).
                # start_deadline = 15 - 2 = 13.
                self.assertEqual(waiting_deadlines[req2.rid], 13.0)

                prefill_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.event_type == "prefill"
                ]
                decode_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.event_type == "decode"
                ]

                # req2 anticipated prefill ends at t=15 (arrival t=13 + prefill=2s).
                self.assertEqual(
                    [(candidate.req.rid, candidate.event.end_timestamp) for candidate in prefill_candidates],
                    [(req2.rid, 15.0)],
                )
                # req1 runs alongside req2 from t=15 onwards. After 6 isolated decode steps
                # (sim_dc=6 > real_dc=5), the anticipated decode event fires.
                # Decode step durations=3s each: t=12 (after prefill) + arrivals/etc → ~33.
                self.assertEqual(len(decode_candidates), 1)
                self.assertEqual(decode_candidates[0].req.rid, req1.rid)
                self.assertEqual(decode_candidates[0].event.completion_number, 6)

    def test_retraction_penalty_pushes_running_bad_decode_deadline_out(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time):
            with patch.object(
                sim_mod, "isolated_prefill_time_estimation", return_value=2.0
            ), patch.object(
                sim_mod, "isolated_decode_time_estimation", return_value=3.0
            ):
                req1 = _mk_req("user_19", "rid_bad_1", 4)
                req2 = _mk_req("user_19", "rid_bad_2", 4)

                unconstrained = AlternateHistorySimulator(
                    max_kv_tokens_per_user=100,
                    fairinf_n=2,
                )
                constrained = AlternateHistorySimulator(
                    max_kv_tokens_per_user=8,
                    fairinf_n=2,
                )

                for simulator in (unconstrained, constrained):
                    simulator.process_new_request(req1)
                    simulator.process_new_request(req2)
                    now["t"] = 11.0
                    simulator.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
                    req1.output_ids = [1]
                    req2.output_ids = [1]
                    now["t"] = 12.0
                    simulator.finished_decode(SimpleNamespace(reqs=[req1, req2]))

                now["t"] = 100.0
                unconstrained.start_of_pass(
                    SimpleNamespace(reqs=[req1, req2]),
                    [],
                )
                constrained.start_of_pass(
                    SimpleNamespace(reqs=[req1]),
                    [req2],
                )

                unc_candidates, _ = unconstrained.build_deadline_candidates(
                    [],
                    SimpleNamespace(reqs=[req1, req2]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )
                con_candidates, _ = constrained.build_deadline_candidates(
                    [req2],
                    SimpleNamespace(reqs=[req1]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                unc_r1_decode = next(
                    candidate
                    for candidate in unc_candidates
                    if candidate.req.rid == req1.rid
                    and candidate.event_type == "decode"
                )
                con_r1_decode = next(
                    candidate
                    for candidate in con_candidates
                    if candidate.req.rid == req1.rid
                    and candidate.event_type == "decode"
                )

                self.assertEqual(unc_r1_decode.deadline, 18.0)
                self.assertGreaterEqual(
                    con_r1_decode.deadline,
                    unc_r1_decode.deadline + sim_mod.RETRACTION_PENALTY_SECONDS,
                )

    def test_retraction_resets_request_to_waiting_prefill_from_new_time(self):
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

                req = _mk_req("user_19", "rid_retracted", 4)
                simulator.process_new_request(req)

                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req]))
                req.output_ids = [1]
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req]))

                # Retraction should reset the live simulator view back to a new waiting start.
                now["t"] = 30.0
                simulator.process_new_request(req)

                simulator.start_of_pass(
                    SimpleNamespace(reqs=[]),
                    [req],
                )
                candidates, waiting_deadlines = simulator.build_deadline_candidates(
                    [req],
                    SimpleNamespace(reqs=[]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )

                self.assertEqual(
                    [(candidate.req.rid, candidate.event_type) for candidate in candidates],
                    [(req.rid, "prefill")],
                )
                self.assertEqual(waiting_deadlines[req.rid], 30.0)

    def test_retracted_bad_request_earliest_decode_deadline_advances_across_passes(self):
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

                req_bad_early = _mk_req("user_19", "rid_bad_early", 4)
                req_bad_blocker = _mk_req("user_19", "rid_bad_blocker", 4)

                simulator.process_new_request(req_bad_early)
                simulator.process_new_request(req_bad_blocker)

                now["t"] = 11.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req_bad_early, req_bad_blocker]))

                req_bad_early.output_ids = [1]
                req_bad_blocker.output_ids = [1]
                now["t"] = 12.0
                simulator.finished_decode(SimpleNamespace(reqs=[req_bad_early, req_bad_blocker]))

                # The blocker gets retracted back to waiting, then later re-prefilled and decoded again.
                now["t"] = 30.0
                simulator.process_new_request(req_bad_blocker)
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req_bad_early]),
                    [req_bad_blocker],
                )
                candidates, _ = simulator.build_deadline_candidates(
                    [req_bad_blocker],
                    SimpleNamespace(reqs=[req_bad_early]),
                    req_is_fair_prefill=lambda req, rb: True,
                    req_is_fair_decode=lambda req, rb: True,
                    event_delta_seconds=lambda tracked, event: 0.0,
                    pooled_prefill_estimate_seconds=lambda req: 2.0,
                    pooled_decode_estimate_seconds=lambda req, rb: 3.0,
                )
                self.assertEqual(
                    [(candidate.req.rid, candidate.event_type) for candidate in candidates],
                    [(req_bad_early.rid, "decode"), (req_bad_blocker.rid, "prefill")],
                )

                now["t"] = 31.0
                simulator.finished_prefill(SimpleNamespace(reqs=[req_bad_blocker]))
                req_bad_blocker.output_ids = [1, 2]
                now["t"] = 32.0
                simulator.finished_decode(SimpleNamespace(reqs=[req_bad_blocker]))

                now["t"] = 40.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req_bad_early, req_bad_blocker]),
                    [],
                )

                # build_deadline_candidates only returns the single earliest decode candidate
                # (req_bad_early wins here). Read the blocker's anticipated event directly.
                blocker_tracked_1 = simulator.requests[req_bad_blocker.rid]
                blocker_ant_1 = blocker_tracked_1.timeline.next_anticipated_event

                req_bad_blocker.output_ids = [1, 2, 3]
                now["t"] = 41.0
                simulator.finished_decode(SimpleNamespace(reqs=[req_bad_blocker]))

                now["t"] = 50.0
                simulator.start_of_pass(
                    SimpleNamespace(reqs=[req_bad_early, req_bad_blocker]),
                    [],
                )

                blocker_tracked_2 = simulator.requests[req_bad_blocker.rid]
                blocker_ant_2 = blocker_tracked_2.timeline.next_anticipated_event

                self.assertIsInstance(blocker_ant_1, RequestDecodeEvent)
                self.assertIsInstance(blocker_ant_2, RequestDecodeEvent)
                self.assertEqual(blocker_ant_1.completion_number, 3)
                self.assertEqual(blocker_ant_2.completion_number, 4)
                self.assertGreater(blocker_ant_2.end_timestamp, blocker_ant_1.end_timestamp)


    def test_isolation_retraction_evicts_fewest_real_completions(self):
        """Retraction evicts the request with the fewest real completions (real decode_count).

        req1 (2 tokens) has real_decode_count=5; req2 (4 tokens) has real_decode_count=1.
        Both arrive together. When both are active in isolation and the KV budget is exceeded,
        req2 (fewer real completions) is evicted — not req1.

        Budget=11: req1(kv=3) + req2(kv=5) = 8. But 8+2 active = 10 ≤ 11, no retraction.
        After one decode: req1(kv=4) + req2(kv=6) = 10. 10+2=12 > 11 → retract.
        Evict min real_dc: req2(1) < req1(5) → req2 evicted → PrefillEvent.
        req1 (more real completions) survives → DecodeEvent.
        """

        with patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=1.0), \
             patch.object(sim_mod, "isolated_decode_time_estimation", return_value=1.0):

            req1 = _mk_req("user_C", "rid_C1", 2)
            req2 = _mk_req("user_C", "rid_C2", 4)

            t1 = TrackedRequest(
                req=req1, arrival_timestamp=0.0,
                timeline=RequestTimeline(history=[
                    RequestStartEvent(req_id=req1.rid, end_timestamp=0.0),
                    RequestDecodeEvent(req_id=req1.rid, end_timestamp=2.0, completion_number=5),
                ]),
            )
            t2 = TrackedRequest(
                req=req2, arrival_timestamp=0.0,
                timeline=RequestTimeline(history=[
                    RequestStartEvent(req_id=req2.rid, end_timestamp=0.0),
                    RequestDecodeEvent(req_id=req2.rid, end_timestamp=2.0, completion_number=1),
                ]),
            )
            ut = UserTimeline(uid="user_C", max_kv_tokens=11, fairinf_n=1)
            ut.request_timelines[req1.rid] = t1
            ut.request_timelines[req2.rid] = t2
            ut.requests_real[req1.rid] = RequestStatusReal(rid=req1.rid, prefill_done=True, decode_count=5)
            ut.requests_real[req2.rid] = RequestStatusReal(rid=req2.rid, prefill_done=True, decode_count=1)

            ut.rebuild_from_real_state()

            req1_anticipated = ut.request_timelines[req1.rid].timeline.next_anticipated_event
            req2_anticipated = ut.request_timelines[req2.rid].timeline.next_anticipated_event

            anticipated_types = {
                req1.rid: type(req1_anticipated).__name__,
                req2.rid: type(req2_anticipated).__name__,
            }
            prefill_rids = [rid for rid, t in anticipated_types.items() if t == "RequestPrefillEvent"]
            decode_rids = [rid for rid, t in anticipated_types.items() if t == "RequestDecodeEvent"]

            self.assertEqual(len(prefill_rids), 1,
                             f"Exactly one request should be retracted to prefill; got {anticipated_types}")
            self.assertEqual(len(decode_rids), 1,
                             f"Exactly one request should continue decoding; got {anticipated_types}")
            self.assertIn(req2.rid, prefill_rids,
                          "req2 (fewest real completions, real_dc=1) should be evicted")
            self.assertIn(req1.rid, decode_rids,
                          "req1 (most real completions, real_dc=5) should survive")

    def test_isolation_retraction_resets_request_and_regains_prefill_slot(self):
        """With max_kv_tokens=11, both requests fit together initially, but retraction fires
        once both are running and KV is exhausted.

        req1 (2 tokens) arrives at t=0 with real_decode_count=3; req2 (4 tokens) at t=3
        with real_decode_count=1. In isolation:
          - req1 prefills at t=1 and runs alone until req2 arrives.
          - At t=3 req2 is prefilled alongside req1.
          - After one combined decode step, KV budget is exceeded.
          - Evict request with fewest real completions: req2 (real_dc=1) < req1 (real_dc=3).
          - req1 survives → RequestDecodeEvent (once sim_dc > real_dc=3).
          - req2 (evicted) → RequestPrefillEvent.
        """

        with patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=1.0), \
             patch.object(sim_mod, "isolated_decode_time_estimation", return_value=1.0):

            req1 = _mk_req("user_B", "rid_B1", 2)
            req2 = _mk_req("user_B", "rid_B2", 4)

            # req1 arrives at t=0 (real_dc=3), req2 at t=3 (real_dc=1).
            t1 = TrackedRequest(
                req=req1, arrival_timestamp=0.0,
                timeline=RequestTimeline(history=[
                    RequestStartEvent(req_id=req1.rid, end_timestamp=0.0),
                    RequestDecodeEvent(req_id=req1.rid, end_timestamp=4.0, completion_number=3),
                ]),
            )
            t2 = TrackedRequest(
                req=req2, arrival_timestamp=3.0,
                timeline=RequestTimeline(history=[
                    RequestStartEvent(req_id=req2.rid, end_timestamp=3.0),
                    RequestDecodeEvent(req_id=req2.rid, end_timestamp=5.0, completion_number=1),
                ]),
            )
            ut = UserTimeline(uid="user_B", max_kv_tokens=11, fairinf_n=1)
            ut.request_timelines[req1.rid] = t1
            ut.request_timelines[req2.rid] = t2
            ut.requests_real[req1.rid] = RequestStatusReal(rid=req1.rid, prefill_done=True, decode_count=3)
            ut.requests_real[req2.rid] = RequestStatusReal(rid=req2.rid, prefill_done=True, decode_count=1)

            ut.rebuild_from_real_state()

            req1_anticipated = ut.request_timelines[req1.rid].timeline.next_anticipated_event
            req2_anticipated = ut.request_timelines[req2.rid].timeline.next_anticipated_event

            anticipated_types = {
                req1.rid: type(req1_anticipated).__name__,
                req2.rid: type(req2_anticipated).__name__,
            }
            prefill_rids = [rid for rid, t in anticipated_types.items() if t == "RequestPrefillEvent"]
            decode_rids = [rid for rid, t in anticipated_types.items() if t == "RequestDecodeEvent"]

            # req2 (fewest real completions, real_dc=1) should be retracted → PrefillEvent.
            # req1 (more real completions, real_dc=3) should continue → DecodeEvent.
            self.assertEqual(len(prefill_rids), 1,
                             f"Exactly one request should be retracted to prefill; got {anticipated_types}")
            self.assertEqual(len(decode_rids), 1,
                             f"Exactly one request should continue decoding; got {anticipated_types}")
            self.assertIn(req2.rid, prefill_rids,
                          "req2 (fewest real completions, real_dc=1) should be evicted")
            self.assertIn(req1.rid, decode_rids,
                          "req1 (more real completions, real_dc=3) should survive")

    def test_isolation_queuing_finite_prefill_time_behind_running_request(self):
        """req2 arrives at t=1 while req1 is running. In isolation req1 is already decoded
        once (kv=6) when req2 arrives. Budget=12 fits both (6+4+2 active = 12 ≤ 12).

        After one combined decode retraction fires (11+2=13 > 12). req2 (real_dc=0) is evicted.
        req1 then records its DecodeEvent (sim_dc=2 > real_dc=1). On the next step req2
        re-prefills (req1_kv=7, 7+4=11 ≤ 12) and gets a finite PrefillEvent. At that point
        all rids have anticipated events, so the loop exits and req2.end_timestamp is finite."""
        now = {"t": 0.0}

        def fake_time():
            return now["t"]

        with patch.object(sim_mod.time, "time", side_effect=fake_time), \
             patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=1.0), \
             patch.object(sim_mod, "isolated_decode_time_estimation", return_value=1.0):

            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=12,
                fairinf_n=1,
            )

            req1 = _mk_req("user_C", "rid_C1", 4)
            req2 = _mk_req("user_C", "rid_C2", 4)

            # req1 arrives at t=0
            simulator.process_new_request(req1, arrival_timestamp=0.0)

            # req1 finishes prefill at t=1
            now["t"] = 1.0
            simulator.finished_prefill(SimpleNamespace(reqs=[req1]))

            # req1 finishes 1 decode at t=2 — real KV = 4+1=5
            req1.output_ids = [1]
            now["t"] = 2.0
            simulator.finished_decode(SimpleNamespace(reqs=[req1]))

            # req2 arrives at t=1 (concurrently with req1's prefill completion)
            simulator.process_new_request(req2, arrival_timestamp=1.0)

            # At t=100: req1 is running, req2 is waiting
            now["t"] = 100.0
            simulator.start_of_pass(
                SimpleNamespace(reqs=[req1]),
                [req2],
            )

            ut = simulator.users["user_C"]
            ut.rebuild_from_real_state()

            req2_anticipated = ut.request_timelines[req2.rid].timeline.next_anticipated_event
            self.assertIsNotNone(req2_anticipated)
            self.assertIsInstance(req2_anticipated, RequestPrefillEvent)
            self.assertNotEqual(req2_anticipated.end_timestamp, float("inf"),
                                "req2 should have a finite anticipated prefill time (queued behind req1), not inf")
            self.assertGreater(req2_anticipated.end_timestamp, 0.0,
                               "req2 anticipated prefill should be after time 0")


if __name__ == "__main__":
    unittest.main()
