import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.doc_policy as doc_policy_mod
import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


def _mk_req(uid: str, rid: str, prompt_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * prompt_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    req.sampling_params = SamplingParams(max_new_tokens=10, min_new_tokens=0)
    return req


class TestDocPolicyCurrentPassUnit(unittest.TestCase):
    def test_pending_new_request_merge_adds_waiting_deadline(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_1", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [])

            now["t"] = 101.0
            policy.process_new_request(waiting_req)
            policy._ensure_current_pass_state(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            self.assertIn(
                waiting_req.rid,
                policy._waiting_prefill_start_deadline_by_rid,
            )

    def test_prepare_built_decode_state_is_consumed_before_merging_new_waiting(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_20", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 12.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )
            running_req.output_ids = [42]
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [])
            self.assertEqual(policy._deadline_queue[0].event.completion_number, 2)

            now["t"] = 100.5
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=True,
            )
            self.assertFalse(policy._simulator_rebuild_prepared)
            self.assertIsNone(policy._prepared_pass_state)

            running_req.output_ids = [42, 43]
            now["t"] = 101.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )
            policy.process_new_request(waiting_req)

            policy._ensure_current_pass_state(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == running_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertEqual(decode_candidates[0].event.completion_number, 4)
            self.assertEqual(policy._last_pass_state_source, "ensure_merge_pending")

    def test_decode_prepare_should_not_need_full_rebuild_on_simple_steady_state(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_19", "rid_running", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[req]))

            running_batch = SimpleNamespace(reqs=[req])
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [])

            req.output_ids = [42, 43]
            now["t"] = 100.5
            with patch.object(
                policy.simulator,
                "rebuild_all_tracked_requests",
                side_effect=AssertionError(
                    "decode prepare unexpectedly invoked full simulator rebuild"
                ),
            ):
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[],
                    event_type="decode",
                    prepare_pass_state=True,
                )

    def test_decode_prepare_with_waiting_should_not_invoke_full_rebuild(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_20", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[req]))

            running_batch = SimpleNamespace(reqs=[req])
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [waiting_req])

            req.output_ids = [42, 43]
            now["t"] = 100.5
            with patch.object(
                policy.simulator,
                "rebuild_all_tracked_requests",
                side_effect=AssertionError(
                    "decode prepare with waiting unexpectedly invoked full simulator rebuild"
                ),
            ):
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[waiting_req],
                    event_type="decode",
                    prepare_pass_state=True,
                )

    def test_prefill_prepare_should_not_invoke_full_rebuild(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_1", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_batch = SimpleNamespace(reqs=[running_req])
            scheduled_batch = SimpleNamespace(reqs=[waiting_req])

            now["t"] = 100.0
            with patch.object(
                policy.simulator,
                "rebuild_all_tracked_requests",
                side_effect=AssertionError(
                    "prefill prepare unexpectedly invoked full simulator rebuild"
                ),
            ):
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[waiting_req],
                    scheduled_batch=scheduled_batch,
                    event_type="prefill",
                    prepare_pass_state=True,
                )

    def test_start_of_pass_should_not_invoke_full_rebuild(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_1", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            with patch.object(
                policy.simulator,
                "start_of_pass",
                side_effect=AssertionError(
                    "start_of_pass unexpectedly invoked full simulator rebuild"
                ),
            ):
                policy.start_of_pass(running_batch, [waiting_req])

    def test_decode_prepare_advances_and_reseeds_next_decode_deadline(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_19", "rid_running", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req]))
            running_batch = SimpleNamespace(reqs=[req])

            now["t"] = 12.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )
            tracked = policy.simulator.requests[req.rid]
            anticipated = tracked.alternate_history_timeline.anticipated_future_events
            self.assertEqual(len(anticipated), 1)
            self.assertEqual(anticipated[0].completion_number, 2)
            first_next_deadline = anticipated[0].end_timestamp

            req.output_ids = [42]
            now["t"] = 13.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )
            anticipated = tracked.alternate_history_timeline.anticipated_future_events
            self.assertEqual(len(anticipated), 1)
            self.assertEqual(anticipated[0].completion_number, 3)
            self.assertGreater(anticipated[0].end_timestamp, first_next_deadline)

    def test_long_decode_stall_should_not_create_large_positive_prefill_slack(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_1", "rid_running", 4)
            waiting_req = _mk_req("user_2", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [waiting_req])

            # Simulate an unexpectedly long real decode stall without any corrective
            # isolated progression updates; the overdue decode should stay overdue.
            now["t"] = 400.0
            policy.process_new_request(waiting_req)
            policy._ensure_current_pass_state(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )
            force_prefill = policy.fairinf_force_prefill_any_waiting(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            self.assertFalse(force_prefill)
            self.assertTrue(policy._has_decode_deadline)
            self.assertLessEqual(policy._max_safe_prefill_tokens or 0, 0)
            self.assertIsNotNone(policy._earliest_decode_start_deadline)
            self.assertLess(policy._earliest_decode_start_deadline - now["t"], 0.0)

    def test_finished_prefill_should_add_decode_frontier_for_newly_running_request(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_1", "rid_running", 4)
            prefilling_req = _mk_req("user_2", "rid_prefilling", 4)
            waiting_req = _mk_req("user_3", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            initial_running = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            policy.start_of_pass(initial_running, [prefilling_req])

            scheduled_batch = SimpleNamespace(reqs=[prefilling_req])
            policy.note_scheduled_prefill_batch(scheduled_batch)
            policy._ensure_current_pass_state(
                [prefilling_req],
                running_batch=initial_running,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            now["t"] = 101.0
            policy.finished_prefill(scheduled_batch)
            policy.process_new_request(waiting_req)

            next_running = SimpleNamespace(reqs=[running_req, prefilling_req])
            policy._ensure_current_pass_state(
                [waiting_req],
                running_batch=next_running,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_rids = {
                candidate.req.rid
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode"
            }
            self.assertIn(prefilling_req.rid, decode_rids)

    def test_prefill_prepare_then_finished_prefill_keeps_waiting_deadlines_subset(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=0.035579429910760005
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=0.0139532681806768
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=0.03
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=0.04
        ):
            policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)

            running = [_mk_req("19", f"run{i}", 101) for i in range(5)]
            for req in running:
                policy.process_new_request(req)

            now["t"] = 11.0
            running_batch = SimpleNamespace(reqs=running)
            policy.finished_prefill(running_batch)
            for req in running:
                req.output_ids = [0] * 100

            now["t"] = 12.0
            policy.finished_decode(running_batch)

            waiting = [_mk_req("1", f"wait{i}", 101) for i in range(20)]
            for req in waiting:
                policy.process_new_request(req)

            now["t"] = 100.0
            policy.start_of_pass(running_batch, waiting)

            scheduled_batch = SimpleNamespace(reqs=waiting[:2])
            policy.note_scheduled_prefill_batch(scheduled_batch)
            policy.prepare_during_gpu_execution(
                event_type="prefill",
                running_batch=running_batch,
                waiting_queue=waiting,
                scheduled_batch=scheduled_batch,
                prepare_pass_state=True,
            )

            now["t"] = 100.03
            policy.finished_prefill(scheduled_batch)
            next_running = SimpleNamespace(reqs=running + waiting[:2])
            remaining_waiting = waiting[2:]
            policy._ensure_current_pass_state(
                remaining_waiting,
                running_batch=next_running,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            remaining_rids = {req.rid for req in remaining_waiting}
            waiting_deadline_rids = set(policy._waiting_prefill_start_deadline_by_rid)
            self.assertLessEqual(
                len(waiting_deadline_rids),
                len(remaining_waiting),
            )
            self.assertTrue(waiting_deadline_rids.issubset(remaining_rids))

    def test_decode_prepare_with_waiting_advances_running_deadline_across_passes(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            waiting_req = _mk_req("user_1", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            waiting_queue = [waiting_req]
            now["t"] = 100.0
            policy.start_of_pass(running_batch, waiting_queue)

            def running_decode_candidate():
                return next(
                    candidate
                    for candidate in policy._deadline_queue
                    if candidate.event_type == "decode"
                    and candidate.req.rid == running_req.rid
                )

            first = running_decode_candidate()
            self.assertGreaterEqual(first.event.completion_number, 2)

            running_req.output_ids = [42, 43]
            now["t"] = 101.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=waiting_queue,
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=1,
            )
            policy._ensure_current_pass_state(
                waiting_queue,
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )
            second = running_decode_candidate()

            running_req.output_ids = [42, 43, 44]
            now["t"] = 102.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=waiting_queue,
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=1,
            )
            policy._ensure_current_pass_state(
                waiting_queue,
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )
            third = running_decode_candidate()

            self.assertGreater(second.event.completion_number, first.event.completion_number)
            self.assertGreater(third.event.completion_number, second.event.completion_number)
            self.assertGreater(second.deadline, first.deadline)
            self.assertGreater(third.deadline, second.deadline)

    def test_retracted_bad_request_in_waiting_does_not_remain_decode_blocker(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            good_running = _mk_req("user_1", "rid_good_running", 4)
            bad_retracted = _mk_req("user_19", "rid_bad_retracted", 4)
            good_waiting = _mk_req("user_1", "rid_good_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            for req in (good_running, bad_retracted):
                policy.process_new_request(req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[good_running, bad_retracted]))
            good_running.output_ids = [1]
            bad_retracted.output_ids = [1]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[good_running, bad_retracted]))

            now["t"] = 30.0
            policy.note_retracted_reqs([bad_retracted])
            policy.process_new_request(good_waiting)

            running_batch = SimpleNamespace(reqs=[good_running])
            waiting_queue = [bad_retracted, good_waiting]
            now["t"] = 100.0
            policy.start_of_pass(running_batch, waiting_queue)
            policy._ensure_current_pass_state(
                waiting_queue,
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_rids = {
                candidate.req.rid
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode"
            }
            waiting_rids = set(policy._waiting_prefill_start_deadline_by_rid)

            self.assertNotIn(bad_retracted.rid, decode_rids)
            self.assertIn(bad_retracted.rid, waiting_rids)
            self.assertIn(good_waiting.rid, waiting_rids)

    def test_pending_merge_drops_decode_candidate_for_request_not_in_live_sets(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            bad_running = _mk_req("user_19", "rid_bad_running", 4)
            good_waiting = _mk_req("user_1", "rid_good_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(bad_running)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[bad_running]))
            bad_running.output_ids = [1]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[bad_running]))

            running_batch = SimpleNamespace(reqs=[bad_running])
            now["t"] = 20.0
            policy.start_of_pass(running_batch, [])

            decode_rids = {
                candidate.req.rid
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode"
            }
            self.assertIn(bad_running.rid, decode_rids)

            now["t"] = 21.0
            policy.process_new_request(good_waiting)
            policy._merge_pending_pass_state_mutations(
                [good_waiting],
                running_batch=None,
            )

            decode_rids = {
                candidate.req.rid
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode"
            }
            waiting_rids = set(policy._waiting_prefill_start_deadline_by_rid)

            self.assertNotIn(bad_running.rid, decode_rids)
            self.assertIn(good_waiting.rid, waiting_rids)

    def test_memory_pressure_filters_unfair_waiting_users_from_prefill_deadlines(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", return_value=2.0
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=3.0
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", return_value=2.0
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            fair_waiting = _mk_req("user_1", "rid_waiting_fair", 4)
            unfair_waiting = _mk_req("user_20", "rid_waiting_unfair", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))
            policy.process_new_request(fair_waiting)
            policy.process_new_request(unfair_waiting)

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            with patch.object(policy, "_memory_pressure_active_for_prefill", return_value=True, create=True), patch.object(
                policy,
                "req_is_fair_prefill",
                side_effect=lambda req, **kwargs: req.uid == "user_1",
            ):
                policy._build_pass_state(
                    [fair_waiting, unfair_waiting],
                    running_batch,
                    policy._deltas_us,
                )

            self.assertIn(fair_waiting.rid, policy._waiting_prefill_start_deadline_by_rid)
            self.assertNotIn(unfair_waiting.rid, policy._waiting_prefill_start_deadline_by_rid)

    def test_safe_prefix_allows_unfair_before_pressure_then_fair_only_after(self):
        now = {"t": 100.0}

        def fake_time():
            return now["t"]

        def fake_pooled_prefill(total_prompt_tokens, max_prompt_tokens, batch_size, fairinf_n):
            del total_prompt_tokens, max_prompt_tokens, fairinf_n
            return 0.01 * batch_size

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", side_effect=fake_pooled_prefill
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", return_value=0.01
        ):
            bad_before_pressure = _mk_req("user_19", "rid_bad_1", 101)
            bad_after_pressure = _mk_req("user_20", "rid_bad_2", 101)
            good_after_pressure_1 = _mk_req("user_1", "rid_good_1", 101)
            good_after_pressure_2 = _mk_req("user_2", "rid_good_2", 101)
            running_req = _mk_req("user_30", "rid_running", 16)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            decode_candidate = SimpleNamespace(
                event_type="decode",
                start_deadline=100.035,
                req=running_req,
                event=SimpleNamespace(completion_number=1),
            )
            waiting_deadlines = {
                bad_before_pressure.rid: 100.0,
                bad_after_pressure.rid: 100.0,
                good_after_pressure_1.rid: 100.0,
                good_after_pressure_2.rid: 100.0,
            }

            with patch.object(
                policy.simulator,
                "build_deadline_candidates",
                return_value=([decode_candidate], waiting_deadlines),
            ), patch.object(
                policy,
                "_no_retraction_prefill_token_cap",
                return_value=101,
                create=True,
            ), patch.object(
                policy,
                "req_is_fair_prefill",
                side_effect=lambda req, **kwargs: req.uid in {"user_1", "user_2"},
            ), patch.object(
                policy,
                "_force_prefill_within_user_headroom",
                return_value=True,
            ):
                policy._build_pass_state(
                    [
                        bad_before_pressure,
                        bad_after_pressure,
                        good_after_pressure_1,
                        good_after_pressure_2,
                    ],
                    SimpleNamespace(reqs=[running_req]),
                    policy._deltas_us,
                )

            self.assertEqual(
                [req.rid for req in policy._forced_prefill_queue],
                [
                    bad_before_pressure.rid,
                    good_after_pressure_1.rid,
                    good_after_pressure_2.rid,
                ],
            )
            self.assertEqual(policy._max_safe_prefill_tokens, 303)

if __name__ == "__main__":
    unittest.main()
