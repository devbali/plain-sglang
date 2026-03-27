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
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
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
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))
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
            self.assertEqual(decode_candidates[0].event.completion_number, 3)
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


if __name__ == "__main__":
    unittest.main()
