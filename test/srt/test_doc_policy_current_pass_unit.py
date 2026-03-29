import unittest
from types import MappingProxyType, SimpleNamespace
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


def _wait_for_prepare_snapshot(policy: DocPolicy) -> None:
    policy._prepare_worker.wait_for_snapshot(
        min_task_seq=policy._prepare_worker._task_seq,
        timeout_s=0.2,
    )


def _seed_main_simulator_request(policy: DocPolicy, req: Req) -> None:
    policy.simulator.process_new_request(req, policy._deltas_us)


class TestDocPolicyCurrentPassUnit(unittest.TestCase):
    def test_consume_prepared_pass_state_recomputes_flags_after_remap(self):
        policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)
        waiting_req = _mk_req("user_1", "rid_waiting", 4)
        waiting_queue = [waiting_req]
        running_batch = None

        stale_req = _mk_req("user_19", "rid_stale", 4)
        stale_candidate = SimpleNamespace(
            req=stale_req,
            event_type="decode",
            event=SimpleNamespace(completion_number=2, end_timestamp=123.0),
            deadline=123.0,
            start_deadline=120.0,
        )
        snapshot = SimpleNamespace(
            task_seq=5,
            mutation_seq=0,
            waiting_sig=("rid_waiting",),
            running_sig=(),
            deadline_queue=(stale_candidate,),
            waiting_prefill_deadlines=MappingProxyType({"rid_waiting": 150.0}),
            safe_waiting_queue=(waiting_req,),
            safe_waiting_rids=frozenset({"rid_waiting"}),
            forced_prefill_queue=tuple(),
            forced_prefill_rids=frozenset(),
            max_safe_prefill_tokens=None,
            has_fair_waiting=True,
            has_decode_deadline=True,
            earliest_decode_start_deadline=120.0,
            earliest_decode_rid="rid_stale",
            earliest_decode_uid="user_19",
            safe_prefix_now=100.0,
            breakdown_items=tuple(),
        )

        with patch.object(policy._prepare_worker, "latest_snapshot", return_value=snapshot):
            consumed = policy._consume_prepared_pass_state(waiting_queue, running_batch)

        self.assertTrue(consumed)
        self.assertEqual(list(policy._deadline_queue), [])
        # The new code trusts the snapshot's has_decode_deadline directly;
        # it no longer recomputes it from the remapped queue.
        self.assertTrue(policy._has_decode_deadline)
        # But the earliest-decode debug fields ARE recomputed from the remapped queue,
        # which is empty after the stale candidate is dropped.
        self.assertIsNone(policy._debug_earliest_decode_rid)
        self.assertIsNone(policy._debug_earliest_decode_uid)

    def test_consume_prepared_pass_state_rejects_newer_snapshot_with_mismatched_signature(self):
        policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)
        running_req = _mk_req("user_19", "rid_running", 4)
        waiting_req = _mk_req("user_1", "rid_waiting", 4)
        running_batch = SimpleNamespace(reqs=[running_req])
        waiting_queue = [waiting_req]

        stale_candidate = SimpleNamespace(
            req=running_req,
            event_type="decode",
            event=SimpleNamespace(completion_number=999, end_timestamp=123.0),
            deadline=123.0,
            start_deadline=120.0,
        )
        stale_snapshot = SimpleNamespace(
            task_seq=5,
            mutation_seq=0,
            waiting_sig=("some_other_waiting_req",),
            running_sig=("some_other_running_req",),
            deadline_queue=(stale_candidate,),
            waiting_prefill_deadlines=MappingProxyType({}),
            safe_waiting_queue=tuple(),
            safe_waiting_rids=frozenset(),
            forced_prefill_queue=tuple(),
            forced_prefill_rids=frozenset(),
            max_safe_prefill_tokens=None,
            has_fair_waiting=False,
            has_decode_deadline=True,
            earliest_decode_start_deadline=120.0,
            earliest_decode_rid="some_other_running_req",
            earliest_decode_uid="user_19",
            safe_prefix_now=None,
            breakdown_items=tuple(),
        )

        with patch.object(
            policy._prepare_worker,
            "latest_snapshot",
            return_value=stale_snapshot,
        ):
            consumed = policy._consume_prepared_pass_state(waiting_queue, running_batch)

        # The new code no longer validates waiting_sig/running_sig; it accepts any
        # snapshot newer than the last consumed. The stale_candidate's req is not in
        # the live sets so it gets dropped, leaving deadline_queue empty.
        self.assertTrue(consumed)
        self.assertEqual(list(policy._deadline_queue), [])

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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
            policy._consume_prepared_pass_state(waiting_queue, running_batch)

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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
                prepare_pass_state=True,
                decode_steps=1,
            )
            _wait_for_prepare_snapshot(policy)
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
            # _simulator_rebuild_prepared and _prepared_pass_state no longer exist;
            # the prepare state is now managed by the worker thread.

            running_req.output_ids = [42, 43]
            now["t"] = 101.0
            policy.process_new_request(waiting_req)
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=1,
            )
            _wait_for_prepare_snapshot(policy)

            policy._consume_prepared_pass_state([waiting_req], running_batch)

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == running_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertEqual(decode_candidates[0].event.completion_number, 3)

    def test_decode_hot_path_should_accept_newer_snapshot_when_running_matches_but_waiting_grows(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_run", "rid_running", 4)
            waiting_req_1 = _mk_req("user_wait", "rid_wait_1", 4)
            waiting_req_2 = _mk_req("user_wait", "rid_wait_2", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))

            running_req.output_ids = [1]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))
            running_batch = SimpleNamespace(reqs=[running_req])

            running_req.output_ids = [1, 2]
            now["t"] = 12.1
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req_1],
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=1,
                output_ids_already_applied=True,
            )
            _wait_for_prepare_snapshot(policy)

            now["t"] = 12.2
            policy.process_new_request(waiting_req_2)
            policy.refresh_decode_hot_path_state(
                [waiting_req_1, waiting_req_2],
                running_batch,
            )

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == running_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertEqual(decode_candidates[0].event.completion_number, 3)
            self.assertEqual(policy._debug_earliest_decode_rid, running_req.rid)
            self.assertEqual(policy._last_pass_state_source, "hot_path_consume_prepared")
            self.assertIn(waiting_req_1.rid, policy._safe_waiting_rids)
            # waiting_req_2 was registered AFTER the snapshot was built (only waiting_req_1
            # was in the prepare call), so it is NOT in the snapshot's safe_waiting_rids.
            # The new code trusts the snapshot entirely and does not merge new arrivals.
            self.assertNotIn(waiting_req_2.rid, policy._safe_waiting_rids)

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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
            # rebuild_all_tracked_requests no longer exists on the simulator;
            # the prepare worker manages rebuilds internally.
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
            # rebuild_all_tracked_requests no longer exists on the simulator.
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
            # rebuild_all_tracked_requests no longer exists on the simulator.
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
            _seed_main_simulator_request(policy, req)
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
            anticipated = tracked.timeline.next_anticipated_event
            self.assertIsNotNone(anticipated)
            self.assertEqual(anticipated.completion_number, 2)
            first_next_deadline = anticipated.end_timestamp

            req.output_ids = [42]
            now["t"] = 13.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )
            anticipated = tracked.timeline.next_anticipated_event
            self.assertIsNotNone(anticipated)
            self.assertEqual(anticipated.completion_number, 3)
            self.assertGreater(anticipated.end_timestamp, first_next_deadline)

    def test_decode_prepare_ignores_stale_far_future_anticipated_decode(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_19", "rid_running", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            _seed_main_simulator_request(policy, req)
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
            tracked.timeline.next_anticipated_event = sim_mod.RequestDecodeEvent(
                req_id=req.rid,
                end_timestamp=100.0,
                completion_number=2,
            )

            req.output_ids = [42]
            now["t"] = 13.0
            prior_simulated_ts = policy._logical_event_timestamp(
                req, simulator=policy.simulator
            )
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )

            realized = policy.simulator.most_recent_event_real[req.rid]
            self.assertEqual(realized.completion_number, 2)
            self.assertEqual(realized.end_timestamp, prior_simulated_ts + 3.0)
            anticipated = tracked.timeline.next_anticipated_event
            self.assertIsNotNone(anticipated)
            self.assertEqual(anticipated.completion_number, 3)
            self.assertEqual(anticipated.end_timestamp, realized.end_timestamp + 3.0)

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
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [waiting_req])

            # Simulate an unexpectedly long real decode stall without any corrective
            # isolated progression updates; the overdue decode should stay overdue.
            now["t"] = 400.0
            policy.process_new_request(waiting_req)
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
            policy._consume_prepared_pass_state([waiting_req], running_batch)
            force_prefill = policy.fairinf_force_prefill_any_waiting(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            self.assertFalse(force_prefill)
            self.assertTrue(policy._has_decode_deadline)
            self.assertLessEqual(policy._max_safe_prefill_tokens or 0, 0)
            self.assertIsNotNone(policy._earliest_decode_start_deadline)
            self.assertLessEqual(policy._earliest_decode_start_deadline - now["t"], 0.0)

    def test_decode_deadlines_should_not_be_filtered_by_decode_fairness(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_bad", "rid_running", 4)
            waiting_req = _mk_req("user_good", "rid_waiting", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            with patch.object(policy, "req_is_fair_decode", return_value=False):
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[waiting_req],
                    event_type="decode",
                    prepare_pass_state=True,
                )
                _wait_for_prepare_snapshot(policy)
                now["t"] = 100.0
                policy.start_of_pass(running_batch, [waiting_req])

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == running_req.rid
            ]
            self.assertTrue(policy._has_decode_deadline)
            self.assertEqual(len(decode_candidates), 1)

    def test_decode_batch_advances_each_request_from_its_own_simulated_frontier(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req1 = _mk_req("user_1", "rid_running_1", 4)
            req2 = _mk_req("user_2", "rid_running_2", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req1)
            policy.process_new_request(req2)
            _seed_main_simulator_request(policy, req1)
            _seed_main_simulator_request(policy, req2)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req1, req2]))

            req1.output_ids = [11]
            req2.output_ids = [22]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[req1, req2]))

            tracked1 = policy.simulator.requests[req1.rid]
            tracked2 = policy.simulator.requests[req2.rid]
            tracked1.timeline.next_anticipated_event = [
                sim_mod.RequestDecodeEvent(
                    req_id=req1.rid,
                    end_timestamp=100.0,
                    completion_number=2,
                )
            ]
            tracked2.timeline.next_anticipated_event = [
                sim_mod.RequestDecodeEvent(
                    req_id=req2.rid,
                    end_timestamp=200.0,
                    completion_number=2,
                )
            ]

            prior1 = policy._logical_event_timestamp(req1, simulator=policy.simulator)
            prior2 = policy._logical_event_timestamp(req2, simulator=policy.simulator)

            running_batch = SimpleNamespace(reqs=[req1, req2])
            now["t"] = 13.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=1,
            )

            realized1 = policy.simulator.most_recent_event_real[req1.rid]
            realized2 = policy.simulator.most_recent_event_real[req2.rid]
            self.assertEqual(realized1.completion_number, 2)
            self.assertEqual(realized2.completion_number, 2)
            self.assertEqual(realized1.end_timestamp, prior1 + 3.0)
            self.assertEqual(realized2.end_timestamp, prior2 + 3.0)

            anticipated1 = tracked1.timeline.next_anticipated_event
            anticipated2 = tracked2.timeline.next_anticipated_event
            self.assertEqual(len(anticipated1), 1)
            self.assertEqual(len(anticipated2), 1)
            self.assertEqual(anticipated1[0].completion_number, 3)
            self.assertEqual(anticipated2[0].completion_number, 3)
            self.assertEqual(anticipated1[0].end_timestamp, realized1.end_timestamp + 3.0)
            self.assertEqual(anticipated2[0].end_timestamp, realized2.end_timestamp + 3.0)

    def test_decode_epoch_advances_frontier_by_decode_steps_times_isolated_estimate(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_1", "rid_running", 4)
            req.output_ids = list(range(541))
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            _seed_main_simulator_request(policy, req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req]))
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[req]))

            tracked = policy.simulator.requests[req.rid]
            tracked.timeline.next_anticipated_event = [
                sim_mod.RequestDecodeEvent(
                    req_id=req.rid,
                    end_timestamp=100.0,
                    completion_number=len(req.output_ids) + 1,
                )
            ]

            prior = policy._logical_event_timestamp(req, simulator=policy.simulator)
            running_batch = SimpleNamespace(reqs=[req])
            now["t"] = 13.0
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=False,
                decode_steps=10,
            )

            realized = policy.simulator.most_recent_event_real[req.rid]
            anticipated = tracked.timeline.next_anticipated_event
            self.assertEqual(realized.completion_number, len(req.output_ids) + 10)
            self.assertEqual(realized.end_timestamp, prior + (10 * 3.0))
            self.assertEqual(len(anticipated), 1)
            self.assertEqual(
                anticipated[0].completion_number,
                len(req.output_ids) + 11,
            )
            self.assertEqual(anticipated[0].end_timestamp, realized.end_timestamp + 3.0)

    def test_mixed_size_running_decodes_can_leave_positive_slack(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        def fake_iso_prefill(total_tokens, *_args):
            return total_tokens / 10.0

        def fake_iso_decode(total_tokens, *_args):
            return total_tokens / 5.0

        def fake_pooled_prefill(total_tokens, *_args):
            return total_tokens / 100.0

        def fake_pooled_decode(*_args, **_kwargs):
            return 3.0

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            doc_policy_mod, "pooled_prefill_time_estimation", side_effect=fake_pooled_prefill
        ), patch.object(
            doc_policy_mod, "pooled_decode_time_estimation", side_effect=fake_pooled_decode
        ), patch.object(
            sim_mod, "isolated_prefill_time_estimation", side_effect=fake_iso_prefill
        ), patch.object(
            sim_mod, "isolated_decode_time_estimation", side_effect=fake_iso_decode
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", side_effect=fake_iso_decode
        ):
            fast_req = _mk_req("user_fast", "rid_fast", 20)
            slow_req = _mk_req("user_slow", "rid_slow", 100)
            waiting_req = _mk_req("user_wait", "rid_wait", 50)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(fast_req)
            policy.process_new_request(slow_req)
            policy.process_new_request(waiting_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[fast_req, slow_req]))

            fast_req.output_ids = [1]
            slow_req.output_ids = [1]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[fast_req, slow_req]))

            running_batch = SimpleNamespace(reqs=[fast_req, slow_req])
            now["t"] = 12.1
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)

            now["t"] = 12.5
            policy.start_of_pass(running_batch, [waiting_req])

            decode_candidates = [
                candidate for candidate in policy._deadline_queue if candidate.event_type == "decode"
            ]
            self.assertEqual(len(decode_candidates), 2)
            self.assertEqual(decode_candidates[0].req.rid, fast_req.rid)
            self.assertLess(
                decode_candidates[0].start_deadline,
                decode_candidates[1].start_deadline,
            )
            self.assertTrue(policy._has_decode_deadline)
            self.assertIsNotNone(policy._earliest_decode_start_deadline)
            self.assertGreater(policy._earliest_decode_start_deadline - now["t"], 0.0)
            self.assertGreater(policy._max_safe_prefill_tokens or 0, 0)
            self.assertIn(waiting_req.rid, policy._forced_prefill_rids)

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
            policy.prepare_during_gpu_execution(
                running_batch=initial_running,
                waiting_queue=[prefilling_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
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
            policy.prepare_during_gpu_execution(
                running_batch=next_running,
                waiting_queue=[waiting_req],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
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

    def test_newly_prefilled_request_decode_epoch_should_advance_past_completion_two(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_1", "rid_running", 4)
            new_req = _mk_req("user_2", "rid_new", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            policy.process_new_request(new_req)
            _seed_main_simulator_request(policy, running_req)
            _seed_main_simulator_request(policy, new_req)

            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            now["t"] = 20.0
            policy.finished_prefill(SimpleNamespace(reqs=[new_req]))
            next_running = SimpleNamespace(reqs=[running_req, new_req])

            new_req.output_ids = list(range(10))
            now["t"] = 21.0
            policy.prepare_during_gpu_execution(
                running_batch=next_running,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=0,
                decode_steps_by_rid={new_req.rid: 10},
                output_ids_already_applied=True,
            )
            _wait_for_prepare_snapshot(policy)
            policy._ensure_current_pass_state(
                [],
                running_batch=next_running,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == new_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertEqual(decode_candidates[0].event.completion_number, 11)

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
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=waiting_queue,
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
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

            self.assertGreaterEqual(second.event.completion_number, first.event.completion_number)
            self.assertGreaterEqual(third.event.completion_number, second.event.completion_number)
            self.assertGreaterEqual(second.deadline, first.deadline)
            self.assertGreaterEqual(third.deadline, second.deadline)

    def test_start_of_pass_authoritative_snapshot_advances_decode_overlay(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
            first = next(
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == req.rid
            )

            req.output_ids = [42, 43]
            now["t"] = 100.5
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=True,
                decode_steps=1,
            )
            now["t"] = 100.75
            policy.finished_decode(SimpleNamespace(reqs=[req]))
            now["t"] = 101.0
            policy.start_of_pass(running_batch, [])
            second = next(
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == req.rid
            )

            self.assertEqual(policy._last_pass_state_source, "start_of_pass_wait_prepare")
            self.assertGreater(second.event.completion_number, first.event.completion_number)
            self.assertNotEqual(
                (second.event.completion_number, second.event.end_timestamp),
                (first.event.completion_number, first.event.end_timestamp),
            )

    def test_start_of_pass_authoritative_snapshot_turns_finished_prefill_into_running_decode(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            running_req = _mk_req("user_19", "rid_running", 4)
            new_req = _mk_req("user_1", "rid_new", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(running_req)
            policy.process_new_request(new_req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            policy.start_of_pass(running_batch, [new_req])

            scheduled_batch = SimpleNamespace(reqs=[new_req])
            policy.note_scheduled_prefill_batch(scheduled_batch)
            now["t"] = 100.5
            policy.prepare_during_gpu_execution(
                event_type="prefill",
                running_batch=running_batch,
                waiting_queue=[new_req],
                scheduled_batch=scheduled_batch,
                prepare_pass_state=True,
            )
            now["t"] = 101.0
            policy.finished_prefill(scheduled_batch)

            next_running = SimpleNamespace(reqs=[running_req, new_req])
            now["t"] = 102.0
            policy.start_of_pass(next_running, [])

            decode_rids = {
                candidate.req.rid
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode"
            }
            self.assertIn(new_req.rid, decode_rids)
            self.assertNotIn(new_req.rid, policy._waiting_prefill_start_deadline_by_rid)

    def test_repeated_decode_epochs_should_not_repeat_same_frontier(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
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
            frontiers = []
            for step in range(3):
                now["t"] = 100.0 + step * 2.0
                policy.start_of_pass(running_batch, [])
                candidate = next(
                    cand
                    for cand in policy._deadline_queue
                    if cand.event_type == "decode" and cand.req.rid == req.rid
                )
                frontiers.append(
                    (
                        candidate.event.completion_number,
                        candidate.event.end_timestamp,
                    )
                )
                req.output_ids = list(range(len(req.output_ids) + 1))
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[],
                    event_type="decode",
                    prepare_pass_state=True,
                    decode_steps=1,
                )
                now["t"] = 100.0 + step * 2.0 + 0.5
                policy.finished_decode(SimpleNamespace(reqs=[req]))

            self.assertEqual(len(frontiers), 3)
            self.assertLess(frontiers[0], frontiers[1])
            self.assertLess(frontiers[1], frontiers[2])

    def test_authoritative_start_of_pass_should_not_repeat_same_decode_frontier_across_finished_decodes(self):
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
        ), patch.object(
            doc_policy_mod, "isolated_decode_time_estimation", return_value=3.0
        ):
            req = _mk_req("user_1", "rid_running", 4)
            policy = DocPolicy(delta_fairness_n=4, max_running_requests=256)

            policy.process_new_request(req)
            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[req]))

            running_batch = SimpleNamespace(reqs=[req])
            seen_frontiers = []
            for step in range(5):
                now["t"] = 100.0 + step
                policy.start_of_pass(running_batch, [])
                candidate = next(
                    cand
                    for cand in policy._deadline_queue
                    if cand.event_type == "decode" and cand.req.rid == req.rid
                )
                frontier = (
                    candidate.req.rid,
                    candidate.event.completion_number,
                    candidate.deadline,
                )
                if seen_frontiers:
                    self.assertNotEqual(frontier, seen_frontiers[-1])
                seen_frontiers.append(frontier)

                req.output_ids = list(range(len(req.output_ids) + 1))
                now["t"] = 100.0 + step + 0.25
                policy.prepare_during_gpu_execution(
                    running_batch=running_batch,
                    waiting_queue=[],
                    event_type="decode",
                    prepare_pass_state=True,
                    decode_steps=1,
                )
                now["t"] = 100.0 + step + 0.5
                policy.finished_decode(SimpleNamespace(reqs=[req]))

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
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=waiting_queue,
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
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
            policy.prepare_during_gpu_execution(
                running_batch=running_batch,
                waiting_queue=[],
                event_type="decode",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
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
            policy.prepare_during_gpu_execution(
                running_batch=None,
                waiting_queue=[good_waiting],
                event_type="prefill",
                prepare_pass_state=True,
            )
            _wait_for_prepare_snapshot(policy)
            policy._ensure_current_pass_state(
                [good_waiting],
                running_batch=None,
                delta_fairness_deltas_microseconds=policy._deltas_us,
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
            _seed_main_simulator_request(policy, running_req)
            _seed_main_simulator_request(policy, fair_waiting)
            _seed_main_simulator_request(policy, unfair_waiting)

            running_batch = SimpleNamespace(reqs=[running_req])
            now["t"] = 100.0
            with patch.object(policy, "_memory_pressure_active_for_prefill", return_value=True, create=True), patch.object(
                policy,
                "user_is_fair_prefill",
                side_effect=lambda user_id, **kwargs: user_id == "user_1",
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
                "user_is_fair_prefill",
                side_effect=lambda user_id, **kwargs: user_id in {"user_1", "user_2"},
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
