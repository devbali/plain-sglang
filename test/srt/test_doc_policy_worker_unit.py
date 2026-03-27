import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.managers.tp_worker as tp_worker_mod
from sglang.global_config import global_config
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.managers.tp_worker import ModelTpServer


class _FakeDocPolicy:
    def __init__(self):
        self._forced_prefill_rids = {"rid_waiting"}
        self._max_safe_prefill_tokens = 101

    def refresh_decode_hot_path_state(self, waiting_queue, running_batch):
        self.hot_path_args = (list(waiting_queue), running_batch)

    def fairinf_prioritize_force_prefill(self):
        return True

    def fairinf_force_decode(self, running_batch, *, delta_fairness_deltas_microseconds=None):
        return False, self._max_safe_prefill_tokens


class TestDocPolicyWorkerUnit(unittest.TestCase):
    def test_force_prefill_path_passes_safe_prefill_cap_into_builder(self):
        fake_policy = _FakeDocPolicy()
        captured = {}

        class FakeServer:
            pass

        server = FakeServer()
        server.fairness_policy = fake_policy
        server.waiting_queue = [SimpleNamespace(rid="rid_waiting")]
        server.running_batch = None
        server.delta_fairness_deltas_microseconds = {"decode": 0}
        server.max_running_requests = 256
        server._last_prepare_async_wait_ms = 0.0
        server.new_token_ratio = 0.0
        server._log_scheduler_pass = lambda **kwargs: captured.setdefault(
            "log_scheduler", kwargs
        )
        server._log_intermediate_gap = lambda **kwargs: captured.setdefault(
            "log_gap", kwargs
        )
        server.check_memory = lambda: None
        server.print_stats = lambda **kwargs: None
        server.forward_prefill_batch = lambda batch: None
        server.forward_decode_batch = lambda *args, **kwargs: None
        def get_new_prefill_batch(max_prefill_size, telemetry=None):
            captured["max_prefill_size"] = max_prefill_size
            return None

        server.get_new_prefill_batch = get_new_prefill_batch
        server.num_generated_tokens = 0
        server.max_total_num_tokens = 1
        server.last_stats_tic = 1.0
        server.tp_rank = 1
        server.out_pyobjs = []

        class _FakeCudaEvent:
            def __init__(self, enable_timing=True):
                pass

            def record(self):
                pass

            def query(self):
                return True

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 0.0

        with patch.object(tp_worker_mod, "DocPolicy", _FakeDocPolicy), patch.object(
            tp_worker_mod.torch.cuda, "Event", _FakeCudaEvent
        ):
            ModelTpServer.forward_step(server)

        self.assertEqual(captured["max_prefill_size"], 101)
        self.assertEqual(captured["log_scheduler"]["max_prefill_size"], 101)
        self.assertEqual(fake_policy.hot_path_args[0][0].rid, "rid_waiting")

    def test_get_new_prefill_batch_should_not_return_early_when_force_prefill_can_retract_for_slots(self):
        force_reservation_called = {"called": False}
        server_holder = {}

        class _FakeDocPolicyForBatch:
            def __init__(self):
                self._forced_prefill_rids = {"rid_waiting"}
                self._last_pass_breakdown_ms = {}

            def start_of_pass(self, running_batch, waiting_queue, *, new_token_ratio=0.0, max_running_requests=None):
                del running_batch, waiting_queue, new_token_ratio, max_running_requests

            def force_prefill_reservations(
                self,
                waiting_queue,
                *,
                token_counters_by_user,
                adder,
                token_to_kv_pool=None,
                running_batch=None,
                delta_fairness_deltas_microseconds=None,
                max_input_size=None,
                prefix_computed=False,
                max_running_requests=None,
            ):
                del (
                    waiting_queue,
                    token_counters_by_user,
                    token_to_kv_pool,
                    running_batch,
                    delta_fairness_deltas_microseconds,
                    max_input_size,
                    prefix_computed,
                    max_running_requests,
                )
                force_reservation_called["called"] = True
                server_holder["server"].running_batch.reqs = []
                adder.can_run_list.append(server_holder["server"].waiting_queue[0])
                return 101, [SimpleNamespace(rid="rid_evicted")]

            def process_waiting_queue_prefills(
                self,
                waiting_queue,
                *,
                adder,
                token_counters_by_user,
                prefix_computed,
                running_batch,
                running_batch_size,
                max_running_requests,
                available_req_slots,
                max_input_size,
            ):
                del (
                    waiting_queue,
                    token_counters_by_user,
                    prefix_computed,
                    running_batch,
                    running_batch_size,
                    max_running_requests,
                    available_req_slots,
                    max_input_size,
                )
                adder.can_run_list.append(SimpleNamespace(reqs=[]))

        class _FakePrefillAdder:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.can_run_list = []
                self.new_inflight_req = None
                self.log_input_tokens = 0
                self.rem_input_tokens = 10_000
                self.rem_total_tokens = 10_000

            def remove_running_tokens(self, running_batch, new_token_ratio):
                del running_batch, new_token_ratio

        class FakeServer:
            pass

        server = FakeServer()
        server_holder["server"] = server
        server._capture_doc_policy_pass_snapshot_state = lambda: None
        server.running_batch = SimpleNamespace(reqs=[SimpleNamespace(rid="rid_running")] * 1)
        server.max_running_requests = 1
        server.req_to_token_pool = SimpleNamespace(free_slots=[1])
        server.waiting_queue = [SimpleNamespace(rid="rid_waiting", uid="1")]
        server.current_inflight_req = None
        server.fairness_policy = _FakeDocPolicyForBatch()
        server.scheduler = SimpleNamespace(calc_priority=lambda waiting_queue: False)
        server.is_mixed_chunk = False
        server.max_prefill_tokens = 5000
        server.chunked_prefill_size = 5000
        server.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
        server.token_to_kv_pool = SimpleNamespace(available_size=lambda: 0)
        server.new_token_ratio = 0.0
        server.delta_fairness_deltas_microseconds = {"decode": 0}
        server.delta_fairness_n = 2
        server._maybe_dump_doc_policy_pass_snapshot = lambda *args, **kwargs: None
        server.tp_rank = 1

        with patch.object(tp_worker_mod, "PrefillAdder", _FakePrefillAdder), patch.object(
            tp_worker_mod.ScheduleBatch, "init_new", return_value=SimpleNamespace(reqs=[])
        ), patch.object(tp_worker_mod, "DocPolicy", _FakeDocPolicyForBatch):
            batch = ModelTpServer.get_new_prefill_batch(server, max_prefill_token_size=101, telemetry={})

        self.assertTrue(force_reservation_called["called"])
        self.assertIsNotNone(batch)

    def test_force_prefill_reservation_retracts_running_decode_when_capacity_is_full(self):
        policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)

        class _TreeCache:
            fairinf_max_per_user = 1

            def evictable_size(self):
                return 0

        class _TokenPool:
            def __init__(self):
                self.available = 0

            def available_size(self):
                return self.available

        class _Adder:
            def __init__(self):
                self.rem_total_tokens = 0
                self.rem_input_tokens = 10_000
                self.log_input_tokens = 0
                self.can_run_list = []

            def expand_capacity(self, delta):
                self.rem_total_tokens += delta

            def add_one_req(self, req, new_extra_for_user):
                self.can_run_list.append((req.rid, new_extra_for_user))
                return "ok"

        pool = _TokenPool()
        adder = _Adder()

        waiting_req = SimpleNamespace(
            rid="rid_waiting_fair",
            uid="1",
            origin_input_ids=[1] * 101,
            extend_input_len=101,
            sampling_params=SimpleNamespace(max_new_tokens=50),
            waiting_time_in_decodes=0,
        )
        waiting_req.init_next_round_input = Mock(return_value="ok")
        waiting_req.get_estimated_prefill_impact = Mock(return_value=101)

        evicted_req = SimpleNamespace(rid="rid_running_unfair", uid="19")

        def retract_decode(required_tokens):
            del required_tokens
            pool.available = 512
            return [evicted_req], 1.0

        running_batch = SimpleNamespace(
            batch_size=lambda: 1,
            retract_decode=Mock(side_effect=retract_decode),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {waiting_req.rid}
        policy._forced_prefill_queue = [waiting_req]
        policy._ensure_current_pass_state = Mock()
        policy.note_retracted_reqs = Mock()
        policy.req_is_fair_prefill = Mock(return_value=True)
        policy._force_prefill_within_user_headroom = Mock(return_value=True)

        extra_space, last_evicted = policy.force_prefill_reservations(
            [waiting_req],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertGreater(extra_space, 0)
        self.assertEqual(last_evicted, [evicted_req])
        running_batch.retract_decode.assert_called()
        policy.note_retracted_reqs.assert_called_once_with([evicted_req])
        self.assertEqual(adder.can_run_list, [(waiting_req.rid, waiting_req.extend_input_len)])
        waiting_req.init_next_round_input.assert_called_once()

    def test_force_prefill_reservations_only_admits_deadline_safe_fair_prefix_under_memory_pressure(self):
        policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)

        class _TreeCache:
            fairinf_max_per_user = 1

            def evictable_size(self):
                return 0

        class _TokenPool:
            def __init__(self):
                self.available = 0

            def available_size(self):
                return self.available

        class _Adder:
            def __init__(self):
                self.rem_total_tokens = 0
                self.rem_input_tokens = 10_000
                self.log_input_tokens = 0
                self.can_run_list = []

            def expand_capacity(self, delta):
                self.rem_total_tokens += delta

            def add_one_req(self, req, new_extra_for_user):
                self.can_run_list.append((req.rid, new_extra_for_user))
                return "ok"

        pool = _TokenPool()
        adder = _Adder()

        fair_waiting = SimpleNamespace(
            rid="rid_waiting_fair",
            uid="1",
            origin_input_ids=[1] * 101,
            extend_input_len=101,
            sampling_params=SimpleNamespace(max_new_tokens=50),
            waiting_time_in_decodes=0,
        )
        fair_waiting.init_next_round_input = Mock(return_value="ok")
        fair_waiting.get_estimated_prefill_impact = Mock(return_value=101)

        unfair_waiting = SimpleNamespace(
            rid="rid_waiting_unfair",
            uid="19",
            origin_input_ids=[1] * 101,
            extend_input_len=101,
            sampling_params=SimpleNamespace(max_new_tokens=50),
            waiting_time_in_decodes=0,
        )
        unfair_waiting.init_next_round_input = Mock(return_value="ok")
        unfair_waiting.get_estimated_prefill_impact = Mock(return_value=101)

        evicted_req = SimpleNamespace(rid="rid_running_unfair", uid="19")

        def retract_decode(required_tokens):
            del required_tokens
            # Enough room for multiple requests, so deadline/fair-prefix logic
            # should be what limits admission here.
            pool.available = 512
            return [evicted_req], 1.0

        running_batch = SimpleNamespace(
            batch_size=lambda: 1,
            retract_decode=Mock(side_effect=retract_decode),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {fair_waiting.rid, unfair_waiting.rid}
        policy._forced_prefill_queue = [fair_waiting, unfair_waiting]
        policy._max_safe_prefill_tokens = 101
        policy._ensure_current_pass_state = Mock()
        policy.note_retracted_reqs = Mock()
        policy.req_is_fair_prefill = Mock(side_effect=lambda req, **kwargs: req.uid == "1")
        policy._force_prefill_within_user_headroom = Mock(side_effect=lambda req, **kwargs: req.uid == "1")

        extra_space, last_evicted = policy.force_prefill_reservations(
            [fair_waiting, unfair_waiting],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertGreater(extra_space, 0)
        self.assertEqual(last_evicted, [evicted_req])
        running_batch.retract_decode.assert_called_once()
        policy.note_retracted_reqs.assert_called_once_with([evicted_req])
        self.assertEqual(adder.can_run_list, [(fair_waiting.rid, fair_waiting.extend_input_len)])
        fair_waiting.init_next_round_input.assert_called_once()
        unfair_waiting.init_next_round_input.assert_not_called()
        self.assertGreaterEqual(policy.req_is_fair_prefill.call_count, 1)
        self.assertGreaterEqual(policy._force_prefill_within_user_headroom.call_count, 1)

    def test_grouped_decode_segment_reports_decode_rounds_once(self):
        decode_round_records = []

        class _Policy:
            def fairinf_prioritize_force_prefill(self):
                return False

            def fairinf_overdue_decode_subset_rids(self, running_batch):
                return None

            def launch_async_decode_epoch_prepare(
                self, running_batch, waiting_queue, selected_rids, decode_steps
            ):
                return False

            def fairinf_force_decode(self, running_batch, *, delta_fairness_deltas_microseconds=None):
                return True, 0

            def finished_decode(self, batch, decode_rounds_arg=1, decode_rounds=None):
                rounds = decode_rounds if decode_rounds is not None else decode_rounds_arg
                decode_round_records.append((rounds, sorted(req.rid for req in batch.reqs)))

        req = SimpleNamespace(rid="rid_running", waiting_time_in_decodes=0)
        running_batch = SimpleNamespace(
            reqs=[req],
            max_running_requests=None,
            delta_fairness_n=None,
            is_empty=lambda: False,
        )

        class FakeServer:
            pass

        server = FakeServer()
        server.fairness_policy = _Policy()
        server.waiting_queue = []
        server.running_batch = running_batch
        server.delta_fairness_deltas_microseconds = {"decode": 0}
        server.max_running_requests = 256
        server.delta_fairness_n = 2
        server._last_prepare_async_wait_ms = 0.0
        server.new_token_ratio = 0.0
        server.num_generated_tokens = 0
        server._log_scheduler_pass = lambda **kwargs: None
        server._log_intermediate_gap = lambda **kwargs: None
        server.print_stats = lambda **kwargs: None
        server.check_memory = lambda: None
        server.forward_prefill_batch = lambda batch: None
        server.out_pyobjs = []

        def forward_decode_batch(
            batch, selected_rids=None, prepare_pass_state=False, decode_steps=1
        ):
            del decode_steps
            return None

        server.forward_decode_batch = forward_decode_batch
        server.get_new_prefill_batch = lambda max_prefill_size, telemetry=None: None

        class _FakeCudaEvent:
            def __init__(self, enable_timing=True):
                pass

            def record(self):
                pass

            def query(self):
                return True

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 0.0

        old_steps = global_config.num_continue_decode_steps
        global_config.num_continue_decode_steps = 10
        try:
            with patch.object(tp_worker_mod.torch.cuda, "Event", _FakeCudaEvent):
                ModelTpServer.forward_step(server)
        finally:
            global_config.num_continue_decode_steps = old_steps

        self.assertEqual(decode_round_records, [(10, ["rid_running"])])


if __name__ == "__main__":
    unittest.main()
