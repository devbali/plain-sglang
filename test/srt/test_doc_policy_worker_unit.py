import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.delta_fairness.doc_policy as doc_policy_mod
import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
import sglang.srt.managers.tp_worker as tp_worker_mod
from sglang.global_config import global_config
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tp_worker import ModelTpServer
from sglang.srt.sampling.sampling_params import SamplingParams


def _mk_req(uid: str, rid: str, prompt_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * prompt_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    req.sampling_params = SamplingParams(max_new_tokens=1000, min_new_tokens=0)
    req.waiting_time_in_decodes = 0
    return req


def _wait_for_prepare_snapshot(policy: DocPolicy) -> None:
    policy._prepare_worker.wait_for_snapshot(
        min_task_seq=policy._prepare_worker._task_seq,
        timeout_s=0.2,
    )


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

            def req_is_fair_prefill(self, req, **kwargs):
                del req, kwargs
                return True

            def _force_prefill_within_user_headroom(self, req, **kwargs):
                del req, kwargs
                return True

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

        self.assertEqual(extra_space, 151)
        self.assertEqual(last_evicted, [evicted_req])
        running_batch.retract_decode.assert_called()
        policy.note_retracted_reqs.assert_called_once_with([evicted_req])
        self.assertEqual(adder.can_run_list, [])
        waiting_req.init_next_round_input.assert_not_called()

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
        policy.user_is_fair_prefill = Mock(side_effect=lambda user_id, **kwargs: user_id == "1")
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

        self.assertEqual(extra_space, 151)
        self.assertEqual(last_evicted, [evicted_req])
        running_batch.retract_decode.assert_called_once()
        policy.note_retracted_reqs.assert_called_once_with([evicted_req])
        self.assertEqual(adder.can_run_list, [])
        fair_waiting.init_next_round_input.assert_not_called()
        unfair_waiting.init_next_round_input.assert_not_called()
        self.assertGreaterEqual(policy.user_is_fair_prefill.call_count, 1)
        self.assertGreaterEqual(policy._force_prefill_within_user_headroom.call_count, 1)

    def test_force_prefill_retractions_only_make_room_for_fair_waiting_requests(self):
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

        def make_waiting(rid, uid):
            req = SimpleNamespace(
                rid=rid,
                uid=uid,
                origin_input_ids=[1] * 101,
                extend_input_len=101,
                sampling_params=SimpleNamespace(max_new_tokens=50),
                waiting_time_in_decodes=0,
            )
            req.init_next_round_input = Mock(return_value="ok")
            req.get_estimated_prefill_impact = Mock(return_value=101)
            return req

        good_waiting = make_waiting("rid_waiting_good", "1")
        bad_waiting = make_waiting("rid_waiting_bad", "19")
        evicted_bad = SimpleNamespace(rid="rid_running_bad", uid="19")

        pool = _TokenPool()
        adder = _Adder()

        def retract_decode(required_tokens):
            del required_tokens
            pool.available = 512
            return [evicted_bad], 1.0

        running_batch = SimpleNamespace(
            batch_size=lambda: 1,
            retract_decode=Mock(side_effect=retract_decode),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {good_waiting.rid, bad_waiting.rid}
        policy._forced_prefill_queue = [good_waiting, bad_waiting]
        policy._max_safe_prefill_tokens = 202
        policy._ensure_current_pass_state = Mock()
        policy.note_retracted_reqs = Mock()
        policy.user_is_fair_prefill = Mock(side_effect=lambda user_id, **kwargs: user_id == "1")
        policy._force_prefill_within_user_headroom = Mock(return_value=True)

        extra_space, last_evicted = policy.force_prefill_reservations(
            [good_waiting, bad_waiting],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertEqual(extra_space, 151)
        self.assertEqual(last_evicted, [evicted_bad])
        running_batch.retract_decode.assert_called_once()
        policy.note_retracted_reqs.assert_called_once_with([evicted_bad])
        self.assertEqual(adder.can_run_list, [])
        good_waiting.init_next_round_input.assert_not_called()
        bad_waiting.init_next_round_input.assert_not_called()

    def test_bad_waiting_request_does_not_retract_bad_running_request_to_admit_itself(self):
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

        bad_waiting = SimpleNamespace(
            rid="rid_waiting_bad",
            uid="19",
            origin_input_ids=[1] * 101,
            extend_input_len=101,
            sampling_params=SimpleNamespace(max_new_tokens=50),
            waiting_time_in_decodes=0,
        )
        bad_waiting.init_next_round_input = Mock(return_value="ok")
        bad_waiting.get_estimated_prefill_impact = Mock(return_value=101)

        evicted_bad = SimpleNamespace(rid="rid_running_bad", uid="19")

        pool = _TokenPool()
        adder = _Adder()

        def retract_decode(required_tokens):
            del required_tokens
            pool.available = 512
            return [evicted_bad], 1.0

        running_batch = SimpleNamespace(
            batch_size=lambda: 1,
            retract_decode=Mock(side_effect=retract_decode),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {bad_waiting.rid}
        policy._forced_prefill_queue = [bad_waiting]
        policy._max_safe_prefill_tokens = 101
        policy._ensure_current_pass_state = Mock()
        policy.note_retracted_reqs = Mock()
        policy.req_is_fair_prefill = Mock(return_value=False)
        policy._force_prefill_within_user_headroom = Mock(return_value=False)

        extra_space, last_evicted = policy.force_prefill_reservations(
            [bad_waiting],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertEqual(extra_space, 0)
        self.assertIsNone(last_evicted)
        running_batch.retract_decode.assert_not_called()
        policy.note_retracted_reqs.assert_not_called()
        self.assertEqual(adder.can_run_list, [])
        bad_waiting.init_next_round_input.assert_not_called()

    def test_force_prefill_reservation_does_not_pay_global_adder_deficit(self):
        policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)

        class _TreeCache:
            fairinf_max_per_user = 1

            def evictable_size(self):
                return 0

            def inc_lock_ref(self, node):
                del node
                return 0

            def dec_lock_ref(self, node):
                del node
                return 0

        class _TokenPool:
            def __init__(self):
                self.available = 1759

            def available_size(self):
                return self.available

        class _Adder:
            def __init__(self):
                self.rem_total_tokens = -92_000
                self.rem_input_tokens = 10_000
                self.log_input_tokens = 0
                self.can_run_list = []

            def expand_capacity(self, delta):
                self.rem_total_tokens += delta

            def add_one_req(self, req, new_extra_for_user):
                self.can_run_list.append((req.rid, new_extra_for_user))
                return "ok"

        good_waiting = SimpleNamespace(
            rid="rid_waiting_good",
            uid="1",
            origin_input_ids=[1] * 100,
            extend_input_len=100,
            prefix_indices=[],
            last_node=object(),
            fill_ids=[1] * 100,
            sampling_params=SimpleNamespace(max_new_tokens=51),
            waiting_time_in_decodes=0,
        )
        good_waiting.init_next_round_input = Mock(return_value="ok")
        good_waiting.get_estimated_prefill_impact = Mock(return_value=100)

        pool = _TokenPool()
        adder = _Adder()

        running_batch = SimpleNamespace(
            batch_size=lambda: 1,
            retract_decode=Mock(return_value=([], 1.0)),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {good_waiting.rid}
        policy._forced_prefill_queue = [good_waiting]
        policy._max_safe_prefill_tokens = 151
        policy.note_retracted_reqs = Mock()
        policy.req_is_fair_prefill = Mock(return_value=True)
        policy._force_prefill_within_user_headroom = Mock(return_value=True)

        extra_space, last_evicted = policy.force_prefill_reservations(
            [good_waiting],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertEqual(extra_space, 151)
        self.assertIsNone(last_evicted)
        running_batch.retract_decode.assert_not_called()
        self.assertEqual(adder.can_run_list, [])
        good_waiting.init_next_round_input.assert_not_called()

    def test_exact_forced_prefill_stops_retracting_after_first_adder_reject(self):
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
            def __init__(self, pool):
                self.pool = pool
                self.rem_total_tokens = 0
                self.rem_input_tokens = 10_000
                self.log_input_tokens = 0
                self.can_run_list = []
                self.calls = 0

            def expand_capacity(self, delta):
                self.rem_total_tokens += delta
                self.pool.available += delta

            def add_one_req(self, req, new_extra_for_user):
                del new_extra_for_user
                self.calls += 1
                if self.calls == 1:
                    self.can_run_list.append((req.rid, req.extend_input_len))
                    self.pool.available = 0
                    return "ok"
                return "rejected"

        def make_waiting(rid):
            req = SimpleNamespace(
                rid=rid,
                uid="1",
                origin_input_ids=[1] * 101,
                extend_input_len=101,
                sampling_params=SimpleNamespace(max_new_tokens=50),
                waiting_time_in_decodes=0,
            )
            req.init_next_round_input = Mock(return_value="ok")
            req.get_estimated_prefill_impact = Mock(return_value=101)
            return req

        waiting_1 = make_waiting("rid_waiting_1")
        waiting_2 = make_waiting("rid_waiting_2")
        waiting_3 = make_waiting("rid_waiting_3")

        pool = _TokenPool()
        adder = _Adder(pool)
        evicted = [
            SimpleNamespace(rid="rid_running_bad_1", uid="19"),
            SimpleNamespace(rid="rid_running_bad_2", uid="19"),
            SimpleNamespace(rid="rid_running_bad_3", uid="19"),
        ]

        def retract_decode(required_tokens):
            del required_tokens
            pool.available = 151
            return [evicted[running_batch.retract_decode.call_count]], 1.0

        running_batch = SimpleNamespace(
            batch_size=lambda: 3,
            retract_decode=Mock(side_effect=retract_decode),
            retract_decode_for_slots=Mock(return_value=([], 1.0)),
        )

        policy.tree_cache = _TreeCache()
        policy._forced_prefill_rids = {waiting_1.rid, waiting_2.rid, waiting_3.rid}
        policy._forced_prefill_queue = [waiting_1, waiting_2, waiting_3]
        policy._max_safe_prefill_tokens = 303
        policy.note_retracted_reqs = Mock()
        policy.user_is_fair_prefill = Mock(return_value=True)
        policy._force_prefill_within_user_headroom = Mock(return_value=True)

        extra_space, last_evicted = policy.force_prefill_reservations(
            [waiting_1, waiting_2, waiting_3],
            token_counters_by_user={},
            adder=adder,
            token_to_kv_pool=pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=policy._deltas_us,
            max_input_size=None,
            prefix_computed=True,
            max_running_requests=256,
        )

        self.assertEqual(extra_space, 453)
        self.assertEqual(running_batch.retract_decode.call_count, 1)
        policy.note_retracted_reqs.assert_called_once()
        self.assertEqual(last_evicted, policy.note_retracted_reqs.call_args.args[0])
        self.assertEqual(adder.can_run_list, [])
        waiting_1.init_next_round_input.assert_not_called()
        waiting_2.init_next_round_input.assert_not_called()
        waiting_3.init_next_round_input.assert_not_called()

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

    def test_grouped_decode_epoch_with_real_doc_policy_advances_new_request_past_completion_two(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            tp_worker_mod.time, "time", side_effect=fake_time
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
            policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)
            old_req = _mk_req("user_1", "rid_old", 4)
            new_req = _mk_req("user_2", "rid_new", 4)

            policy.process_new_request(old_req)
            policy.process_new_request(new_req)
            policy.simulator.process_new_request(old_req, policy._deltas_us)
            policy.simulator.process_new_request(new_req, policy._deltas_us)

            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[old_req]))
            old_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[old_req]))

            now["t"] = 20.0
            policy.finished_prefill(SimpleNamespace(reqs=[new_req]))

            running_batch = SimpleNamespace(
                reqs=[old_req, new_req],
                max_running_requests=None,
                delta_fairness_n=None,
                is_empty=lambda: False,
            )

            class FakeServer:
                pass

            server = FakeServer()
            server.fairness_policy = policy
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
            server.get_new_prefill_batch = lambda max_prefill_size, telemetry=None: None

            def forward_decode_batch(
                batch, selected_rids=None, prepare_pass_state=False, decode_steps=0
            ):
                del prepare_pass_state, decode_steps
                chosen = selected_rids
                now["t"] += 0.025
                for req in batch.reqs:
                    if chosen is None or req.rid in chosen:
                        req.output_ids.append(0)
                return None

            server.forward_decode_batch = forward_decode_batch

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

            _wait_for_prepare_snapshot(policy)
            policy._ensure_current_pass_state(
                [],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == new_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertEqual(decode_candidates[0].event.completion_number, 11)

    def test_grouped_decode_epoch_with_real_doc_policy_advances_new_request_timestamp_by_epoch(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            tp_worker_mod.time, "time", side_effect=fake_time
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
            policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)
            old_req = _mk_req("user_1", "rid_old_ts", 4)
            new_req = _mk_req("user_2", "rid_new_ts", 4)

            policy.process_new_request(old_req)
            policy.process_new_request(new_req)
            policy.simulator.process_new_request(old_req, policy._deltas_us)
            policy.simulator.process_new_request(new_req, policy._deltas_us)

            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[old_req]))
            old_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[old_req]))

            now["t"] = 20.0
            policy.finished_prefill(SimpleNamespace(reqs=[new_req]))
            prior_candidate = policy._decode_candidate_for_req(
                new_req, SimpleNamespace(reqs=[old_req, new_req])
            )
            self.assertIsNotNone(prior_candidate)

            running_batch = SimpleNamespace(
                reqs=[old_req, new_req],
                max_running_requests=None,
                delta_fairness_n=None,
                is_empty=lambda: False,
            )

            class FakeServer:
                pass

            server = FakeServer()
            server.fairness_policy = policy
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
            server.get_new_prefill_batch = lambda max_prefill_size, telemetry=None: None

            def forward_decode_batch(
                batch, selected_rids=None, prepare_pass_state=False, decode_steps=0
            ):
                del prepare_pass_state, decode_steps
                chosen = selected_rids
                now["t"] += 0.025
                for req in batch.reqs:
                    if chosen is None or req.rid in chosen:
                        req.output_ids.append(0)
                return None

            server.forward_decode_batch = forward_decode_batch

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

            _wait_for_prepare_snapshot(policy)
            policy._ensure_current_pass_state(
                [],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=policy._deltas_us,
            )

            decode_candidates = [
                candidate
                for candidate in policy._deadline_queue
                if candidate.event_type == "decode" and candidate.req.rid == new_req.rid
            ]
            self.assertEqual(len(decode_candidates), 1)
            self.assertGreater(
                decode_candidates[0].event.end_timestamp - prior_candidate.event.end_timestamp,
                25.0,
            )

    def test_grouped_decode_epoch_with_real_doc_policy_does_not_stay_stuck_at_completion_two_across_passes(self):
        now = {"t": 10.0}

        def fake_time():
            return now["t"]

        with patch.object(doc_policy_mod.time, "time", side_effect=fake_time), patch.object(
            sim_mod.time, "time", side_effect=fake_time
        ), patch.object(
            tp_worker_mod.time, "time", side_effect=fake_time
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
            policy = DocPolicy(delta_fairness_n=2, max_running_requests=256)
            old_req = _mk_req("user_1", "rid_old_repeat", 4)
            new_req = _mk_req("user_2", "rid_new_repeat", 4)

            policy.process_new_request(old_req)
            policy.process_new_request(new_req)
            policy.simulator.process_new_request(old_req, policy._deltas_us)
            policy.simulator.process_new_request(new_req, policy._deltas_us)

            now["t"] = 11.0
            policy.finished_prefill(SimpleNamespace(reqs=[old_req]))
            old_req.output_ids = [42]
            now["t"] = 12.0
            policy.finished_decode(SimpleNamespace(reqs=[old_req]))

            now["t"] = 20.0
            policy.finished_prefill(SimpleNamespace(reqs=[new_req]))

            running_batch = SimpleNamespace(
                reqs=[old_req, new_req],
                max_running_requests=None,
                delta_fairness_n=None,
                is_empty=lambda: False,
            )

            class FakeServer:
                pass

            server = FakeServer()
            server.fairness_policy = policy
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
            server.get_new_prefill_batch = lambda max_prefill_size, telemetry=None: None

            def forward_decode_batch(
                batch, selected_rids=None, prepare_pass_state=False, decode_steps=0
            ):
                del prepare_pass_state, decode_steps
                chosen = selected_rids
                now["t"] += 0.025
                for req in batch.reqs:
                    if chosen is None or req.rid in chosen:
                        req.output_ids.append(0)
                return None

            server.forward_decode_batch = forward_decode_batch

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
                    _wait_for_prepare_snapshot(policy)
                    policy._ensure_current_pass_state(
                        [],
                        running_batch=running_batch,
                        delta_fairness_deltas_microseconds=policy._deltas_us,
                    )
                    first_completion = next(
                        candidate.event.completion_number
                        for candidate in policy._deadline_queue
                        if candidate.event_type == "decode"
                        and candidate.req.rid == new_req.rid
                    )
                    ModelTpServer.forward_step(server)
                    _wait_for_prepare_snapshot(policy)
                    policy._ensure_current_pass_state(
                        [],
                        running_batch=running_batch,
                        delta_fairness_deltas_microseconds=policy._deltas_us,
                    )
                    second_completion = next(
                        candidate.event.completion_number
                        for candidate in policy._deadline_queue
                        if candidate.event_type == "decode"
                        and candidate.req.rid == new_req.rid
                    )
            finally:
                global_config.num_continue_decode_steps = old_steps

            self.assertGreater(first_completion, 2)
            self.assertGreater(second_completion, first_completion)

    def test_get_new_prefill_batch_retraction_only_admits_exact_fair_safe_subset(self):
        class _FakeDocPolicyForBatch:
            def __init__(self):
                self._forced_prefill_rids = {"rid_good", "rid_good_2"}
                self._max_safe_prefill_tokens = 202
                self._last_pass_breakdown_ms = {}

            def start_of_pass(
                self,
                running_batch,
                waiting_queue,
                *,
                new_token_ratio=0.0,
                max_running_requests=None,
            ):
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
                adder.can_run_list.append(server.waiting_queue[0])
                adder.can_run_list.append(server.waiting_queue[1])
                return 101, [SimpleNamespace(rid="rid_evicted_bad", uid="19")]

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
                    token_counters_by_user,
                    prefix_computed,
                    running_batch,
                    running_batch_size,
                    max_running_requests,
                    available_req_slots,
                    max_input_size,
                )
                for req in waiting_queue[2:]:
                    adder.can_run_list.append(req)

            def req_is_fair_prefill(self, req, **kwargs):
                del kwargs
                return req.uid in {"1", "2"}

            def _force_prefill_within_user_headroom(self, req, **kwargs):
                del req, kwargs
                return True

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

        server = SimpleNamespace()
        server._capture_doc_policy_pass_snapshot_state = lambda: None
        server.running_batch = SimpleNamespace(reqs=[SimpleNamespace(rid="rid_running_bad", uid="19")])
        server.max_running_requests = 8
        server.req_to_token_pool = SimpleNamespace(free_slots=list(range(8)))
        server.waiting_queue = [
            SimpleNamespace(rid="rid_good", uid="1"),
            SimpleNamespace(rid="rid_good_2", uid="2"),
            SimpleNamespace(rid="rid_good_3", uid="1"),
            SimpleNamespace(rid="rid_bad", uid="19"),
            SimpleNamespace(rid="rid_bad_2", uid="20"),
        ]
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
            tp_worker_mod.ScheduleBatch,
            "init_new",
            side_effect=lambda can_run_list, *args, **kwargs: SimpleNamespace(reqs=list(can_run_list)),
        ), patch.object(tp_worker_mod, "DocPolicy", _FakeDocPolicyForBatch):
            batch = ModelTpServer.get_new_prefill_batch(
                server, max_prefill_token_size=202, telemetry={}
            )

        self.assertIsNotNone(batch)
        self.assertEqual([req.rid for req in batch.reqs], ["rid_good", "rid_good_2"])


if __name__ == "__main__":
    unittest.main()
