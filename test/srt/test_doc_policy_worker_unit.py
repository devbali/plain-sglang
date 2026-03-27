import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.managers.tp_worker as tp_worker_mod
from sglang.global_config import global_config
from sglang.srt.managers.tp_worker import ModelTpServer


class _FakeDocPolicy:
    def __init__(self):
        self._forced_prefill_rids = {"rid_waiting"}
        self._max_safe_prefill_tokens = 101

    def _ensure_current_pass_state(self, waiting_queue, running_batch, deltas):
        self.ensure_args = (list(waiting_queue), running_batch, deltas)

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
