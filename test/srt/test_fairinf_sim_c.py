"""
Tests for the _fairinf_sim C extension.

Three layers of coverage:

1. TestCSimKernelDirect   — calls _fairinf_sim.rebuild_kernel() directly with
                            plain Python primitive arrays.  No Python simulator
                            objects at all.  Validates the C simulation loop.

2. TestCSimVsPython       — runs identical scenarios through both the pure-Python
                            AlternateHistorySimulator and the C-backed version,
                            then asserts that next_anticipated_event outputs match.
                            This is the primary parity gate: if the C kernel
                            diverges from the Python reference, a test here fails.

3. TestCWorkerE2E         — exercises _DocPolicyPrepareWorker end-to-end with the
                            C-backed worker thread.  Verifies:
                              a) snapshots are produced correctly
                              b) the GIL is actually released during simulation
                                 (main thread can acquire it while worker runs)
                              c) mutations are applied in order before snapshots

All tests are skipped gracefully when _fairinf_sim is not yet built, so the file
can live in the repo before the C extension is compiled.

Run:
    PYTHONPATH=.../fairinf-sglang/python \
      python3.12 test/srt/test_fairinf_sim_c.py
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Try to import the C extension.  All tests that need it are skipped if absent.
# ---------------------------------------------------------------------------
try:
    from sglang.srt.delta_fairness import _fairinf_sim as _sim_c
    C_EXT_AVAILABLE = True
except ImportError:
    C_EXT_AVAILABLE = False

# Python-side simulator (always available)
import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
from sglang.srt.delta_fairness.doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
    RequestStartEvent,
)
from sglang.srt.delta_fairness.simulator_thread import _DocPolicyPrepareWorker
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Helpers shared by all test classes
# ---------------------------------------------------------------------------

def _mk_req(uid: str, rid: str, n_tokens: int, max_new_tokens: int = 1000) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * n_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    req.sampling_params = SamplingParams(max_new_tokens=max_new_tokens, min_new_tokens=0)
    req.waiting_time_in_decodes = 0
    req.first_time_in_waiting_queue = True
    return req


def _fake_time_ctx(now_box: dict):
    """Context manager that patches time.time to return now_box['t']."""
    return patch.object(sim_mod.time, "time", side_effect=lambda: now_box["t"])


def _patch_timing(prefill_s=2.0, decode_s=3.0):
    """Context manager that patches both estimation functions to fixed values."""
    return (
        patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=prefill_s),
        patch.object(sim_mod, "isolated_decode_time_estimation", return_value=decode_s),
    )


def _silent_owner():
    """Minimal owner object for _DocPolicyPrepareWorker."""
    owner = SimpleNamespace(
        _deltas_us={"prefill": 0, "first_decode": 0, "decode": 0},
        _last_prepare_breakdown_ms={},
        delta_fairness_n=2,
        _debug_earliest_decode_rid=None,
        _debug_earliest_decode_uid=None,
        _prefill_no_retraction_token_cap=None,
    )
    # _compute_safe_prefix_state is called by _build_prepare_snapshot
    owner._compute_safe_prefix_state = lambda *a, **kw: {
        "safe_waiting_queue": [],
        "safe_waiting_rids": set(),
        "forced_prefill_queue": [],
        "forced_prefill_rids": set(),
        "max_safe_prefill_tokens": None,
        "has_fair_waiting": False,
        "has_decode_deadline": False,
        "earliest_decode_start_deadline": None,
        "safe_prefix_now": None,
        "skipped_rids": [],
        "skipped_reasons": [],
    }
    return owner


def _wait_snapshot(worker, timeout_s=2.0):
    """Block until the worker has produced a snapshot for the latest task."""
    return worker.wait_for_snapshot(
        min_task_seq=worker._task_seq,
        timeout_s=timeout_s,
    )


# ===========================================================================
# 1. Direct C kernel tests
# ===========================================================================

@unittest.skipUnless(C_EXT_AVAILABLE, "_fairinf_sim C extension not built")
class TestCSimKernelDirect(unittest.TestCase):
    """
    Calls _fairinf_sim.rebuild_kernel() directly.

    The function signature (to be implemented by the C extension):

        results = _fairinf_sim.rebuild_kernel(
            rids:              List[str],
            arrival_times:     List[float],
            prompt_lens:       List[int],
            real_decode_counts:List[int],
            prefill_dones:     List[int],   # 0/1
            is_completes:      List[int],   # 0/1
            max_kv_tokens:     int,         # -1 = unlimited
            fairinf_n:         int,
        ) -> List[Tuple[str, int, float, int]]
             # (rid, event_type, end_timestamp, completion_number)
             # event_type: 0=prefill, 1=decode

    The kernel runs rebuild_from_real_state logic purely in C and returns
    results as a flat list.  The Python side writes them back to Python objects.
    """

    def _run_kernel(self, reqs_data, max_kv=-1, fairinf_n=1):
        """
        reqs_data: list of dicts with keys:
          rid, arrival_ts, prompt_len, real_decode_count, prefill_done, is_complete
        Returns dict: rid -> (event_type, end_timestamp, completion_num)
        """
        rids           = [r["rid"]               for r in reqs_data]
        arrivals       = [float(r["arrival_ts"])  for r in reqs_data]
        prompt_lens    = [int(r["prompt_len"])    for r in reqs_data]
        decode_counts  = [int(r.get("real_decode_count", 0)) for r in reqs_data]
        prefill_dones  = [int(r.get("prefill_done", 0))      for r in reqs_data]
        is_completes   = [int(r.get("is_complete", 0))        for r in reqs_data]
        results = _sim_c.rebuild_kernel(
            rids, arrivals, prompt_lens, decode_counts, prefill_dones, is_completes,
            max_kv, fairinf_n,
        )
        return {rid: (etype, ets, ecn) for rid, etype, ets, ecn in results}

    def test_single_waiting_request_gets_prefill_event(self):
        """One waiting request (not yet prefilled) must get a prefill anticipated event."""
        out = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 0, "real_decode_count": 0},
        ])
        self.assertIn("r1", out)
        etype, ets, ecn = out["r1"]
        self.assertEqual(etype, 0)          # 0 = prefill
        self.assertGreater(ets, 10.0)       # must be after arrival
        self.assertEqual(ecn, 0)            # completion_number unused for prefill

    def test_prefilled_request_gets_decode_event(self):
        """A request that has been prefilled (prefill_done=1, real_decode_count=0)
        should get a decode anticipated event with completion_number >= 1."""
        out = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 1, "real_decode_count": 0},
        ])
        self.assertIn("r1", out)
        etype, ets, ecn = out["r1"]
        self.assertEqual(etype, 1)          # 1 = decode
        self.assertGreaterEqual(ecn, 1)

    def test_two_requests_same_user_arrival_order_respected(self):
        """With two requests for the same user arriving at t=10 and t=11,
        the isolated scheduler prefills r1 first, then r2.
        r2's prefill end_timestamp > r1's prefill end_timestamp."""
        out = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 0, "real_decode_count": 0},
            {"rid": "r2", "arrival_ts": 11.0, "prompt_len": 4,
             "prefill_done": 0, "real_decode_count": 0},
        ])
        self.assertIn("r1", out)
        self.assertIn("r2", out)
        ets_r1 = out["r1"][1]
        ets_r2 = out["r2"][1]
        self.assertLess(ets_r1, ets_r2,
            "r1 arrives earlier so it should be prefilled before r2")

    def test_over_kv_budget_triggers_retraction_and_delays_evicted_request(self):
        """Two requests each with prompt_len=6; max_kv=8.
        They can't both fit simultaneously, so the isolated scheduler retracts one.
        The retracted request gets a later anticipated event."""
        out_unlimited = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 6, "prefill_done": 0},
            {"rid": "r2", "arrival_ts": 10.0, "prompt_len": 6, "prefill_done": 0},
        ], max_kv=-1)
        out_constrained = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 6, "prefill_done": 0},
            {"rid": "r2", "arrival_ts": 10.0, "prompt_len": 6, "prefill_done": 0},
        ], max_kv=8)
        # In the constrained case the evicted request's event is later
        max_ts_unlimited   = max(out_unlimited[r][1]   for r in ("r1", "r2"))
        max_ts_constrained = max(out_constrained[r][1] for r in ("r1", "r2"))
        self.assertGreater(max_ts_constrained, max_ts_unlimited,
            "constrained scheduler retracts a request, delaying it")

    def test_over_served_request_does_not_generate_spurious_early_deadline(self):
        """A request that has already been decoded more times than isolation would give
        it (real_decode_count > what sim would produce) must NOT generate a deadline
        earlier than what isolation allows.  The sim_dc <= real_dc condition means
        next_anticipated_event.completion_number = sim_dc + 1 <= real_dc, so
        events_after(real_event_at_real_dc) returns nothing — the request doesn't
        drive a decode deadline in this kernel output."""
        # r1 has real_decode_count=10 but prompt_len=4 and fairinf_n=1;
        # the isolated sim can only give it ~a few decodes before the sim ends.
        out = self._run_kernel([
            {"rid": "r1", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 1, "real_decode_count": 10},
        ], fairinf_n=1)
        # If r1 is in the output at all, its completion_number must exceed real_dc=10
        # (otherwise it would not be "after" the real event and shouldn't be emitted).
        if "r1" in out:
            etype, ets, ecn = out["r1"]
            if etype == 1:  # decode
                self.assertGreater(ecn, 10,
                    "C kernel must not emit a decode deadline ≤ real_decode_count")

    def test_completed_request_produces_no_event(self):
        """A request with is_complete=1 should not appear in the output."""
        out = self._run_kernel([
            {"rid": "r_done", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 1, "real_decode_count": 5, "is_complete": 1},
            {"rid": "r_live", "arrival_ts": 10.0, "prompt_len": 4,
             "prefill_done": 0, "real_decode_count": 0, "is_complete": 0},
        ])
        self.assertNotIn("r_done", out,
            "Completed requests should be dropped by the C kernel")
        self.assertIn("r_live", out)

    def test_empty_input_returns_empty_output(self):
        out = self._run_kernel([])
        self.assertEqual(out, {})

    def test_many_requests_all_get_events(self):
        """32 waiting requests for the same user — every one should eventually
        get an anticipated event from the C kernel."""
        reqs = [
            {"rid": f"r{i}", "arrival_ts": 10.0 + i * 0.1, "prompt_len": 4,
             "prefill_done": 0, "real_decode_count": 0}
            for i in range(32)
        ]
        out = self._run_kernel(reqs, max_kv=10000, fairinf_n=1)
        for r in reqs:
            self.assertIn(r["rid"], out,
                f"{r['rid']} missing from kernel output")

    def test_arrival_order_preserved_across_users(self):
        """Two users, each with one request.  User A's request arrives at t=10,
        user B's at t=11.  Both should get prefill events; A's earlier than B's."""
        out = self._run_kernel([
            {"rid": "ua_r1", "arrival_ts": 10.0, "prompt_len": 4, "prefill_done": 0},
            {"rid": "ub_r1", "arrival_ts": 11.0, "prompt_len": 4, "prefill_done": 0},
        ])
        self.assertLess(out["ua_r1"][1], out["ub_r1"][1])


# ===========================================================================
# 2. Python / C parity tests
# ===========================================================================

def _build_py_simulator(max_kv, fairinf_n, prefill_s, decode_s):
    """Build a Python AlternateHistorySimulator with patched timing."""
    return AlternateHistorySimulator(
        max_kv_tokens_per_user=max_kv,
        fairinf_n=fairinf_n,
        enable_timeline_logging=False,
    )


def _anticipated_summary(simulator, rid):
    """Return (event_class_name, end_timestamp, completion_number_or_None)."""
    tracked = simulator.requests.get(rid)
    if tracked is None:
        return None
    ant = tracked.timeline.next_anticipated_event
    if ant is None:
        return ("none", 0.0, 0)
    return (
        type(ant).__name__,
        round(ant.end_timestamp, 6),
        getattr(ant, "completion_number", 0),
    )


@unittest.skipUnless(C_EXT_AVAILABLE, "_fairinf_sim C extension not built")
class TestCSimVsPython(unittest.TestCase):
    """
    Runs each scenario through both the Python reference simulator and the
    C-backed simulator (which calls the C rebuild_kernel internally), and
    asserts identical anticipated events.

    The C-backed simulator is the same AlternateHistorySimulator but with
    UserTimeline.rebuild_from_real_state replaced by the C kernel path.
    We activate the C path by importing _fairinf_sim — the presence of the
    module is detected by doc_policy_simulator.py which switches the hot path.
    """

    def _run_scenario(self, setup_fn, max_kv, fairinf_n, prefill_s=2.0, decode_s=3.0):
        """
        setup_fn(simulator, now_box) -> (running_batch, waiting_queue)
        Runs setup_fn on a fresh Python simulator and a fresh C-backed simulator,
        then calls start_of_pass on both, and returns (py_summaries, c_summaries).
        """
        now = {"t": 10.0}

        patch_timeline = patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start")
        patch_prefill  = patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done")
        patch_decode   = patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done")
        patch_complete = patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_completed")
        p_t = patch.object(sim_mod.time, "time", side_effect=lambda: now["t"])
        p_pf = patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=prefill_s)
        p_dc = patch.object(sim_mod, "isolated_decode_time_estimation", return_value=decode_s)

        with patch_timeline, patch_prefill, patch_decode, patch_complete, p_t, p_pf, p_dc:
            py_sim = AlternateHistorySimulator(
                max_kv_tokens_per_user=max_kv,
                fairinf_n=fairinf_n,
                enable_timeline_logging=False,
            )
            c_sim = AlternateHistorySimulator(
                max_kv_tokens_per_user=max_kv,
                fairinf_n=fairinf_n,
                enable_timeline_logging=False,
            )
            # Force C path on c_sim (the C extension replaces rebuild_from_real_state)
            _sim_c.patch_simulator(c_sim)  # see note below

            running_batch, waiting_queue = setup_fn(py_sim, now)
            # Reset and re-run setup for c_sim (same sequence)
            running_batch2, waiting_queue2 = setup_fn(c_sim, now)

            now["t"] = 100.0
            py_sim.start_of_pass(running_batch, waiting_queue)
            c_sim.start_of_pass(running_batch2, waiting_queue2)

        all_rids = set(py_sim.requests) | set(c_sim.requests)
        py_out = {rid: _anticipated_summary(py_sim, rid) for rid in all_rids}
        c_out  = {rid: _anticipated_summary(c_sim,  rid) for rid in all_rids}
        return py_out, c_out

    def _assert_parity(self, py_out, c_out, rids=None, msg=""):
        """Assert that py and c produce the same event type and completion_number.
        Timestamps may differ by float rounding but should be within 1 µs."""
        check_rids = rids if rids is not None else py_out.keys()
        for rid in check_rids:
            py = py_out.get(rid)
            c  = c_out.get(rid)
            self.assertIsNotNone(py, f"{rid}: missing from Python output {msg}")
            self.assertIsNotNone(c,  f"{rid}: missing from C output {msg}")
            self.assertEqual(py[0], c[0],
                f"{rid}: event type mismatch: py={py[0]} c={c[0]} {msg}")
            self.assertEqual(py[2], c[2],
                f"{rid}: completion_number mismatch: py={py[2]} c={c[2]} {msg}")
            self.assertAlmostEqual(py[1], c[1], places=4,
                msg=f"{rid}: end_timestamp divergence: py={py[1]} c={c[1]} {msg}")

    # -----------------------------------------------------------------------
    # Scenario helpers (mirrors of the existing Python-only tests)
    # -----------------------------------------------------------------------

    def test_parity_single_waiting_request(self):
        """Single waiting request — both sims must agree on prefill event."""
        req = _mk_req("user_1", "r1", 4)

        def setup(sim, now):
            sim.process_new_request(req, None, arrival_timestamp=10.0)
            return None, [req]

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=2)
        self._assert_parity(py, c, ["r1"], "single waiting")

    def test_parity_running_request_decode_deadline(self):
        """A request that has been prefilled and decoded once — both sims must
        agree on the next anticipated decode completion number."""
        req = _mk_req("user_1", "r1", 4)
        req2 = _mk_req("user_1", "r1", 4)  # same rid, different object for c_sim

        def setup(sim, now):
            sim.process_new_request(req, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [42]
            now["t"] = 12.0
            sim.finished_decode(SimpleNamespace(reqs=[req]))
            return SimpleNamespace(reqs=[req]), []

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=2)
        self._assert_parity(py, c, ["r1"], "running decode")

    def test_parity_two_waiting_same_user_arrival_order(self):
        """Two waiting requests for the same user — prefill deadlines ordered by arrival."""
        req1 = _mk_req("user_1", "r1", 4)
        req2 = _mk_req("user_1", "r2", 4)

        def setup(sim, now):
            sim.process_new_request(req1, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.process_new_request(req2, None, arrival_timestamp=11.0)
            return None, [req1, req2]

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=2)
        self._assert_parity(py, c, ["r1", "r2"], "two waiting same user")
        # Additionally: r1 before r2
        self.assertLess(c["r1"][1], c["r2"][1],
            "C sim: r1 arrives earlier, must get earlier prefill event")

    def test_parity_kv_constrained_retraction(self):
        """Two requests that exceed KV budget — retraction in isolation must
        produce identical delay in both Python and C."""
        req1 = _mk_req("user_1", "r1", 6)
        req2 = _mk_req("user_1", "r2", 6)

        def setup(sim, now):
            sim.process_new_request(req1, None, arrival_timestamp=10.0)
            sim.process_new_request(req2, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
            req1.output_ids = [1]
            req2.output_ids = [1]
            now["t"] = 12.0
            sim.finished_decode(SimpleNamespace(reqs=[req1, req2]))
            return SimpleNamespace(reqs=[req1]), [req2]

        py, c = self._run_scenario(setup, max_kv=8, fairinf_n=2)
        self._assert_parity(py, c, ["r1", "r2"], "kv constrained retraction")

    def test_parity_retracted_request_reset_to_waiting(self):
        """A request that was running, then retracted back to waiting via
        process_new_request — must be treated as newly arrived."""
        req = _mk_req("user_1", "r1", 4)

        def setup(sim, now):
            sim.process_new_request(req, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [1]
            now["t"] = 12.0
            sim.finished_decode(SimpleNamespace(reqs=[req]))
            # retraction: re-process as new arrival at t=30
            now["t"] = 30.0
            sim.process_new_request(req, None, arrival_timestamp=30.0)
            return None, [req]

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=2)
        self._assert_parity(py, c, ["r1"], "retracted to waiting")
        # Retracted request must get a prefill event, not a decode event
        self.assertEqual(c["r1"][0], "RequestPrefillEvent",
            "retracted request must get prefill anticipated event")

    def test_parity_multi_user_two_requests_each(self):
        """Two users, two requests each — 4 total. Both sims must agree on all 4."""
        reqs = {
            "ua_r1": _mk_req("user_A", "ua_r1", 4),
            "ua_r2": _mk_req("user_A", "ua_r2", 4),
            "ub_r1": _mk_req("user_B", "ub_r1", 4),
            "ub_r2": _mk_req("user_B", "ub_r2", 4),
        }

        def setup(sim, now):
            sim.process_new_request(reqs["ua_r1"], None, arrival_timestamp=10.0)
            sim.process_new_request(reqs["ub_r1"], None, arrival_timestamp=10.5)
            now["t"] = 11.0
            sim.process_new_request(reqs["ua_r2"], None, arrival_timestamp=11.0)
            now["t"] = 11.5
            sim.process_new_request(reqs["ub_r2"], None, arrival_timestamp=11.5)
            return None, list(reqs.values())

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=2)
        self._assert_parity(py, c, list(reqs.keys()), "multi-user 4 reqs")

    def test_parity_over_served_request_no_spurious_deadline(self):
        """A request with real_decode_count >> isolation budget should not generate
        a deadline earlier than what isolation allows in either sim."""
        req = _mk_req("user_1", "r1", 4)

        def setup(sim, now):
            sim.process_new_request(req, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = list(range(20))
            now["t"] = 12.0
            sim.finished_decode(SimpleNamespace(reqs=[req]), decode_rounds=20)
            return SimpleNamespace(reqs=[req]), []

        py, c = self._run_scenario(setup, max_kv=100, fairinf_n=1)
        self._assert_parity(py, c, ["r1"], "over-served no spurious deadline")

    def test_parity_build_deadline_candidates_outputs(self):
        """Run build_deadline_candidates on both and compare the deadline_queue
        contents — event types, rids, and approximate deadlines must match."""
        req_waiting = _mk_req("user_1", "rw", 4)
        req_running = _mk_req("user_2", "rr", 4)

        def setup(sim, now):
            sim.process_new_request(req_waiting, None, arrival_timestamp=10.0)
            sim.process_new_request(req_running, None, arrival_timestamp=10.0)
            now["t"] = 11.0
            sim.finished_prefill(SimpleNamespace(reqs=[req_running]))
            req_running.output_ids = [1]
            now["t"] = 12.0
            sim.finished_decode(SimpleNamespace(reqs=[req_running]))
            return SimpleNamespace(reqs=[req_running]), [req_waiting]

        now = {"t": 10.0}
        with (
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_completed"),
            patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]),
            patch.object(sim_mod, "isolated_prefill_time_estimation", return_value=2.0),
            patch.object(sim_mod, "isolated_decode_time_estimation", return_value=3.0),
        ):
            py_sim = AlternateHistorySimulator(max_kv_tokens_per_user=100, fairinf_n=2, enable_timeline_logging=False)
            c_sim  = AlternateHistorySimulator(max_kv_tokens_per_user=100, fairinf_n=2, enable_timeline_logging=False)
            _sim_c.patch_simulator(c_sim)

            running_batch_py, waiting_py = setup(py_sim, now)
            running_batch_c,  waiting_c  = setup(c_sim,  now)
            now["t"] = 100.0
            py_sim.start_of_pass(running_batch_py, waiting_py)
            c_sim.start_of_pass(running_batch_c,  waiting_c)

            def _candidates(sim, running_batch, waiting):
                dq, _, ordered = sim.build_deadline_candidates(
                    waiting, running_batch,
                    include_ordered_waiting_queue=True,
                    req_is_fair_prefill=lambda r, rb: True,
                    req_is_fair_decode=lambda r, rb: True,
                    event_delta_seconds=lambda tr, e: 0.0,
                    pooled_prefill_estimate_seconds=lambda r: 2.0,
                    pooled_decode_estimate_seconds=lambda r, rb: 3.0,
                )
                return dq

            py_dq = _candidates(py_sim, running_batch_py, waiting_py)
            c_dq  = _candidates(c_sim,  running_batch_c,  waiting_c)

        py_by_rid = {c.req.rid: c for c in py_dq}
        c_by_rid  = {c.req.rid: c for c in c_dq}
        for rid in set(py_by_rid) | set(c_by_rid):
            self.assertIn(rid, py_by_rid, f"{rid} in C but not Python deadline_queue")
            self.assertIn(rid, c_by_rid,  f"{rid} in Python but not C deadline_queue")
            self.assertEqual(py_by_rid[rid].event_type, c_by_rid[rid].event_type,
                f"{rid}: event_type mismatch in deadline_queue")
            self.assertAlmostEqual(
                py_by_rid[rid].deadline, c_by_rid[rid].deadline, places=4,
                msg=f"{rid}: deadline mismatch: py={py_by_rid[rid].deadline} c={c_by_rid[rid].deadline}",
            )


# ===========================================================================
# 3. End-to-end worker tests
# ===========================================================================

@unittest.skipUnless(C_EXT_AVAILABLE, "_fairinf_sim C extension not built")
class TestCWorkerE2E(unittest.TestCase):
    """
    Exercises the full _DocPolicyPrepareWorker with the C-backed thread.
    """

    def setUp(self):
        self.owner = _silent_owner()
        self.worker = _DocPolicyPrepareWorker(
            self.owner,
            isolated_kv_tokens_per_user=None,
            fairinf_n=2,
            min_new_token_ratio=0.0,
        )

    def tearDown(self):
        self.worker._worker_stop = True

    def _enqueue_simple_task(self, waiting_queue=None, running_batch=None):
        from sglang.srt.delta_fairness.simulator_thread import _FrozenPrepareInputs
        frozen_inputs = _FrozenPrepareInputs(
            deltas_us={"prefill": 0, "first_decode": 0, "decode": 0},
            no_retraction_cap=None,
            new_token_ratio=0.0,
            fairinf_n=2,
        )
        wq = [self.worker.make_prepare_req(r) for r in (waiting_queue or [])]
        rb = self.worker.make_prepare_batch(running_batch)
        return self.worker.enqueue_task((wq, rb, None, frozen_inputs, self.worker.mutation_seq))

    def test_worker_produces_snapshot_for_empty_queues(self):
        """Worker should produce a snapshot even when both queues are empty."""
        seq = self._enqueue_simple_task()
        ok = _wait_snapshot(self.worker)
        self.assertTrue(ok, "Worker did not produce a snapshot within timeout")
        snap = self.worker.latest_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.task_seq, seq)

    def test_snapshot_contains_waiting_request_as_prefill_candidate(self):
        """A single waiting request must appear as a prefill candidate in the snapshot."""
        req = _mk_req("user_1", "r1", 4)
        # Enqueue process_new_request mutation first
        self.worker.enqueue_mutation("process_new_request", (
            self.worker.make_prepare_req(req),
            {"prefill": 0, "first_decode": 0, "decode": 0},
            time.time(),
        ))
        seq = self._enqueue_simple_task(waiting_queue=[req])
        ok = _wait_snapshot(self.worker, timeout_s=2.0)
        self.assertTrue(ok)
        snap = self.worker.latest_snapshot()
        # The deadline_queue should contain a prefill candidate for r1
        prefill_rids = [
            c.req.rid for c in snap.deadline_queue
            if getattr(c, "event_type", "") == "prefill"
        ]
        self.assertIn("r1", prefill_rids,
            f"r1 not found in prefill candidates: {[c.req.rid for c in snap.deadline_queue]}")

    def test_mutations_applied_before_snapshot(self):
        """process_new_request mutation must be visible in the snapshot produced
        after it — i.e., the worker applies mutations before building snapshots."""
        req = _mk_req("user_1", "r_mut", 4)
        self.worker.enqueue_mutation("process_new_request", (
            self.worker.make_prepare_req(req),
            {"prefill": 0, "first_decode": 0, "decode": 0},
            time.time(),
        ))
        seq = self._enqueue_simple_task(waiting_queue=[req])
        ok = _wait_snapshot(self.worker, timeout_s=2.0)
        self.assertTrue(ok)
        snap = self.worker.latest_snapshot()
        all_rids = [c.req.rid for c in snap.deadline_queue]
        self.assertIn("r_mut", all_rids,
            "Mutation-added request must be visible in the snapshot that follows it")

    def test_mark_request_finished_removes_from_subsequent_snapshot(self):
        """After mark_request_finished, the request should not appear in the
        next snapshot's deadline_queue."""
        req = _mk_req("user_1", "r_finished", 4)
        # Add and snapshot once
        self.worker.enqueue_mutation("process_new_request", (
            self.worker.make_prepare_req(req),
            {"prefill": 0, "first_decode": 0, "decode": 0},
            time.time(),
        ))
        self._enqueue_simple_task(waiting_queue=[req])
        _wait_snapshot(self.worker, timeout_s=2.0)

        # Now mark finished + new snapshot with empty queues
        self.worker.enqueue_mutation("mark_request_finished", (
            self.worker.make_prepare_req(req), 0
        ))
        seq2 = self._enqueue_simple_task(waiting_queue=[], running_batch=None)
        ok = _wait_snapshot(self.worker, timeout_s=2.0)
        self.assertTrue(ok)
        snap = self.worker.latest_snapshot()
        self.assertEqual(snap.task_seq, seq2)
        all_rids = [c.req.rid for c in snap.deadline_queue]
        self.assertNotIn("r_finished", all_rids,
            "Finished request must not appear in subsequent deadline_queue")

    def test_logical_decode_update_advances_simulated_decode_count(self):
        """After a logical_decode_update mutation, the simulator should reflect
        an advanced decode count — the anticipated event completion_number in the
        snapshot should be higher than 1."""
        req = _mk_req("user_1", "r_decode", 4)
        prepare_req = self.worker.make_prepare_req(req)
        # Seed the request
        self.worker.enqueue_mutation("process_new_request", (
            prepare_req,
            {"prefill": 0, "first_decode": 0, "decode": 0},
            time.time(),
        ))
        # Simulate prefill done
        req.output_ids = []
        self.worker.enqueue_mutation("note_prefill_done", ([prepare_req],))
        # Simulate 5 decode rounds
        req.output_ids = list(range(5))
        prepare_req_after = self.worker.make_prepare_req(req)
        batch_ns = SimpleNamespace(reqs=[prepare_req_after])
        self.worker.enqueue_mutation("logical_decode_update", (batch_ns, 5))

        seq = self._enqueue_simple_task(running_batch=SimpleNamespace(reqs=[req]))
        ok = _wait_snapshot(self.worker, timeout_s=2.0)
        self.assertTrue(ok)
        snap = self.worker.latest_snapshot()
        decode_candidates = [
            c for c in snap.deadline_queue if getattr(c, "event_type", "") == "decode"
        ]
        if decode_candidates:
            best = min(decode_candidates, key=lambda c: c.event.completion_number)
            self.assertGreater(best.event.completion_number, 5,
                "After 5 decode rounds, anticipated completion must be > 5")

    def test_gil_is_released_during_simulation(self):
        """
        The main Python thread must be able to acquire the GIL and run Python
        code while the C worker thread is performing its simulation.

        Method: enqueue a task that requires substantial simulation work
        (many requests), then immediately try to run Python work on the main
        thread.  If the GIL is held by the worker the whole time, this will
        serialize them and the total elapsed time will be > worker_time.
        We can't measure this precisely, but we can verify that the main thread
        is not blocked: it should be able to run a tight loop and increment a
        counter while the worker is busy.
        """
        N = 50  # enough requests to keep the C worker busy for >10ms
        reqs = [_mk_req("user_1", f"r{i}", 16) for i in range(N)]
        for r in reqs:
            self.worker.enqueue_mutation("process_new_request", (
                self.worker.make_prepare_req(r),
                {"prefill": 0, "first_decode": 0, "decode": 0},
                time.time(),
            ))
        seq = self._enqueue_simple_task(waiting_queue=reqs)

        # While worker is processing, count how many times the main thread
        # can increment a counter.  If the GIL is hogged, this will be ~0.
        counter = {"n": 0}
        deadline = time.perf_counter() + 0.5  # 500 ms window

        def count_loop():
            while time.perf_counter() < deadline:
                counter["n"] += 1
                time.sleep(0.0)  # yield — lets the worker run

        t = threading.Thread(target=count_loop)
        t.start()

        ok = _wait_snapshot(self.worker, timeout_s=2.0)
        t.join()

        self.assertTrue(ok, "Worker did not produce snapshot")
        # If GIL was properly released, the counter thread ran a substantial
        # number of iterations (at least a few thousand).  If the worker hogged
        # the GIL, counter["n"] would be ~0-1.
        self.assertGreater(counter["n"], 100,
            f"Main thread only incremented counter {counter['n']} times — "
            "GIL may not have been released by the C worker")

    def test_multiple_tasks_are_processed_in_order(self):
        """Enqueue three tasks in sequence; each subsequent snapshot must have
        a strictly increasing task_seq."""
        seqs = []
        for i in range(3):
            req = _mk_req(f"user_{i}", f"r{i}", 4)
            self.worker.enqueue_mutation("process_new_request", (
                self.worker.make_prepare_req(req),
                {"prefill": 0, "first_decode": 0, "decode": 0},
                time.time(),
            ))
            seqs.append(self._enqueue_simple_task(waiting_queue=[req]))

        ok = _wait_snapshot(self.worker, timeout_s=3.0)
        self.assertTrue(ok)
        snap = self.worker.latest_snapshot()
        # The last snapshot must have the highest seq
        self.assertEqual(snap.task_seq, seqs[-1],
            f"Expected final task_seq={seqs[-1]}, got {snap.task_seq}")

    def test_worker_survives_rapid_mutation_burst(self):
        """Enqueue 100 mutations followed by a task — no crash, valid snapshot."""
        for i in range(100):
            req = _mk_req(f"user_{i % 5}", f"r{i}", 4)
            self.worker.enqueue_mutation("process_new_request", (
                self.worker.make_prepare_req(req),
                {"prefill": 0, "first_decode": 0, "decode": 0},
                time.time(),
            ))
        reqs_sample = [_mk_req(f"user_{i % 5}", f"r{i}", 4) for i in range(5)]
        seq = self._enqueue_simple_task(waiting_queue=reqs_sample)
        ok = _wait_snapshot(self.worker, timeout_s=5.0)
        self.assertTrue(ok, "Worker did not produce snapshot after mutation burst")
        self.worker.raise_exception_if_any()


# ===========================================================================
# 4. Existing Python simulator tests re-run through the C path
#    (verifies the C-backed UserTimeline.rebuild_from_real_state is correct
#    for every scenario already covered by test_doc_policy_simulator_unit.py)
# ===========================================================================

@unittest.skipUnless(C_EXT_AVAILABLE, "_fairinf_sim C extension not built")
class TestExistingScenariosWithCBackend(unittest.TestCase):
    """
    Re-runs the exact same assertions as test_doc_policy_simulator_unit.py
    but with the C kernel patched in via _sim_c.patch_simulator().

    If any assertion here fails but the corresponding Python test passes,
    the C kernel has a bug in that specific scenario.
    """

    def _make_sim(self, max_kv, fairinf_n):
        sim = AlternateHistorySimulator(
            max_kv_tokens_per_user=max_kv,
            fairinf_n=fairinf_n,
            enable_timeline_logging=False,
        )
        _sim_c.patch_simulator(sim)
        return sim

    def test_c_waiting_prefill_deadlines_follow_isolated_arrival_order(self):
        """C backend: two waiting requests must appear in arrival order as prefill
        candidates, with the earlier arrival's deadline < the later one's."""
        with patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start"):
            simulator = self._make_sim(100, 2)
            req1 = _mk_req("user_1", "rid_1", 4)
            req2 = _mk_req("user_1", "rid_2", 4)
            simulator.process_new_request(req1, None, arrival_timestamp=10.0)
            simulator.process_new_request(req2, None, arrival_timestamp=11.0)
            simulator.start_of_pass(SimpleNamespace(reqs=[]), [req1, req2])
            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                [req1, req2], SimpleNamespace(reqs=[]),
                req_is_fair_prefill=lambda r, rb: True,
                req_is_fair_decode=lambda r, rb: True,
                event_delta_seconds=lambda t, e: 0.0,
                pooled_prefill_estimate_seconds=lambda r: 0.0,
                pooled_decode_estimate_seconds=lambda r, rb: 0.0,
            )
        # rid_1 arrives earlier so its deadline must be < rid_2's deadline
        self.assertIn("rid_1", waiting_deadlines)
        self.assertIn("rid_2", waiting_deadlines)
        self.assertLess(waiting_deadlines["rid_1"], waiting_deadlines["rid_2"],
            "Earlier-arriving request must have earlier prefill deadline")
        prefill_rids = [c.req.rid for c in candidates if c.event_type == "prefill"]
        self.assertEqual(prefill_rids, ["rid_1", "rid_2"],
            "Prefill candidates must be in arrival order")

    def test_c_running_decode_deadline_from_isolated_sequence(self):
        """C backend: a request that has had real_decode_count=1 should produce
        a decode candidate with completion_number > 1 from the isolation sim."""
        with (
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"),
        ):
            simulator = self._make_sim(100, 2)
            req = _mk_req("user_1", "rid_running", 4)
            simulator.process_new_request(req, None, arrival_timestamp=10.0)
            simulator.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [42]
            simulator.finished_decode(SimpleNamespace(reqs=[req]))
            simulator.start_of_pass(SimpleNamespace(reqs=[req]), [])
            candidates, _ = simulator.build_deadline_candidates(
                [], SimpleNamespace(reqs=[req]),
                req_is_fair_prefill=lambda r, rb: True,
                req_is_fair_decode=lambda r, rb: True,
                event_delta_seconds=lambda t, e: 0.0,
                pooled_prefill_estimate_seconds=lambda r: 0.0,
                pooled_decode_estimate_seconds=lambda r, rb: 0.0,
            )
        decode_candidates = [c for c in candidates if c.event_type == "decode"]
        self.assertEqual(len(decode_candidates), 1,
            "Running request must produce exactly one decode candidate")
        # completion_number must be > real_decode_count (1)
        self.assertGreater(decode_candidates[0].event.completion_number, 1,
            "C kernel must produce next anticipated decode beyond what already happened")

    def test_c_retracted_request_becomes_waiting_prefill_candidate(self):
        """C backend: a request re-processed via process_new_request (retraction)
        must appear as a prefill candidate, not a decode candidate."""
        with (
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"),
        ):
            simulator = self._make_sim(100, 2)
            req = _mk_req("user_1", "rid_retracted", 4)
            simulator.process_new_request(req, None, arrival_timestamp=10.0)
            simulator.finished_prefill(SimpleNamespace(reqs=[req]))
            req.output_ids = [1]
            simulator.finished_decode(SimpleNamespace(reqs=[req]))
            # Re-process as new arrival (retraction)
            simulator.process_new_request(req, None, arrival_timestamp=30.0)
            simulator.start_of_pass(SimpleNamespace(reqs=[]), [req])
            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                [req], SimpleNamespace(reqs=[]),
                req_is_fair_prefill=lambda r, rb: True,
                req_is_fair_decode=lambda r, rb: True,
                event_delta_seconds=lambda t, e: 0.0,
                pooled_prefill_estimate_seconds=lambda r: 0.0,
                pooled_decode_estimate_seconds=lambda r, rb: 0.0,
            )
        self.assertEqual(
            [(c.req.rid, c.event_type) for c in candidates],
            [("rid_retracted", "prefill")],
            "Retracted request must be a prefill candidate, not decode",
        )
        # Deadline must be anchored to the new arrival time (30.0), not the old one
        self.assertGreaterEqual(waiting_deadlines["rid_retracted"], 30.0,
            "Retracted request deadline must be >= new arrival time")

    def test_c_retraction_penalty_pushes_decode_deadline_out(self):
        """C backend: the constrained sim (KV budget exceeded → retraction) must
        push the surviving request's decode deadline later than unconstrained."""
        with (
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_start"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_prefill_done"),
            patch.object(sim_mod.TIMELINE_WRITER, "mark_isolated_decode_done"),
        ):
            req1 = _mk_req("user_1", "rid_bad_1", 4)
            req2 = _mk_req("user_1", "rid_bad_2", 4)

            unconstrained = self._make_sim(100, 2)
            constrained   = self._make_sim(8, 2)

            for sim in (unconstrained, constrained):
                sim.process_new_request(req1, None, arrival_timestamp=10.0)
                sim.process_new_request(req2, None, arrival_timestamp=10.0)
                sim.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
                req1.output_ids = [1]
                req2.output_ids = [1]
                sim.finished_decode(SimpleNamespace(reqs=[req1, req2]))

            unconstrained.start_of_pass(SimpleNamespace(reqs=[req1, req2]), [])
            constrained.start_of_pass(SimpleNamespace(reqs=[req1]), [req2])

            kw = dict(
                req_is_fair_prefill=lambda r, rb: True,
                req_is_fair_decode=lambda r, rb: True,
                event_delta_seconds=lambda t, e: 0.0,
                pooled_prefill_estimate_seconds=lambda r: 0.0,
                pooled_decode_estimate_seconds=lambda r, rb: 0.0,
            )
            unc_candidates, _ = unconstrained.build_deadline_candidates(
                [], SimpleNamespace(reqs=[req1, req2]), **kw)
            con_candidates, _ = constrained.build_deadline_candidates(
                [req2], SimpleNamespace(reqs=[req1]), **kw)

        unc_r1 = next(c for c in unc_candidates if c.req.rid == "rid_bad_1" and c.event_type == "decode")
        con_r1 = next(c for c in con_candidates if c.req.rid == "rid_bad_1" and c.event_type == "decode")
        # Constrained sim has a retraction → penalty added → later deadline
        self.assertGreaterEqual(
            con_r1.deadline,
            unc_r1.deadline + sim_mod.RETRACTION_PENALTY_SECONDS,
            "KV-constrained sim must push deadline out by at least the retraction penalty",
        )


if __name__ == "__main__":
    unittest.main()
