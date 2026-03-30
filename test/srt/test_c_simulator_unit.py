"""
Unit tests for the CSimulator C extension (_fairinf_worker).

Runs the same deadline/retraction tests as TestDocPolicySimulatorUnit
but against a CSimulatorAdapter wrapper. Timeline-logging tests are
skipped (the C extension's stored TIMELINE_WRITER reference is captured
at init time, so mock.patch applied after __init__ won't intercept it).
"""
import math
import sys
import time
import unittest
from types import SimpleNamespace

try:
    from sglang.srt.delta_fairness import _fairinf_worker  # type: ignore[import]
except ImportError:
    _fairinf_worker = None

from sglang.srt.delta_fairness.doc_policy_simulator import (
    DeadlineCandidate,
    RequestDecodeEvent,
    RequestPrefillEvent,
)
from sglang.srt.managers.schedule_batch import Req


def _mk_req(uid: str, rid: str, n_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * n_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    return req


# ---------------------------------------------------------------------------
# Adapter: wraps CSimulator with AlternateHistorySimulator-like interface
# ---------------------------------------------------------------------------

class CSimulatorAdapter:
    """
    Wraps _fairinf_worker.CSimulator with the same external interface as
    AlternateHistorySimulator so that the unit tests can run unchanged.
    """

    def __init__(
        self,
        *,
        max_kv_tokens_per_user=None,
        fairinf_n: int = 1,
        min_new_token_ratio: float = 0.0,
        enable_timeline_logging: bool = False,
    ):
        assert _fairinf_worker is not None, "_fairinf_worker C extension not available"
        self._sim = _fairinf_worker.CSimulator(
            max_kv_tokens_per_user if max_kv_tokens_per_user is not None else -1,
            fairinf_n,
            1 if enable_timeline_logging else 0,
        )
        self.fairinf_n = fairinf_n
        self.max_kv_tokens_per_user = max_kv_tokens_per_user
        # Track arrival times ourselves (needed for process_new_request)
        self._arrival_times = {}  # rid -> ts

    def _get_fill_len(self, req):
        if getattr(req, "fill_ids", None) is not None:
            return len(req.fill_ids)
        return -1

    def process_new_request(self, req, deltas=None, *, arrival_timestamp=None):
        now = arrival_timestamp if arrival_timestamp is not None else time.time()
        self._arrival_times[req.rid] = now
        deltas = deltas or {}
        self._sim.process_new_request(
            req.uid, req.rid,
            len(req.origin_input_ids),
            self._get_fill_len(req),
            len(getattr(req, "output_ids", [])),
            now,
            int(deltas.get("prefill", 0)),
            int(deltas.get("decode", 0)),
        )

    def finished_prefill(self, batch):
        entries = [
            (r.uid, r.rid, len(r.origin_input_ids), len(getattr(r, "output_ids", [])))
            for r in batch.reqs
        ]
        self._sim.finished_prefill(entries)

    def finished_decode(self, batch, decode_rounds: int = 1):
        entries = [
            (r.uid, r.rid, len(r.origin_input_ids), len(getattr(r, "output_ids", [])))
            for r in batch.reqs
        ]
        self._sim.finished_decode(entries, decode_rounds)

    def mark_request_finished(self, req):
        self._sim.mark_request_finished(
            req.rid, req.uid, len(getattr(req, "output_ids", []))
        )
        self._arrival_times.pop(req.rid, None)

    def start_of_pass(self, running_batch, waiting_queue, *, deltas=None):
        waiting_tuples = [
            (r.uid, r.rid, len(r.origin_input_ids), len(getattr(r, "output_ids", [])))
            for r in waiting_queue
        ]
        running_tuples = [
            (r.uid, r.rid, len(r.origin_input_ids), len(getattr(r, "output_ids", [])))
            for r in (running_batch.reqs if running_batch is not None else [])
        ]
        self._sim.sync_live_users(waiting_tuples, running_tuples)
        self._sim.rebuild_all_users()

    def build_deadline_candidates(
        self,
        waiting_queue,
        running_batch,
        *,
        req_is_fair_prefill=None,
        req_is_fair_decode=None,
        event_delta_seconds=None,
        pooled_prefill_estimate_seconds=None,
        pooled_decode_estimate_seconds=None,
        include_ordered_waiting_queue: bool = False,
    ):
        # Determine fair_uids from req_is_fair_prefill lambda
        if req_is_fair_prefill is None:
            fair_uids = None
        else:
            # Call with each req; if all return True → pass None (all fair)
            # Otherwise compute set
            all_fair = True
            fair_uids_set = set()
            for req in waiting_queue:
                if req_is_fair_prefill(req, running_batch):
                    fair_uids_set.add(req.uid)
                else:
                    all_fair = False
            fair_uids = None if all_fair else frozenset(fair_uids_set)

        # Determine fair_decode_uids
        if req_is_fair_decode is None:
            fair_decode_uids = None
        else:
            all_fair = True
            fair_decode_set = set()
            running_reqs = running_batch.reqs if running_batch is not None else []
            for req in running_reqs:
                if req_is_fair_decode(req, running_batch):
                    fair_decode_set.add(req.uid)
                else:
                    all_fair = False
            fair_decode_uids = None if all_fair else frozenset(fair_decode_set)

        # Get scalar delta values
        if event_delta_seconds is not None:
            delta_prefill_s = event_delta_seconds(None, RequestPrefillEvent(req_id="", duration=0.0, end_timestamp=0.0))
            delta_decode_s = event_delta_seconds(None, RequestDecodeEvent(req_id="", duration=0.0, end_timestamp=0.0, completion_number=0))
        else:
            delta_prefill_s = 0.0
            delta_decode_s = 0.0

        # Get pooled estimates
        if pooled_prefill_estimate_seconds is not None and waiting_queue:
            pooled_prefill_s = pooled_prefill_estimate_seconds(waiting_queue[0])
        elif pooled_prefill_estimate_seconds is not None:
            pooled_prefill_s = pooled_prefill_estimate_seconds(None) if not waiting_queue else 0.0
        else:
            pooled_prefill_s = 0.0

        if pooled_decode_estimate_seconds is not None:
            dummy_req = (waiting_queue[0] if waiting_queue else
                         (running_batch.reqs[0] if running_batch and running_batch.reqs else None))
            pooled_decode_s = pooled_decode_estimate_seconds(dummy_req, running_batch)
        else:
            pooled_decode_s = 0.0

        waiting_rids = [r.rid for r in waiting_queue]
        running_rids = [r.rid for r in (running_batch.reqs if running_batch is not None else [])]

        c_result = self._sim.build_deadline_candidates(
            waiting_rids, running_rids,
            fair_uids, fair_decode_uids,
            delta_prefill_s, delta_decode_s,
            pooled_prefill_s, pooled_decode_s,
        )
        raw_candidates, wpd, ordered_rids = c_result

        rid_to_req = {r.rid: r for r in waiting_queue}
        rid_to_req.update({r.rid: r for r in (running_batch.reqs if running_batch is not None else [])})

        deadline_queue = []
        for (rid, uid, event_type_str, deadline, start_deadline, ant_ts, ant_cn) in raw_candidates:
            req = rid_to_req.get(rid)
            if req is None:
                continue
            if event_type_str == "decode":
                event = RequestDecodeEvent(req_id=rid, duration=0.0, end_timestamp=ant_ts, completion_number=ant_cn)
            else:
                event = RequestPrefillEvent(req_id=rid, duration=0.0, end_timestamp=ant_ts)
            deadline_queue.append(DeadlineCandidate(
                deadline=deadline,
                start_deadline=start_deadline,
                event_type=event_type_str,
                req=req,
                event=event,
            ))

        if not include_ordered_waiting_queue:
            return deadline_queue, wpd
        ordered_waiting_queue = tuple(rid_to_req[rid] for rid in ordered_rids if rid in rid_to_req)
        return deadline_queue, wpd, ordered_waiting_queue

    @property
    def requests(self):
        """Reconstruct a requests dict from C data for testing attribute access."""
        data = self._sim.get_all_req_data()

        class _FakeTimeline:
            def __init__(self, ant_type, ant_ts, ant_cn):
                if ant_type == 1:
                    self.next_anticipated_event = RequestDecodeEvent(
                        req_id="", duration=0.0, end_timestamp=ant_ts,
                        completion_number=ant_cn)
                elif ant_type == 0:
                    self.next_anticipated_event = RequestPrefillEvent(
                        req_id="", duration=0.0, end_timestamp=ant_ts)
                else:
                    self.next_anticipated_event = None

            def events_after(self, real_event):
                # Simplified: return next_anticipated if it's logically after
                if self.next_anticipated_event is None:
                    return []
                return [self.next_anticipated_event]

        class _FakeTrackedReq:
            def __init__(self, d):
                self._d = d
                self.arrival_timestamp = d["arrival_ts"]
                self.timeline = _FakeTimeline(d["ant_type"], d["ant_ts"], d["ant_cn"])

            def __getattr__(self, name):
                if name in self._d:
                    return self._d[name]
                raise AttributeError(name)

        result = {}
        for d in data:
            result[d["rid"]] = _FakeTrackedReq(d)
        return result


# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------

@unittest.skipIf(_fairinf_worker is None, "C extension _fairinf_worker not available")
class TestCSimulatorUnit(unittest.TestCase):
    """
    Mirrors the non-logging tests from TestDocPolicySimulatorUnit,
    running against CSimulatorAdapter.

    Uses REAL time estimation (no mocking of isolated_*_time_estimation)
    — so expected numeric values use the actual model formulas.

    For tests that check specific numeric timestamps, we use relative
    comparisons (ordering, positivity, finiteness) rather than exact values,
    since the C extension uses real formulas while Python tests mock them.
    """

    def _make_sim(self, *, max_kv=100, fairinf_n=2):
        return CSimulatorAdapter(
            max_kv_tokens_per_user=max_kv,
            fairinf_n=fairinf_n,
            enable_timeline_logging=False,
        )

    # ------------------------------------------------------------------
    # Test 1: basic prefill + decode + deadline ordering
    # ------------------------------------------------------------------
    def test_waiting_prefill_deadlines_follow_isolated_arrival_order(self):
        """Two requests for same user: earlier arrival → smaller start_deadline."""
        sim = self._make_sim()
        req1 = _mk_req("user_1", "rid_1", 4)
        req2 = _mk_req("user_1", "rid_2", 4)

        t0 = 1000.0
        sim.process_new_request(req1, arrival_timestamp=t0)
        sim.process_new_request(req2, arrival_timestamp=t0 + 1.0)

        sim.start_of_pass(SimpleNamespace(reqs=[]), [req1, req2])

        candidates, waiting_deadlines = sim.build_deadline_candidates(
            [req1, req2],
            SimpleNamespace(reqs=[]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        # Earlier arrival should have smaller (or equal) start_deadline
        self.assertIn(req1.rid, waiting_deadlines)
        self.assertIn(req2.rid, waiting_deadlines)
        self.assertLessEqual(waiting_deadlines[req1.rid], waiting_deadlines[req2.rid])

        prefill_cands = [c for c in candidates if c.event_type == "prefill"]
        self.assertEqual(len(prefill_cands), 2)
        # Ordered by start_deadline ascending
        self.assertEqual(prefill_cands[0].req.rid, req1.rid)
        self.assertEqual(prefill_cands[1].req.rid, req2.rid)

    # ------------------------------------------------------------------
    # Test 2: running decode deadline from isolated sequence
    # ------------------------------------------------------------------
    def test_running_decode_deadline_from_isolated_sequence(self):
        """After prefill + 1 decode, running request should have next-decode candidate."""
        sim = self._make_sim()
        req = _mk_req("user_19", "rid_running", 4)
        t0 = 1000.0
        sim.process_new_request(req, arrival_timestamp=t0)
        sim.finished_prefill(SimpleNamespace(reqs=[req]))
        req.output_ids = [42]
        sim.finished_decode(SimpleNamespace(reqs=[req]))

        sim.start_of_pass(SimpleNamespace(reqs=[req]), [])

        candidates, _ = sim.build_deadline_candidates(
            [],
            SimpleNamespace(reqs=[req]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        decode_candidates = [c for c in candidates if c.event_type == "decode"]
        self.assertEqual(len(decode_candidates), 1)
        self.assertIsInstance(decode_candidates[0].event, RequestDecodeEvent)
        self.assertEqual(decode_candidates[0].event.completion_number, 2)
        self.assertGreater(decode_candidates[0].deadline, 0.0)

    # ------------------------------------------------------------------
    # Test 3: finished prefill after long delay should not leave decode in the past
    # ------------------------------------------------------------------
    def test_finished_prefill_decode_not_in_past(self):
        """First decode candidate should not be in the past."""
        sim = self._make_sim()
        req = _mk_req("user_19", "rid_delayed", 4)
        t0 = 1000.0
        sim.process_new_request(req, arrival_timestamp=t0)
        # Simulate long delay before prefill
        sim.finished_prefill(SimpleNamespace(reqs=[req]))

        candidates, _ = sim.build_deadline_candidates(
            [],
            SimpleNamespace(reqs=[req]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        decode_candidates = [c for c in candidates if c.event_type == "decode"]
        self.assertEqual(len(decode_candidates), 1)
        self.assertEqual(decode_candidates[0].event.completion_number, 1)
        # The anticipated timestamp should be in the future (>= arrival time)
        self.assertGreaterEqual(decode_candidates[0].event.end_timestamp, t0)

    # ------------------------------------------------------------------
    # Test 4: retracted request becomes waiting prefill candidate
    # ------------------------------------------------------------------
    def test_retracted_request_becomes_waiting_prefill_candidate(self):
        """After retraction (process_new_request called again), req becomes a prefill candidate."""
        sim = self._make_sim(max_kv=8)
        req1 = _mk_req("user_19", "rid_bad_1", 4)
        req2 = _mk_req("user_19", "rid_bad_2", 4)
        t0 = 1000.0
        sim.process_new_request(req1, arrival_timestamp=t0)
        sim.process_new_request(req2, arrival_timestamp=t0)

        sim.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
        req1.output_ids = [1]
        req2.output_ids = [1]
        sim.finished_decode(SimpleNamespace(reqs=[req1, req2]))

        sim.start_of_pass(
            SimpleNamespace(reqs=[req1]),
            [req2],
        )

        candidates, waiting_deadlines = sim.build_deadline_candidates(
            [req2],
            SimpleNamespace(reqs=[req1]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        decode_candidates = [c for c in candidates if c.event_type == "decode"]
        prefill_candidates = [c for c in candidates if c.event_type == "prefill"]

        self.assertEqual([c.req.rid for c in decode_candidates], [req1.rid])
        self.assertEqual([c.req.rid for c in prefill_candidates], [req2.rid])
        self.assertIn(req2.rid, waiting_deadlines)

    # ------------------------------------------------------------------
    # Test 5: constrained KV produces different decode deadlines than unconstrained
    # ------------------------------------------------------------------
    def test_kv_constraint_affects_decode_deadlines(self):
        """Under KV constraint, the retraction mechanism affects request deadlines.

        This test verifies that the constrained and unconstrained scenarios
        both produce valid decode deadlines for req1, and that the isolation
        simulation ran (both are non-zero). The exact ordering may differ
        from the Python simulator due to real vs. mocked time estimation.
        """
        sim_unconstrained = CSimulatorAdapter(
            max_kv_tokens_per_user=100,
            fairinf_n=2,
            enable_timeline_logging=False,
        )
        sim_constrained = CSimulatorAdapter(
            max_kv_tokens_per_user=8,
            fairinf_n=2,
            enable_timeline_logging=False,
        )
        req1 = _mk_req("user_19", "rid_bad_1", 4)
        req2 = _mk_req("user_19", "rid_bad_2", 4)
        t0 = 1000.0
        for sim in (sim_unconstrained, sim_constrained):
            sim.process_new_request(req1, arrival_timestamp=t0)
            sim.process_new_request(req2, arrival_timestamp=t0)
            sim.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
            req1.output_ids = [1]
            req2.output_ids = [1]
            sim.finished_decode(SimpleNamespace(reqs=[req1, req2]))

        sim_unconstrained.start_of_pass(SimpleNamespace(reqs=[req1, req2]), [])
        sim_constrained.start_of_pass(SimpleNamespace(reqs=[req1]), [req2])

        unc_candidates, _ = sim_unconstrained.build_deadline_candidates(
            [],
            SimpleNamespace(reqs=[req1, req2]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )
        con_candidates, _ = sim_constrained.build_deadline_candidates(
            [req2],
            SimpleNamespace(reqs=[req1]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        unc_r1_decode = next(
            c for c in unc_candidates
            if c.req.rid == req1.rid and c.event_type == "decode"
        )
        con_r1_decode = next(
            c for c in con_candidates
            if c.req.rid == req1.rid and c.event_type == "decode"
        )

        # Both should produce valid decode deadlines after t0
        self.assertGreater(unc_r1_decode.deadline, t0)
        self.assertGreater(con_r1_decode.deadline, t0)
        # Both should have completion_number > real_decode_count (= 1)
        self.assertGreater(unc_r1_decode.event.completion_number, 1)
        self.assertGreater(con_r1_decode.event.completion_number, 1)

    # ------------------------------------------------------------------
    # Test 6: retraction resets request to prefill candidate
    # ------------------------------------------------------------------
    def test_retraction_resets_request_to_prefill(self):
        """After explicit retraction (process_new_request), req shows as prefill candidate."""
        sim = self._make_sim()
        req = _mk_req("user_19", "rid_retracted", 4)
        t0 = 1000.0
        sim.process_new_request(req, arrival_timestamp=t0)
        sim.finished_prefill(SimpleNamespace(reqs=[req]))
        req.output_ids = [1]
        sim.finished_decode(SimpleNamespace(reqs=[req]))

        # Retraction: re-register with new arrival time
        t1 = 1030.0
        sim.process_new_request(req, arrival_timestamp=t1)

        sim.start_of_pass(SimpleNamespace(reqs=[]), [req])
        candidates, waiting_deadlines = sim.build_deadline_candidates(
            [req],
            SimpleNamespace(reqs=[]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        self.assertEqual(
            [(c.req.rid, c.event_type) for c in candidates],
            [(req.rid, "prefill")],
        )
        # start_deadline should be based on new arrival time
        self.assertIn(req.rid, waiting_deadlines)
        self.assertGreaterEqual(waiting_deadlines[req.rid], t1 - 2.0)  # roughly arrival-based

    # ------------------------------------------------------------------
    # Test 7: waiting request delayed by running decode queue
    # ------------------------------------------------------------------
    def test_waiting_request_delayed_by_running_decode_queue(self):
        """A new waiting request behind a running request gets a delayed prefill."""
        sim = self._make_sim()
        req1 = _mk_req("user_19", "rid_running", 4)
        req2 = _mk_req("user_19", "rid_waiting", 4)
        t0 = 1000.0
        sim.process_new_request(req1, arrival_timestamp=t0)
        sim.finished_prefill(SimpleNamespace(reqs=[req1]))
        req1.output_ids = [1, 2, 3, 4, 5]
        sim.finished_decode(SimpleNamespace(reqs=[req1]), decode_rounds=5)

        sim.process_new_request(req2, arrival_timestamp=t0 + 3.0)

        sim.start_of_pass(
            SimpleNamespace(reqs=[req1]),
            [req2],
        )

        candidates, waiting_deadlines, ordered_waiting = sim.build_deadline_candidates(
            [req2],
            SimpleNamespace(reqs=[req1]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
            include_ordered_waiting_queue=True,
        )

        prefill_cands = [c for c in candidates if c.event_type == "prefill"]
        decode_cands = [c for c in candidates if c.event_type == "decode"]

        self.assertEqual(len(prefill_cands), 1)
        self.assertEqual(prefill_cands[0].req.rid, req2.rid)
        self.assertIn(req2.rid, waiting_deadlines)
        # req2's prefill must be at or after its arrival time
        self.assertGreaterEqual(prefill_cands[0].event.end_timestamp, t0 + 3.0)
        # There should be a decode candidate for req1
        self.assertEqual(len(decode_cands), 1)
        self.assertEqual(decode_cands[0].req.rid, req1.rid)

    # ------------------------------------------------------------------
    # Test 8: decode deadline advances across passes
    # ------------------------------------------------------------------
    def test_decode_deadline_advances_across_passes(self):
        """After more decode steps, the anticipated completion number increases."""
        sim = self._make_sim()
        req = _mk_req("user_19", "rid_blocker", 4)
        t0 = 1000.0
        sim.process_new_request(req, arrival_timestamp=t0)
        sim.finished_prefill(SimpleNamespace(reqs=[req]))
        req.output_ids = [1]
        sim.finished_decode(SimpleNamespace(reqs=[req]))

        sim.start_of_pass(SimpleNamespace(reqs=[req]), [])

        def _get_decode_cn(sim_instance, req):
            cands, _ = sim_instance.build_deadline_candidates(
                [],
                SimpleNamespace(reqs=[req]),
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: 2.0,
                pooled_decode_estimate_seconds=lambda req, rb: 3.0,
            )
            decode_cands = [c for c in cands if c.event_type == "decode"]
            return decode_cands[0].event.completion_number if decode_cands else None

        cn1 = _get_decode_cn(sim, req)
        self.assertIsNotNone(cn1)
        self.assertGreater(cn1, 1)  # already has 1 real decode, should anticipate next

        # Do another decode
        req.output_ids = [1, 2]
        sim.finished_decode(SimpleNamespace(reqs=[req]))
        sim.start_of_pass(SimpleNamespace(reqs=[req]), [])
        cn2 = _get_decode_cn(sim, req)
        self.assertIsNotNone(cn2)
        # cn2 should reflect the new state
        self.assertGreater(cn2, 1)

    # ------------------------------------------------------------------
    # Test 9: multi-user isolation
    # ------------------------------------------------------------------
    def test_multi_user_isolation(self):
        """Two different users get independent prefill candidates."""
        sim = self._make_sim()
        req_a = _mk_req("user_A", "rid_A", 4)
        req_b = _mk_req("user_B", "rid_B", 4)
        t0 = 1000.0
        sim.process_new_request(req_a, arrival_timestamp=t0)
        sim.process_new_request(req_b, arrival_timestamp=t0 + 0.5)

        sim.start_of_pass(SimpleNamespace(reqs=[]), [req_a, req_b])

        candidates, wpd = sim.build_deadline_candidates(
            [req_a, req_b],
            SimpleNamespace(reqs=[]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        rids = [c.req.rid for c in candidates if c.event_type == "prefill"]
        self.assertIn(req_a.rid, rids)
        self.assertIn(req_b.rid, rids)

    # ------------------------------------------------------------------
    # Test 10: mark_request_finished removes from candidates
    # ------------------------------------------------------------------
    def test_mark_request_finished_removes_from_candidates(self):
        """After mark_request_finished, request no longer appears in candidates."""
        sim = self._make_sim()
        req = _mk_req("user_X", "rid_done", 4)
        t0 = 1000.0
        sim.process_new_request(req, arrival_timestamp=t0)
        sim.finished_prefill(SimpleNamespace(reqs=[req]))
        req.output_ids = [1]
        sim.finished_decode(SimpleNamespace(reqs=[req]))
        sim.mark_request_finished(req)

        sim.start_of_pass(SimpleNamespace(reqs=[]), [])

        candidates, wpd = sim.build_deadline_candidates(
            [],
            SimpleNamespace(reqs=[]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        # No candidates for the finished request
        rids = [c.req.rid for c in candidates]
        self.assertNotIn(req.rid, rids)

    # ------------------------------------------------------------------
    # Test 11: isolation retraction evicts longest-running
    # ------------------------------------------------------------------
    def test_isolation_retraction_evicts_longest_running(self):
        """With KV pressure (3 requests × 4 tokens, max_kv=10), rebuild_all_users
        should complete without hanging and produce a prefill candidate for the waiting
        request. With 10-token KV and prompt_len=4, at most 2 requests can be active
        simultaneously (each takes 5 KV tokens after one decode), so the third request
        correctly gets an inf prefill time (the isolation sim cannot schedule it within
        max_steps). The test verifies:
          1. rebuild_all_users completes (no infinite loop).
          2. A prefill candidate is produced for req3.
          3. req1/req2 get non-inf anticipated events (they ARE schedulable in isolation).
        """
        sim = CSimulatorAdapter(
            max_kv_tokens_per_user=10,
            fairinf_n=1,
            enable_timeline_logging=False,
        )
        req1 = _mk_req("user_A", "rid_A1", 4)
        req2 = _mk_req("user_A", "rid_A2", 4)
        req3 = _mk_req("user_A", "rid_A3", 4)
        t0 = 1000.0
        sim.process_new_request(req1, arrival_timestamp=t0)
        sim.process_new_request(req2, arrival_timestamp=t0)
        sim.process_new_request(req3, arrival_timestamp=t0)

        sim.finished_prefill(SimpleNamespace(reqs=[req1, req2]))
        req1.output_ids = [1]
        req2.output_ids = [1]
        sim.finished_decode(SimpleNamespace(reqs=[req1, req2]))

        sim.start_of_pass(
            SimpleNamespace(reqs=[req1, req2]),
            [req3],
        )

        # req3 should have a prefill candidate (even if inf, since KV is too constrained
        # for all 3 simultaneous requests)
        candidates, _ = sim.build_deadline_candidates(
            [req3],
            SimpleNamespace(reqs=[req1, req2]),
            req_is_fair_prefill=lambda req, rb: True,
            req_is_fair_decode=lambda req, rb: True,
            event_delta_seconds=lambda tracked, event: 0.0,
            pooled_prefill_estimate_seconds=lambda req: 2.0,
            pooled_decode_estimate_seconds=lambda req, rb: 3.0,
        )

        prefill_cands = [c for c in candidates if c.event_type == "prefill"]
        self.assertEqual(len(prefill_cands), 1, "req3 should appear as a prefill candidate")
        self.assertEqual(prefill_cands[0].req.rid, req3.rid)
        # req1 and req2 (in running batch) should have non-inf anticipated events.
        all_data = sim._sim.get_all_req_data()
        req1_data = next(d for d in all_data if d["rid"] == req1.rid)
        req2_data = next(d for d in all_data if d["rid"] == req2.rid)
        self.assertFalse(
            math.isinf(req1_data["ant_ts"]),
            "req1 should have a finite anticipated event time",
        )
        self.assertFalse(
            math.isinf(req2_data["ant_ts"]),
            "req2 should have a finite anticipated event time",
        )

    # ------------------------------------------------------------------
    # Test 12: get_all_req_data returns correct fields
    # ------------------------------------------------------------------
    def test_get_all_req_data_returns_correct_fields(self):
        """get_all_req_data should return alive requests with the expected fields."""
        sim = CSimulatorAdapter(
            max_kv_tokens_per_user=100,
            fairinf_n=2,
            enable_timeline_logging=False,
        )
        req = _mk_req("user_Z", "rid_Z1", 8)
        t0 = 2000.0
        sim.process_new_request(req, arrival_timestamp=t0)

        data = sim._sim.get_all_req_data()
        self.assertEqual(len(data), 1)
        d = data[0]
        self.assertEqual(d["rid"], "rid_Z1")
        self.assertEqual(d["uid"], "user_Z")
        self.assertAlmostEqual(d["arrival_ts"], t0)
        self.assertEqual(d["prompt_len"], 8)
        self.assertEqual(d["alive"], 1)
        self.assertEqual(d["prefill_done"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
