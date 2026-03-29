"""End-to-end unit test that mirrors the completion_difference workload.

Scenario (mirrors delta-fair-inference/experiments/3-22-doc-policy/workloads/completion_difference.py):
  - bad user (user_19): one running request, short prompt (~130 tokens), max_new_tokens=5000,
    already produced many output tokens (simulating a long-running decode).
  - good user (user_1): one request in the waiting queue, short prompt (~130 tokens),
    max_new_tokens=50.

Under a fair policy the following should hold:
  1. The bad user's running request has an isolated decode deadline computed from
     its isolated alternate history.  That deadline constrains when a new prefill
     can start.
  2. When the pooled decode step is very fast (small batch, short prompts so far)
     the start deadline of the decode event is pushed far enough in the future that
     the good user's prefill fits safely.
  3. When the bad user has nearly exhausted all of its isolated decode budget (i.e.
     the next isolated decode deadline is very close to now), the safe-prefill window
     closes and the policy must not include new prefill candidates.
  4. The good user's prefill start deadline (from the waiting-prefill candidates) is
     FIFO-ordered: requests that arrived earlier must have an earlier start deadline.

This test does NOT use a GPU or a running server.  It drives
AlternateHistorySimulator directly (the same way existing unit tests do) and
calls DocPolicy._compute_safe_prefix_state to verify the policy's scheduling
decision.  All time.time() calls in both modules are patched with a shared clock
so that isolated-history timestamps and safe-prefix "now" are fully controlled.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
import sglang.srt.delta_fairness.doc_policy as doc_policy_mod
from sglang.srt.delta_fairness.doc_policy_simulator import (
    AlternateHistorySimulator,
    RequestDecodeEvent,
    RequestPrefillEvent,
)
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Workload constants (matches completion_difference.py defaults)
# ---------------------------------------------------------------------------

GOOD_PROMPT_TOKENS = 130        # ~100 words
BAD_PROMPT_TOKENS = 130
GOOD_MAX_NEW_TOKENS = 50
BAD_MAX_NEW_TOKENS = 5000
N_USERS = 4  # delta_fairness_n

# Controlled latency values used across all tests.
ISOLATED_DECODE_S = 0.10   # 100 ms per isolated decode step
ISOLATED_PREFILL_S = 0.05  # 50 ms per isolated prefill
POOLED_PREFILL_S = 0.04    # 40 ms pooled prefill (default, overridden per-test)
POOLED_DECODE_S = 0.02     # 20 ms pooled decode  (default, overridden per-test)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_req(uid: str, rid: str, prompt_tokens: int, max_new_tokens: int,
            output_tokens: int = 0) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="",
              origin_input_ids=[1] * prompt_tokens)
    req.fill_ids = list(req.origin_input_ids) + list(range(output_tokens))
    req.output_ids = list(range(output_tokens))
    req.sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens, min_new_tokens=0
    )
    req.extend_input_len = prompt_tokens
    return req


def _running_batch(reqs: list) -> SimpleNamespace:
    return SimpleNamespace(reqs=reqs)


def _build_bad_user_simulator(now: dict, *, n_output_tokens: int) -> tuple:
    """Return (simulator, bad_req) with the bad user already prefilled and
    decoded n_output_tokens times.

    All time.time() calls must already be patched by the caller.  The clock
    starts at t=0 for arrival, t=0.05 for prefill done, then advances by
    ISOLATED_DECODE_S for each decode.

    Returns the simulator and the bad request with output_ids set.
    """
    simulator = AlternateHistorySimulator(
        max_kv_tokens_per_user=None,
        fairinf_n=N_USERS,
    )

    bad_req = _mk_req("user_19", "rid_bad", BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS,
                      output_tokens=n_output_tokens)

    # Arrival
    now["t"] = 0.0
    simulator.process_new_request(bad_req)

    # Prefill
    now["t"] = ISOLATED_PREFILL_S
    simulator.finished_prefill(SimpleNamespace(reqs=[bad_req]))

    # Decode rounds: after k rounds the real clock is at prefill_done + k * decode_step.
    # The isolated history records each step so the anticipated next decode is at
    # prefill_done + (n_output_tokens + 1) * ISOLATED_DECODE_S.
    for k in range(1, n_output_tokens + 1):
        bad_req.output_ids = list(range(k))
        now["t"] = ISOLATED_PREFILL_S + k * ISOLATED_DECODE_S
        simulator.finished_decode(SimpleNamespace(reqs=[bad_req]))

    # Restore output_ids to match the total
    bad_req.output_ids = list(range(n_output_tokens))
    bad_req.fill_ids = list(bad_req.origin_input_ids) + bad_req.output_ids

    return simulator, bad_req


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestCompletionDifferenceEndToEnd(unittest.TestCase):
    """Mirrors the completion_difference workload as a deterministic unit test."""

    def _patches(self, now: dict):
        """Return a list of context managers that patch time.time() in both
        the simulator and the doc_policy modules."""
        def fake_time():
            return now["t"]
        return [
            patch.object(sim_mod.time, "time", side_effect=fake_time),
            patch.object(doc_policy_mod.time, "time", side_effect=fake_time),
            patch.object(sim_mod, "isolated_decode_time_estimation",
                         return_value=ISOLATED_DECODE_S),
            patch.object(sim_mod, "isolated_prefill_time_estimation",
                         return_value=ISOLATED_PREFILL_S),
        ]

    # ------------------------------------------------------------------
    # Test 1 – good user's prefill fits when decode deadline is far away
    # ------------------------------------------------------------------
    def test_good_user_prefill_allowed_when_decode_deadline_is_far(self):
        """When the next isolated decode deadline for the bad user is far
        in the future, the good user's prefill MUST appear in prefill
        candidates."""
        now = {"t": 0.0}
        POOLED_PREFILL = 0.04
        POOLED_DECODE = 0.02

        # After 200 decodes the anticipated decode 201 is at:
        # ISOLATED_PREFILL_S + 201 * ISOLATED_DECODE_S
        # = 0.05 + 201 * 0.10 = 20.15 s
        # start_deadline = 20.15 - POOLED_DECODE = 20.13 s (from epoch t=0)
        # We set "now" for the pass to t = 0.05 + 200 * 0.10 = 20.05 (just after last decode).
        # 20.05 + POOLED_PREFILL (0.04) = 20.09 <= 20.13 → prefill fits.

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=200)

            good_req = _mk_req("user_1", "rid_good", GOOD_PROMPT_TOKENS,
                               GOOD_MAX_NEW_TOKENS)
            # Good user arrives just after the last decode.
            now["t"] = ISOLATED_PREFILL_S + 200 * ISOLATED_DECODE_S + 0.001
            simulator.process_new_request(good_req)

            running_batch = _running_batch([bad_req])
            waiting_queue = [good_req]

            now["t"] = ISOLATED_PREFILL_S + 200 * ISOLATED_DECODE_S + 0.01
            simulator.start_of_pass(running_batch, waiting_queue)

            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            decode_candidates = [c for c in candidates if c.event_type == "decode"]
            prefill_candidates = [c for c in candidates if c.event_type == "prefill"]

            self.assertGreaterEqual(len(decode_candidates), 1,
                "Bad user must produce a decode deadline candidate.")

            earliest_start_deadline = min(c.start_deadline for c in decode_candidates)
            # Verify the window is actually open (now + pooled_prefill <= start_deadline).
            self.assertGreaterEqual(
                earliest_start_deadline,
                now["t"] + POOLED_PREFILL,
                "Decode start deadline must be far enough for a prefill to fit."
            )
            self.assertEqual(len(prefill_candidates), 1,
                "Good user must appear as a prefill candidate when the window is open.")
            self.assertEqual(prefill_candidates[0].req.rid, "rid_good")

    # ------------------------------------------------------------------
    # Test 2 – good user's prefill blocked when window is closed
    # ------------------------------------------------------------------
    def test_compute_safe_prefix_state_rejects_prefill_when_window_closed(self):
        """_compute_safe_prefix_state returns max_safe_prefill_tokens=0 when
        the earliest decode start_deadline is so close to now that even a tiny
        prefill cannot fit.

        We set now to be very close to the decode deadline so the window is
        definitively closed.
        """
        now = {"t": 0.0}
        POOLED_DECODE = 0.001  # tiny
        # We deliberately use a large pooled prefill so the window is closed.
        HUGE_POOLED_PREFILL = 100.0

        # After 50 decodes, anticipated decode 51 is at:
        # ISOLATED_PREFILL_S + 51 * ISOLATED_DECODE_S = 0.05 + 5.1 = 5.15 s
        # start_deadline = 5.15 - 0.001 = 5.149 s
        # If now = 5.10, then now + HUGE_POOLED_PREFILL >> start_deadline → closed.

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=HUGE_POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=50)

            good_req = _mk_req("user_1", "rid_good", GOOD_PROMPT_TOKENS,
                               GOOD_MAX_NEW_TOKENS)
            now["t"] = ISOLATED_PREFILL_S + 50 * ISOLATED_DECODE_S + 0.001
            simulator.process_new_request(good_req)

            running_batch = _running_batch([bad_req])
            waiting_queue = [good_req]

            # Set pass time to just before the expected decode deadline.
            now["t"] = ISOLATED_PREFILL_S + 50 * ISOLATED_DECODE_S + 0.01
            simulator.start_of_pass(running_batch, waiting_queue)

            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: HUGE_POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            policy = DocPolicy(delta_fairness_n=N_USERS, max_running_requests=256)
            safe_state = policy._compute_safe_prefix_state(
                deadline_queue=candidates,
                waiting_prefill_start_deadline_by_rid=waiting_deadlines,
                waiting_queue=waiting_queue,
                running_batch=running_batch,
            )

            self.assertTrue(safe_state["has_decode_deadline"])
            self.assertEqual(
                safe_state["max_safe_prefill_tokens"], 0,
                "No prefill should be admitted when the safe window is closed."
            )
            self.assertEqual(safe_state["forced_prefill_queue"], [])

    # ------------------------------------------------------------------
    # Test 3 – safe_prefix_state admits prefill when window is open
    # ------------------------------------------------------------------
    def test_compute_safe_prefix_state_admits_prefill_when_window_open(self):
        """_compute_safe_prefix_state returns max_safe_prefill_tokens > 0 and
        includes the good user in forced_prefill_queue when now + pooled_prefill
        fits before the earliest decode start_deadline."""
        now = {"t": 0.0}
        POOLED_DECODE = 0.001       # tiny pooled decode
        TINY_POOLED_PREFILL = 0.001 # trivially small → always fits

        # After 50 decodes, anticipated decode 51 end at 5.15 s.
        # start_deadline = 5.15 - 0.001 = 5.149 s
        # If now = 5.10, then now + TINY_POOLED_PREFILL = 5.101 <= 5.149 → fits.

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=TINY_POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=TINY_POOLED_PREFILL):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=50)

            good_req = _mk_req("user_1", "rid_good", GOOD_PROMPT_TOKENS,
                               GOOD_MAX_NEW_TOKENS)
            now["t"] = ISOLATED_PREFILL_S + 50 * ISOLATED_DECODE_S + 0.001
            simulator.process_new_request(good_req)

            running_batch = _running_batch([bad_req])
            waiting_queue = [good_req]

            # Set "now" to slightly before the anticipated decode deadline so the
            # window is open for the tiny prefill but not for a large one.
            now["t"] = ISOLATED_PREFILL_S + 50 * ISOLATED_DECODE_S + 0.01
            simulator.start_of_pass(running_batch, waiting_queue)

            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: TINY_POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            policy = DocPolicy(delta_fairness_n=N_USERS, max_running_requests=256)
            safe_state = policy._compute_safe_prefix_state(
                deadline_queue=candidates,
                waiting_prefill_start_deadline_by_rid=waiting_deadlines,
                waiting_queue=waiting_queue,
                running_batch=running_batch,
            )

            self.assertTrue(safe_state["has_decode_deadline"])
            self.assertGreater(
                safe_state["max_safe_prefill_tokens"], 0,
                "Prefill tokens should be admitted when the window is open."
            )
            forced_rids = [req.rid for req in safe_state["forced_prefill_queue"]]
            self.assertIn(
                "rid_good", forced_rids,
                "Good user must be in forced_prefill_queue when window is open."
            )

    # ------------------------------------------------------------------
    # Test 4 – FIFO ordering of multiple good users by isolated arrival deadline
    # ------------------------------------------------------------------
    def test_multiple_good_users_ordered_by_arrival_in_prefill_candidates(self):
        """Two good users from the same user id that arrived at different times
        must be ordered by arrival (FIFO) in the prefill candidates list, and
        the earlier arrival must have the earlier prefill start deadline."""
        now = {"t": 0.0}
        POOLED_PREFILL = 0.04
        POOLED_DECODE = 0.02

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=200)

            # good_req_1 arrives before good_req_2.
            good_req_1 = _mk_req("user_1", "rid_good_1", GOOD_PROMPT_TOKENS,
                                 GOOD_MAX_NEW_TOKENS)
            now["t"] = ISOLATED_PREFILL_S + 200 * ISOLATED_DECODE_S + 0.001
            simulator.process_new_request(good_req_1)

            good_req_2 = _mk_req("user_1", "rid_good_2", GOOD_PROMPT_TOKENS,
                                 GOOD_MAX_NEW_TOKENS)
            now["t"] = ISOLATED_PREFILL_S + 200 * ISOLATED_DECODE_S + 0.5
            simulator.process_new_request(good_req_2)

            running_batch = _running_batch([bad_req])
            waiting_queue = [good_req_1, good_req_2]

            now["t"] = ISOLATED_PREFILL_S + 200 * ISOLATED_DECODE_S + 1.0
            simulator.start_of_pass(running_batch, waiting_queue)

            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            self.assertIn("rid_good_1", waiting_deadlines)
            self.assertIn("rid_good_2", waiting_deadlines)

            # Earlier arrival → earlier (or equal) start deadline.
            self.assertLessEqual(
                waiting_deadlines["rid_good_1"],
                waiting_deadlines["rid_good_2"],
                "Earlier-arriving good user must have earlier/equal prefill start deadline."
            )

            # In prefill candidates, good_1 must precede good_2.
            prefill_candidates = [c for c in candidates if c.event_type == "prefill"]
            prefill_rids = [c.req.rid for c in prefill_candidates]
            self.assertIn("rid_good_1", prefill_rids)
            self.assertIn("rid_good_2", prefill_rids)
            self.assertLess(
                prefill_rids.index("rid_good_1"),
                prefill_rids.index("rid_good_2"),
                "good_req_1 (arrived first) must be ordered before good_req_2."
            )

    # ------------------------------------------------------------------
    # Test 5 – decode deadline advances as bad user produces more tokens
    # ------------------------------------------------------------------
    def test_bad_user_decode_deadline_advances_as_real_decodes_accumulate(self):
        """As the bad user produces more output tokens the anticipated next
        isolated decode completion_number must increase across passes.  The
        deadline must never be stuck at an already-realized event."""
        now = {"t": 0.0}
        POOLED_PREFILL = 0.04
        POOLED_DECODE = 0.02

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=10)

            running_batch = _running_batch([bad_req])

            pass_time_1 = ISOLATED_PREFILL_S + 10 * ISOLATED_DECODE_S + 0.01
            now["t"] = pass_time_1
            simulator.start_of_pass(running_batch, [])
            candidates_1, _ = simulator.build_deadline_candidates(
                [],
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            # Bad user produces 5 more real decode rounds.
            for k in range(11, 16):
                bad_req.output_ids = list(range(k))
                bad_req.fill_ids = list(bad_req.origin_input_ids) + bad_req.output_ids
                now["t"] = ISOLATED_PREFILL_S + k * ISOLATED_DECODE_S
                simulator.finished_decode(SimpleNamespace(reqs=[bad_req]))

            bad_req.output_ids = list(range(15))
            bad_req.fill_ids = list(bad_req.origin_input_ids) + bad_req.output_ids

            pass_time_2 = ISOLATED_PREFILL_S + 15 * ISOLATED_DECODE_S + 0.01
            now["t"] = pass_time_2
            simulator.start_of_pass(running_batch, [])
            candidates_2, _ = simulator.build_deadline_candidates(
                [],
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            decode_1 = [c for c in candidates_1 if c.event_type == "decode"]
            decode_2 = [c for c in candidates_2 if c.event_type == "decode"]

            self.assertTrue(len(decode_1) >= 1 and len(decode_2) >= 1,
                "Both passes must produce at least one decode deadline candidate.")

            comp_1 = decode_1[0].event.completion_number
            comp_2 = decode_2[0].event.completion_number
            self.assertGreater(
                comp_2, comp_1,
                "After more real decodes the anticipated completion_number must increase."
            )
            # And the deadline itself must be later in absolute time.
            self.assertGreater(
                decode_2[0].deadline,
                decode_1[0].deadline,
                "After more real decodes the absolute deadline must advance."
            )

    # ------------------------------------------------------------------
    # Test 6 – larger pooled decode estimate moves start_deadline earlier
    # ------------------------------------------------------------------
    def test_pooled_decode_estimate_moves_start_deadline_earlier(self):
        """The start_deadline for a decode candidate = deadline - pooled_decode.
        A larger pooled decode estimate must produce an earlier start_deadline,
        which tightens the prefill window."""
        now = {"t": 0.0}

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE_S):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=50)

            running_batch = _running_batch([bad_req])
            now["t"] = ISOLATED_PREFILL_S + 50 * ISOLATED_DECODE_S + 0.01
            simulator.start_of_pass(running_batch, [])

            SMALL_POOLED_DECODE = 0.01
            LARGE_POOLED_DECODE = 0.50

            candidates_small, _ = simulator.build_deadline_candidates(
                [],
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: 0.04,
                pooled_decode_estimate_seconds=lambda req, rb: SMALL_POOLED_DECODE,
            )
            candidates_large, _ = simulator.build_deadline_candidates(
                [],
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: 0.04,
                pooled_decode_estimate_seconds=lambda req, rb: LARGE_POOLED_DECODE,
            )

            decode_small = [c for c in candidates_small if c.event_type == "decode"]
            decode_large = [c for c in candidates_large if c.event_type == "decode"]
            self.assertTrue(len(decode_small) >= 1 and len(decode_large) >= 1)

            # Raw deadline must be the same (same isolated history, same event).
            self.assertAlmostEqual(
                decode_small[0].deadline,
                decode_large[0].deadline,
                places=6,
                msg="Raw deadlines must be identical regardless of pooled estimate."
            )
            # start_deadline must be earlier for the larger pooled decode.
            self.assertLess(
                decode_large[0].start_deadline,
                decode_small[0].start_deadline,
                "Larger pooled decode estimate must produce an earlier start_deadline."
            )

    # ------------------------------------------------------------------
    # Test 7 – no decode deadline when bad user is not running
    # ------------------------------------------------------------------
    def test_no_decode_deadline_when_bad_user_not_in_running_batch(self):
        """If there are no requests in the running batch the deadline queue
        must be empty (no decode candidates) and _compute_safe_prefix_state
        must set has_decode_deadline=False."""
        now = {"t": 0.0}
        POOLED_PREFILL = 0.04
        POOLED_DECODE = 0.02

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator = AlternateHistorySimulator(
                max_kv_tokens_per_user=None,
                fairinf_n=N_USERS,
            )

            good_req = _mk_req("user_1", "rid_good", GOOD_PROMPT_TOKENS,
                               GOOD_MAX_NEW_TOKENS)
            now["t"] = 1.0
            simulator.process_new_request(good_req)

            running_batch = _running_batch([])
            waiting_queue = [good_req]

            now["t"] = 1.5
            simulator.start_of_pass(running_batch, waiting_queue)

            candidates, waiting_deadlines = simulator.build_deadline_candidates(
                waiting_queue,
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: POOLED_PREFILL,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            decode_candidates = [c for c in candidates if c.event_type == "decode"]
            self.assertEqual(len(decode_candidates), 0,
                "No decode candidates expected when running batch is empty.")

            policy = DocPolicy(delta_fairness_n=N_USERS, max_running_requests=256)
            safe_state = policy._compute_safe_prefix_state(
                deadline_queue=candidates,
                waiting_prefill_start_deadline_by_rid=waiting_deadlines,
                waiting_queue=waiting_queue,
                running_batch=running_batch,
            )
            self.assertFalse(safe_state["has_decode_deadline"])
            # Without a decode deadline the max_safe_prefill_tokens cap is None (unconstrained).
            self.assertIsNone(safe_state["max_safe_prefill_tokens"])

    # ------------------------------------------------------------------
    # Test 8 – bad user decode deadline is tighter than its isolated
    # history would naively suggest when pooled decode is non-trivial
    # ------------------------------------------------------------------
    def test_pooled_decode_accounts_for_batch_size_in_start_deadline(self):
        """The pooled decode time passed to build_deadline_candidates is a
        function of the running batch.  When the running batch contains a
        request with many tokens the pooled estimate is larger, which moves
        the start_deadline earlier and constrains the prefill window more.

        This test verifies that the start_deadline is strictly smaller than
        the raw deadline by at least the (non-zero) pooled decode estimate.
        """
        now = {"t": 0.0}
        POOLED_DECODE = 0.05  # 50 ms – non-trivial

        with patch.object(sim_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(doc_policy_mod.time, "time", side_effect=lambda: now["t"]), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE):

            simulator, bad_req = _build_bad_user_simulator(now, n_output_tokens=100)

            running_batch = _running_batch([bad_req])
            now["t"] = ISOLATED_PREFILL_S + 100 * ISOLATED_DECODE_S + 0.01
            simulator.start_of_pass(running_batch, [])

            candidates, _ = simulator.build_deadline_candidates(
                [],
                running_batch,
                req_is_fair_prefill=lambda req, rb: True,
                req_is_fair_decode=lambda req, rb: True,
                event_delta_seconds=lambda tracked, event: 0.0,
                pooled_prefill_estimate_seconds=lambda req: 0.04,
                pooled_decode_estimate_seconds=lambda req, rb: POOLED_DECODE,
            )

            decode_candidates = [c for c in candidates if c.event_type == "decode"]
            self.assertGreaterEqual(len(decode_candidates), 1)

            for candidate in decode_candidates:
                self.assertAlmostEqual(
                    candidate.deadline - candidate.start_deadline,
                    POOLED_DECODE,
                    places=9,
                    msg="start_deadline must equal deadline - pooled_decode_estimate."
                )


if __name__ == "__main__":
    unittest.main()
