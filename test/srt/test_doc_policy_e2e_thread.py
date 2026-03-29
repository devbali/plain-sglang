"""End-to-end tests for DocPolicy decode bookkeeping.

All tests drive DocPolicy through the same public API the tp_worker scheduler
uses — no direct simulator calls, no GPU.  Time is faked with a shared clock
so all arithmetic is deterministic.

Real scheduler loop (from tp_worker.py):

  policy.process_new_request(req)                        # on request arrival
  policy.start_of_pass(running_batch, waiting_queue)     # top of each pass
  policy.finished_prefill(batch)                         # after prefill GPU fwd
  # --- GPU runs decode ---
  policy.prepare_during_gpu_execution(                   # during GPU decode fwd
      event_type="decode", running_batch=...,
      output_ids_already_applied=True, ...)
  policy.finished_decode(batch)                          # after decode completes

The prepare worker thread runs asynchronously and publishes snapshots that
start_of_pass consumes.

Design principle under test (design_implemented.md):
  The only timestamps the simulator takes from the real world are request
  arrival timestamps.  All isolated-timeline timestamps (prefill end,
  decode 1, decode 2, ...) are computed as:
      arrival + isolated_prefill_duration
      arrival + isolated_prefill_duration + isolated_decode_duration
      ...
  regardless of wall-clock time.
"""

from __future__ import annotations

import threading
import time
import unittest
from copy import copy
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.doc_policy_simulator as sim_mod
import sglang.srt.delta_fairness.doc_policy as doc_policy_mod
from sglang.srt.delta_fairness.doc_policy import DocPolicy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Latency constants — everything is deterministic
# ---------------------------------------------------------------------------

ISOLATED_DECODE_S  = 0.10   # 100 ms per isolated decode step
ISOLATED_PREFILL_S = 0.05   #  50 ms per isolated prefill
POOLED_PREFILL_S   = 0.04   #  40 ms pooled prefill
POOLED_DECODE_S    = 0.02   #  20 ms pooled decode

# Pooled system is 10× slower than isolated — simulates heavy queueing /
# memory pressure, which is the failure scenario from the workload.
WALL_DECODE_STEP = 10 * ISOLATED_DECODE_S   # 1.0 s per decode in wall time

N_USERS            = 4
BAD_PROMPT_TOKENS  = 130
GOOD_PROMPT_TOKENS = 130
BAD_MAX_NEW_TOKENS  = 5000
GOOD_MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Thread-safe fake clock
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    @property
    def t(self) -> float:
        with self._lock:
            return self._t

    @t.setter
    def t(self, value: float) -> None:
        with self._lock:
            self._t = value

    def advance(self, delta: float) -> float:
        with self._lock:
            self._t += delta
            return self._t


# ---------------------------------------------------------------------------
# Request / batch helpers
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
    req.waiting_time_in_decodes = 0
    req.first_time_in_waiting_queue = True
    req.prefix_indices = []
    return req


def _with_output(req: Req, output_tokens: int) -> Req:
    """Return a shallow copy of req with output_tokens decode tokens."""
    r = copy(req)
    r.output_ids = list(range(output_tokens))
    r.fill_ids = list(req.origin_input_ids) + r.output_ids
    return r


def _batch(*reqs) -> SimpleNamespace:
    return SimpleNamespace(reqs=list(reqs))


def _make_policy() -> DocPolicy:
    return DocPolicy(
        delta_fairness_n=N_USERS,
        max_running_requests=256,
        isolated_kv_tokens_per_user=None,
    )


def _wait_for_snapshot(policy: DocPolicy, *, min_task_seq: int,
                       timeout: float = 3.0) -> bool:
    """Spin until the prepare worker publishes a snapshot for min_task_seq."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        snapshot = policy._prepare_worker.latest_snapshot()
        if snapshot is not None and snapshot.task_seq >= min_task_seq:
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealSchedulerLoopDecodeBookkeeping(unittest.TestCase):
    """Drive DocPolicy through the real tp_worker scheduler loop and check
    that isolated decode deadlines remain anchored to arrival time, not wall
    clock, even when the pooled system is much slower than isolated speed.

    The failure mode from the completion_difference workload:
      - Wall clock advances much faster than isolated timeline
        (memory full, high queueing, pooled system slow).
      - Isolated decode deadlines drift to wall-clock time instead of
        arrival + prefill + N * isolated_decode.
      - The safe-prefill window never closes because the deadline always
        looks far in the future.
      - Good users starve; bad users keep decoding unchecked.
    """

    def test_isolated_deadline_anchored_to_arrival_not_wall_clock(self):
        """Simplest case: one bad user, wall clock 10× faster than isolated.

        Loop (mirrors tp_worker decode epoch):
          process_new_request
          start_of_pass (triggers prepare worker: process_new_request mutation,
                         then prefill prepare → finished_prefill on worker sim)
          finished_prefill
          [for each decode step:]
            start_of_pass
            prepare_during_gpu_execution(decode, prepare_pass_state=True)
            → worker: sync_request_progress_from_live → snapshot
            finished_decode

        After K decode rounds the anticipated next isolated decode deadline
        in the snapshot must be:
          arrival + ISOLATED_PREFILL_S + (K+1) * ISOLATED_DECODE_S

        It must NOT be:
          wall_clock_now + ISOLATED_DECODE_S
        which would happen if wall-clock timestamps leaked into the isolated
        timeline.
        """
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE_S):

            policy = _make_policy()

            # --- Arrival ---
            arrival_time = 0.0
            clock.t = arrival_time
            bad_req = _mk_req("user_19", "rid_bad",
                              BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
            policy.process_new_request(bad_req)

            # --- Prefill pass ---
            # start_of_pass triggers the prepare worker to apply the
            # process_new_request mutation (populating its simulator).
            policy.start_of_pass(None, [bad_req])
            # GPU runs prefill; wall clock advances to isolated prefill time.
            clock.t = arrival_time + ISOLATED_PREFILL_S
            # prepare_during_gpu_execution(prefill) tells the worker to call
            # finished_prefill on its simulator, recording the isolated prefill.
            policy.prepare_during_gpu_execution(
                event_type="prefill",
                running_batch=None,
                waiting_queue=[],
                scheduled_batch=_batch(bad_req),
                prepare_pass_state=False,
            )
            policy.finished_prefill(_batch(bad_req))

            K = 20
            last_task_seq = 0

            # --- Decode epoch: mirrors the tp_worker loop ---
            # Each iteration: start_of_pass → GPU decode →
            #   prepare_during_gpu_execution(output already applied) →
            #   finished_decode
            for k in range(1, K + 1):
                running = _batch(bad_req)
                policy.start_of_pass(running, [])

                # GPU runs decode — wall clock advances by WALL_DECODE_STEP
                # (10× isolated speed, simulating a congested pool).
                clock.t = arrival_time + ISOLATED_PREFILL_S + k * WALL_DECODE_STEP

                # Sampler produces one output token.
                bad_req = _with_output(bad_req, k)
                running = _batch(bad_req)

                last_task_seq = policy._prepare_worker._task_seq + 1
                policy.prepare_during_gpu_execution(
                    event_type="decode",
                    running_batch=running,
                    waiting_queue=[],
                    decode_steps=1,
                    output_ids_already_applied=True,
                    prepare_pass_state=True,   # request a snapshot
                )
                policy.finished_decode(running)

            # Wait for the worker to publish the last snapshot.
            got = _wait_for_snapshot(policy, min_task_seq=last_task_seq)
            self.assertTrue(got, "Prepare worker did not publish a snapshot in time.")

            snapshot = policy._prepare_worker.latest_snapshot()
            self.assertIsNotNone(snapshot)

            # Find the bad user's decode deadline in the snapshot.
            decode_candidates = [
                c for c in snapshot.deadline_queue
                if c.event_type == "decode" and c.req.rid == "rid_bad"
            ]
            self.assertGreaterEqual(
                len(decode_candidates), 1,
                "No decode deadline candidate for bad user in snapshot."
            )
            actual_deadline = decode_candidates[0].deadline
            wall_clock_now = clock.t

            # What the isolated timeline says the next decode should end at:
            expected_deadline = (
                arrival_time + ISOLATED_PREFILL_S + (K + 1) * ISOLATED_DECODE_S
            )
            # What the buggy path gives if wall-clock leaked in:
            buggy_deadline = wall_clock_now + ISOLATED_DECODE_S

            self.assertAlmostEqual(
                actual_deadline,
                expected_deadline,
                delta=ISOLATED_DECODE_S,
                msg=(
                    f"Decode deadline in snapshot after {K} rounds: {actual_deadline:.4f}s.\n"
                    f"Expected (isolated timeline): {expected_deadline:.4f}s.\n"
                    f"Wall clock now: {wall_clock_now:.4f}s.\n"
                    f"Buggy value (wall-clock anchor): {buggy_deadline:.4f}s.\n"
                    "Wall-clock time is leaking into the isolated timeline — "
                    "the deadline drifts far into the future and the safe-prefill "
                    "window never closes."
                ),
            )

    def test_safe_prefill_window_closes_at_isolated_deadline_not_wall_clock(self):
        """With a good user waiting and wall clock 10× faster than isolated,
        the safe-prefill window must close at the isolated deadline, not at
        the far-future wall-clock-anchored deadline.

        After K decode rounds the isolated start_deadline is:
          arrival + ISOLATED_PREFILL_S + (K+1) * ISOLATED_DECODE_S - POOLED_DECODE_S

        Setting now to just past that value must close the window
        (max_safe_prefill_tokens == 0).

        If the bug is present the deadline is at wall_clock_now + ISOLATED_DECODE_S
        and the window stays open — good user can always be prefilled regardless
        of how far the bad user is in isolation.
        """
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE_S):

            policy = _make_policy()

            arrival_time = 0.0
            clock.t = arrival_time

            bad_req = _mk_req("user_19", "rid_bad",
                              BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
            good_req = _mk_req("user_1", "rid_good",
                               GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)
            policy.process_new_request(bad_req)
            policy.process_new_request(good_req)

            policy.start_of_pass(None, [bad_req, good_req])
            clock.t = arrival_time + ISOLATED_PREFILL_S
            policy.prepare_during_gpu_execution(
                event_type="prefill",
                running_batch=None,
                waiting_queue=[good_req],
                scheduled_batch=_batch(bad_req),
                prepare_pass_state=False,
            )
            policy.finished_prefill(_batch(bad_req))

            K = 20
            last_task_seq = 0

            for k in range(1, K + 1):
                running = _batch(bad_req)
                policy.start_of_pass(running, [good_req])

                clock.t = arrival_time + ISOLATED_PREFILL_S + k * WALL_DECODE_STEP

                bad_req = _with_output(bad_req, k)
                running = _batch(bad_req)

                last_task_seq = policy._prepare_worker._task_seq + 1
                policy.prepare_during_gpu_execution(
                    event_type="decode",
                    running_batch=running,
                    waiting_queue=[good_req],
                    decode_steps=1,
                    output_ids_already_applied=True,
                    prepare_pass_state=True,
                )
                policy.finished_decode(running)

            got = _wait_for_snapshot(policy, min_task_seq=last_task_seq)
            self.assertTrue(got, "Prepare worker did not publish a snapshot in time.")

            snapshot = policy._prepare_worker.latest_snapshot()
            self.assertIsNotNone(snapshot)

            wall_clock_now = clock.t

            # The isolated next-decode deadline and its start_deadline.
            expected_isolated_deadline = (
                arrival_time + ISOLATED_PREFILL_S + (K + 1) * ISOLATED_DECODE_S
            )
            expected_start_deadline = expected_isolated_deadline - POOLED_DECODE_S

            # Set clock to just past the isolated start_deadline before consuming
            # the snapshot so _compute_safe_prefix_state sees the correct now.
            # Correct: window closed (0 safe tokens).
            # Buggy:   window open  (deadline far in future → unrestricted).
            clock.t = expected_start_deadline + 0.001

            # Consume the snapshot — _compute_safe_prefix_state runs here with
            # clock.t already past the start_deadline, so max_safe_prefill_tokens
            # must come out 0.
            policy._consume_prepared_pass_state([good_req], _batch(bad_req),
                                                allow_waiting_sig_mismatch=True)

            buggy_start_deadline = wall_clock_now + ISOLATED_DECODE_S - POOLED_DECODE_S

            self.assertEqual(
                policy._max_safe_prefill_tokens,
                0,
                f"Safe-prefill window must be CLOSED at now={clock.t:.4f}s "
                f"(just past isolated start_deadline={expected_start_deadline:.4f}s).\n"
                f"If open, the isolated deadline is floating at wall-clock-anchored "
                f"~{buggy_start_deadline:.4f}s instead of the correct isolated value.\n"
                f"Wall clock: {wall_clock_now:.4f}s.",
            )


class TestCompletionDifferenceWorkload(unittest.TestCase):
    """Replicate the completion_difference burst workload.

    Scenario (mirrors what the CSV data shows):
      - N_PREFILL users arrive simultaneously at t=0.
      - Each is prefilled in sequence (one prefill pass each), driving the
        full prepare_during_gpu_execution(prefill) → finished_prefill loop.
      - After all are prefilled, one bad user decodes alone.  Wall clock
        advances at WALL_DECODE_STEP (10× isolated speed).
      - After enough decodes the bad user's isolated deadline is past.
        start_of_pass must then return force_decode=True so the scheduler
        stops admitting new prefills.

    Failure mode being tested:
      If the snapshot is never consumed (start_of_pass_no_prepared every
      time) OR the snapshot has stale timestamps (deadline always in the
      past at prefill time, so max_safe_prefill_tokens=0 before decoding
      even begins), the force_decode signal never fires during decoding and
      the scheduler keeps prefilling indefinitely.
    """

    N_PREFILL = 4   # number of users prefilled before decode epoch

    def _run_scheduler_loop(
        self,
        clock,
        policy,
        *,
        patch_ctx,
    ):
        """
        Simulate: N_PREFILL users arrive and are prefilled one per pass,
        then the bad user decodes K times with wall clock 10× faster than
        isolated.

        We wait for each snapshot BEFORE calling the next start_of_pass so
        the pass immediately consumes it (no 1-second timeout blocking).
        This mirrors the real scheduler: the GPU forward pass provides
        natural time for the worker to finish.

        Returns (force_decode_seen, snapshot_consumed_ever).
        """
        del patch_ctx  # used as context manager outside

        users = [
            _mk_req(f"user_{i}", f"rid_{i}",
                    BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
            for i in range(self.N_PREFILL)
        ]
        bad_req = users[0]

        arrival_time = clock.t

        # --- Arrival: register all requests ---
        for u in users:
            policy.process_new_request(u)

        # --- Prefill epoch: one user per pass ---
        # Mirrors: start_of_pass → GPU prefill →
        #   prepare_during_gpu_execution(prefill) → finished_prefill
        # We wait for the snapshot AFTER prepare so the NEXT start_of_pass
        # finds it immediately.
        waiting = list(users)
        for u in users:
            policy.start_of_pass(None, waiting)
            clock.advance(POOLED_PREFILL_S)
            scheduled = _batch(u)
            task_seq = policy._prepare_worker._task_seq + 1
            policy.prepare_during_gpu_execution(
                event_type="prefill",
                running_batch=None,
                waiting_queue=[w for w in waiting if w is not u],
                scheduled_batch=scheduled,
                prepare_pass_state=True,
            )
            policy.finished_prefill(scheduled)
            waiting = [w for w in waiting if w is not u]
            # Wait for worker to publish snapshot before next start_of_pass.
            _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        # --- Decode epoch: bad user decodes, wall clock 10× faster ---
        K = 25
        force_decode_seen = False
        snapshot_consumed_ever = False

        for k in range(1, K + 1):
            running = _batch(bad_req)
            policy.start_of_pass(running, [])

            source = policy._last_pass_state_source
            # Both "start_of_pass_consume_prepared" (direct) and
            # "start_of_pass_wait_prepare" (after waiting) represent a
            # successful snapshot consumption.
            if "consume" in source or source == "start_of_pass_wait_prepare":
                snapshot_consumed_ever = True
            fd, _ = policy.fairinf_force_decode(running)
            if fd:
                force_decode_seen = True
                break

            # GPU runs decode — wall clock 10× isolated speed.
            clock.t = (
                arrival_time
                + ISOLATED_PREFILL_S
                + k * WALL_DECODE_STEP
            )
            bad_req = _with_output(bad_req, k)
            running = _batch(bad_req)

            task_seq = policy._prepare_worker._task_seq + 1
            policy.prepare_during_gpu_execution(
                event_type="decode",
                running_batch=running,
                waiting_queue=[],
                decode_steps=1,
                output_ids_already_applied=True,
                prepare_pass_state=True,
            )
            policy.finished_decode(running)
            # Wait for snapshot before next start_of_pass.
            _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        return force_decode_seen, snapshot_consumed_ever

    def test_force_decode_fires_after_isolated_deadline_passes(self):
        """After K wall-clock-fast decodes, force_decode must become True.

        If start_of_pass never consumes a snapshot (always no_prepared),
        _has_decode_deadline stays False and force_decode never fires.
        If the snapshot has stale timestamps the deadline stays in the past
        forever and max_safe_prefill_tokens is always 0 at prefill time —
        not the right signal for forcing decode during the decode epoch.
        """
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE_S) as patch_ctx:

            policy = _make_policy()
            force_decode_seen, snapshot_consumed_ever = self._run_scheduler_loop(
                clock, policy, patch_ctx=patch_ctx
            )

        self.assertTrue(
            snapshot_consumed_ever,
            "start_of_pass never consumed a prepared snapshot during the "
            "decode epoch.  The worker is not producing snapshots in time, "
            "so _has_decode_deadline is always False and force_decode never "
            "fires regardless of how far past the isolated deadline the bad "
            "user is."
        )
        self.assertTrue(
            force_decode_seen,
            f"force_decode never became True after {self.N_PREFILL} prefills "
            f"and {25} decode rounds at 10× wall speed.  "
            "The isolated deadline is not being tracked correctly — either "
            "the snapshot timestamps are stale or the deadline window never "
            "closes."
        )


class TestForcedPrefillWithFullMemory(unittest.TestCase):
    """Force-prefill must succeed even when adder.rem_total_tokens <= 0.

    Failure mode from the real workload (pass 649 in scheduler_passes.csv):
      - A bad user has decoded past its isolated deadline → the request is in
        _forced_prefill_rids.
      - Memory is heavily overcommitted: after remove_running_tokens, the
        adder's rem_total_tokens is deeply negative.
      - force_prefill_reservations succeeds (there is enough raw capacity or
        it has already evicted), but expand_capacity is only called when
        retraction was needed.
      - process_waiting_queue_prefills calls adder.add_one_req on the forced
        request, which checks rem_total_tokens and returns False.
      - can_run_list is empty → chosen_event=decode with reason=adder_can_run_list_empty.
      - Waiting queue grows unboundedly while decodes continue.

    Fix: bypass rem_total_tokens for forced prefill requests by setting
    _ignore_global_prefill_budget=True during add_one_req, mirroring the
    pattern used in DeltaFairnessPolicy.force_prefill_reservations.
    """

    def _build_policy_with_forced_prefill(self, clock):
        """Drive the policy to a state where _forced_prefill_rids is non-empty.

        The bad user decodes with wall clock at the same speed as isolated, so
        the decode deadline approaches at a predictable pace.  The good user is
        waiting.  We stop just before the window closes so the good user is in
        _safe_waiting_queue and _forced_prefill_rids (it fits within the
        remaining window).
        """
        policy = _make_policy()

        # Bad user has many decode steps remaining; good user is waiting.
        bad_req = _mk_req("user_bad", "rid_bad", BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
        good_req = _mk_req("user_good", "rid_good", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)

        arrival_time = clock.t
        policy.process_new_request(bad_req)
        policy.process_new_request(good_req)

        # --- Prefill bad user ---
        policy.start_of_pass(None, [bad_req, good_req])
        clock.advance(ISOLATED_PREFILL_S)
        task_seq = policy._prepare_worker._task_seq + 1
        policy.prepare_during_gpu_execution(
            event_type="prefill",
            running_batch=None,
            waiting_queue=[good_req],
            scheduled_batch=_batch(bad_req),
            prepare_pass_state=True,
        )
        policy.finished_prefill(_batch(bad_req))
        _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        # --- Decode bad user a few steps (wall clock == isolated speed) ---
        # After a few steps the deadline is approaching but the good user still
        # fits within the window → _forced_prefill_rids is populated.
        # We use 2 decode steps so the next isolated deadline is at:
        #   arrival + ISOLATED_PREFILL_S + 3 * ISOLATED_DECODE_S
        # and now = arrival + ISOLATED_PREFILL_S + 2 * ISOLATED_DECODE_S
        # which leaves ISOLATED_DECODE_S of window — enough for one prefill
        # (POOLED_PREFILL_S=0.04 < ISOLATED_DECODE_S=0.10).
        K_DECODE = 2
        for k in range(1, K_DECODE + 1):
            running = _batch(bad_req)
            policy.start_of_pass(running, [good_req])
            clock.t = arrival_time + ISOLATED_PREFILL_S + k * ISOLATED_DECODE_S
            bad_req = _with_output(bad_req, k)
            running = _batch(bad_req)
            task_seq = policy._prepare_worker._task_seq + 1
            policy.prepare_during_gpu_execution(
                event_type="decode",
                running_batch=running,
                waiting_queue=[good_req],
                decode_steps=1,
                output_ids_already_applied=True,
                prepare_pass_state=True,
            )
            policy.finished_decode(running)
            _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        # Final start_of_pass with clock just before the next isolated deadline:
        # now = arrival + ISOLATED_PREFILL_S + 2 * ISOLATED_DECODE_S
        # next deadline ≈ arrival + ISOLATED_PREFILL_S + 3 * ISOLATED_DECODE_S
        # Remaining window ≈ ISOLATED_DECODE_S > POOLED_PREFILL_S → good user fits.
        clock.t = arrival_time + ISOLATED_PREFILL_S + K_DECODE * ISOLATED_DECODE_S
        policy.start_of_pass(_batch(bad_req), [good_req])

        return policy, bad_req, good_req

    def test_forced_prefill_bypasses_full_memory_budget(self):
        """Forced prefill must be admitted even when adder.rem_total_tokens <= 0.

        After enough decodes the good user should be in _forced_prefill_rids.
        We then call process_waiting_queue_prefills with an adder whose
        rem_total_tokens is deeply negative (simulating the overcommitted state
        seen in the real workload).  The forced prefill must still reach
        adder.can_run_list.
        """
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S), \
             patch.object(doc_policy_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                          return_value=POOLED_PREFILL_S), \
             patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                          return_value=POOLED_DECODE_S):

            policy, bad_req, good_req = self._build_policy_with_forced_prefill(clock)

            self.assertTrue(
                bool(policy._forced_prefill_rids),
                "Precondition failed: _forced_prefill_rids is empty after decode epoch. "
                "The good user should have been promoted to forced-prefill status once "
                "the bad user's isolated deadline passed.",
            )
            self.assertIn(
                good_req.rid,
                policy._forced_prefill_rids,
                f"Expected good_req ({good_req.rid}) in _forced_prefill_rids, "
                f"got {policy._forced_prefill_rids}.",
            )

            # Build a fake adder that simulates memory overcommit:
            # rem_total_tokens is deeply negative (running batch consumed all
            # capacity and then some, as happens in the real workload at ~85%
            # token usage).
            #
            # In the real workload _has_delta_limit() returns True because the
            # tree cache has fairinf_max_per_user set.  We mock it here to
            # True so process_waiting_queue_prefills reaches the forced-prefill
            # code path (otherwise it delegates directly to super()).
            class _FakeTreeCache:
                """Minimal tree-cache stub: lock ops return 0 delta."""
                def inc_lock_ref(self, node):
                    return 0
                def dec_lock_ref(self, node):
                    return 0

            from sglang.srt.managers.policy_scheduler import PrefillAdder
            from unittest.mock import patch as _patch

            fake_cache = _FakeTreeCache()
            # rem_total_tokens << 0: simulates the real scenario where
            # running_batch tokens consumed all GPU KV budget.
            overcommit_tokens = -50_000
            adder = PrefillAdder(
                tree_cache=fake_cache,
                rem_total_tokens=overcommit_tokens,
                rem_input_tokens=5000,   # input budget still available
                rem_chunk_tokens=None,
                fairness_policy=policy,
            )

            # process_waiting_queue_prefills must bypass the global budget for
            # the forced prefill and add it to can_run_list.
            with _patch.object(policy, "_has_delta_limit", return_value=True):
                policy.process_waiting_queue_prefills(
                    [good_req],
                    adder=adder,
                    token_counters_by_user={},
                    prefix_computed=True,   # skip tree_cache.match_prefix
                    running_batch=_batch(bad_req),
                    running_batch_size=1,
                    max_running_requests=256,
                    available_req_slots=255,
                    max_input_size=None,
                )

            self.assertIn(
                good_req,
                adder.can_run_list,
                "Forced prefill request was not admitted despite memory overcommit. "
                "process_waiting_queue_prefills must bypass rem_total_tokens for "
                "requests in _forced_prefill_rids (same as DeltaFairnessPolicy does "
                "in force_prefill_reservations).",
            )


class TestForcedPrefillSelectivity(unittest.TestCase):
    """process_waiting_queue_prefills must only bypass the global token budget
    for requests that are in _forced_prefill_rids.  Non-forced safe-waiting
    requests must remain budget-gated even when a forced peer is admitted.

    Two invariants tested:
      1. Non-forced request is NOT admitted when rem_total_tokens <= 0.
         (Budget bypass is limited to forced prefills; ordinary safe-waiting
         requests must still obey the global token limit.)
      2. Only the request whose rid is in _forced_prefill_rids is admitted;
         a second waiting request that is NOT in _forced_prefill_rids is
         blocked by the overcommitted budget.
    """

    def _build_policy_with_two_waiting(self, clock):
        """Drive policy to a state with:
          - bad_req: running, has decoded 2 steps, approaching isolated deadline.
          - good_req: waiting, in _forced_prefill_rids (fits within window).
          - extra_req: waiting, NOT in _forced_prefill_rids (window too small).

        We arrange for good_req to be admitted into _forced_prefill_rids by
        giving it a small token count (GOOD_PROMPT_TOKENS = 130), and keep
        extra_req out by making it arrive after the window is almost closed
        (we manually keep it out of the forced set by not registering it with
        the policy — it shows up only in the waiting_queue argument passed to
        process_waiting_queue_prefills).
        """
        policy = _make_policy()
        bad_req = _mk_req("user_bad", "rid_bad", BAD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
        good_req = _mk_req("user_good", "rid_good", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)

        arrival_time = clock.t
        policy.process_new_request(bad_req)
        policy.process_new_request(good_req)

        # Prefill bad user
        policy.start_of_pass(None, [bad_req, good_req])
        clock.advance(ISOLATED_PREFILL_S)
        task_seq = policy._prepare_worker._task_seq + 1
        policy.prepare_during_gpu_execution(
            event_type="prefill",
            running_batch=None,
            waiting_queue=[good_req],
            scheduled_batch=_batch(bad_req),
            prepare_pass_state=True,
        )
        policy.finished_prefill(_batch(bad_req))
        _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        # Decode bad user 2 steps at isolated speed so the window stays open
        K_DECODE = 2
        for k in range(1, K_DECODE + 1):
            running = _batch(bad_req)
            policy.start_of_pass(running, [good_req])
            clock.t = arrival_time + ISOLATED_PREFILL_S + k * ISOLATED_DECODE_S
            bad_req = _with_output(bad_req, k)
            running = _batch(bad_req)
            task_seq = policy._prepare_worker._task_seq + 1
            policy.prepare_during_gpu_execution(
                event_type="decode",
                running_batch=running,
                waiting_queue=[good_req],
                decode_steps=1,
                output_ids_already_applied=True,
                prepare_pass_state=True,
            )
            policy.finished_decode(running)
            _wait_for_snapshot(policy, min_task_seq=task_seq, timeout=5.0)

        clock.t = arrival_time + ISOLATED_PREFILL_S + K_DECODE * ISOLATED_DECODE_S
        policy.start_of_pass(_batch(bad_req), [good_req])

        # extra_req is a request that is in the waiting_queue but was never
        # registered with the policy — it will not appear in _safe_waiting_queue
        # or _forced_prefill_rids.
        extra_req = _mk_req("user_extra", "rid_extra", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)

        return policy, bad_req, good_req, extra_req

    def _make_overcommit_adder(self, policy):
        """Return a PrefillAdder with rem_total_tokens deeply negative."""
        from sglang.srt.managers.policy_scheduler import PrefillAdder

        class _FakeTreeCache:
            def inc_lock_ref(self, node): return 0
            def dec_lock_ref(self, node): return 0

        return PrefillAdder(
            tree_cache=_FakeTreeCache(),
            rem_total_tokens=-50_000,
            rem_input_tokens=5000,
            rem_chunk_tokens=None,
            fairness_policy=policy,
        )

    def _run_process(self, policy, adder, waiting_queue, bad_req):
        """Call process_waiting_queue_prefills with _has_delta_limit mocked."""
        from unittest.mock import patch as _patch
        with _patch.object(policy, "_has_delta_limit", return_value=True):
            policy.process_waiting_queue_prefills(
                waiting_queue,
                adder=adder,
                token_counters_by_user={},
                prefix_computed=True,
                running_batch=_batch(bad_req),
                running_batch_size=1,
                max_running_requests=256,
                available_req_slots=255,
                max_input_size=None,
            )

    def _patches(self, clock):
        return (
            patch.object(sim_mod.time, "time", side_effect=clock),
            patch.object(doc_policy_mod.time, "time", side_effect=clock),
            patch.object(sim_mod, "isolated_decode_time_estimation",
                         return_value=ISOLATED_DECODE_S),
            patch.object(sim_mod, "isolated_prefill_time_estimation",
                         return_value=ISOLATED_PREFILL_S),
            patch.object(doc_policy_mod, "isolated_decode_time_estimation",
                         return_value=ISOLATED_DECODE_S),
            patch.object(doc_policy_mod, "pooled_prefill_time_estimation",
                         return_value=POOLED_PREFILL_S),
            patch.object(doc_policy_mod, "pooled_decode_time_estimation",
                         return_value=POOLED_DECODE_S),
        )

    def test_non_forced_request_blocked_by_full_memory(self):
        """A safe-waiting request that is NOT in _forced_prefill_rids must not
        bypass the global token budget.  When rem_total_tokens is deeply
        negative, only forced requests are admitted; non-forced ones are blocked.

        Scenario: good_req is in _forced_prefill_rids.  extra_req is a second
        waiting request that is NOT registered with the policy (so it is neither
        in _safe_waiting_queue nor in _forced_prefill_rids).  We pass both to
        process_waiting_queue_prefills.  Only good_req should reach can_run_list.
        """
        clock = _FakeClock(start=0.0)
        patches = self._patches(clock)
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            policy, bad_req, good_req, extra_req = self._build_policy_with_two_waiting(clock)

            self.assertIn(
                good_req.rid, policy._forced_prefill_rids,
                "Precondition: good_req must be in _forced_prefill_rids.",
            )
            self.assertNotIn(
                extra_req.rid, policy._forced_prefill_rids,
                "Precondition: extra_req must NOT be in _forced_prefill_rids.",
            )

            adder = self._make_overcommit_adder(policy)
            # Pass both requests; extra_req is not in _safe_waiting_queue so
            # it is ignored by the iteration (which walks _safe_waiting_queue).
            # We verify only good_req ends up in can_run_list.
            self._run_process(policy, adder, [good_req, extra_req], bad_req)

            self.assertIn(
                good_req, adder.can_run_list,
                "Forced request (good_req) must be admitted despite full memory.",
            )
            self.assertNotIn(
                extra_req, adder.can_run_list,
                "Non-forced request (extra_req) must be blocked by overcommitted "
                "rem_total_tokens.  Budget bypass is only for _forced_prefill_rids.",
            )

    def test_only_forced_prefill_rids_bypass_budget(self):
        """If a request is in _safe_waiting_queue but NOT in _forced_prefill_rids,
        it must not bypass the global token budget even when a forced peer is
        being processed in the same pass.

        We use two separate waiting requests:
          - good_req  → in _forced_prefill_rids (budget bypass applies).
          - safe_req  → manually inserted into _safe_waiting_queue but NOT into
                        _forced_prefill_rids (budget gate must still apply).

        With rem_total_tokens << 0, good_req must be admitted and safe_req must
        not, proving the bypass is scoped to the forced set only.
        """
        clock = _FakeClock(start=0.0)
        patches = self._patches(clock)
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            policy, bad_req, good_req, _ = self._build_policy_with_two_waiting(clock)

            self.assertIn(
                good_req.rid, policy._forced_prefill_rids,
                "Precondition: good_req must be in _forced_prefill_rids.",
            )

            # Inject a second waiting request directly into _safe_waiting_queue
            # but keep it out of _forced_prefill_rids — this simulates a client
            # that is eligible for safe prefill but has not missed a deadline yet.
            safe_req = _mk_req("user_safe", "rid_safe", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)
            # _safe_waiting_queue is a tuple/list; we append to the mutable copy.
            existing = list(policy._safe_waiting_queue)
            existing.append(safe_req)
            policy._safe_waiting_queue = existing

            self.assertNotIn(
                safe_req.rid, policy._forced_prefill_rids,
                "Precondition: safe_req must NOT be in _forced_prefill_rids.",
            )

            adder = self._make_overcommit_adder(policy)
            self._run_process(policy, adder, [good_req, safe_req], bad_req)

            self.assertIn(
                good_req, adder.can_run_list,
                "Forced request (good_req) must be admitted despite full memory.",
            )
            self.assertNotIn(
                safe_req, adder.can_run_list,
                "Safe-waiting request (safe_req) must NOT bypass the global "
                "token budget.  Only _forced_prefill_rids members get the bypass.",
            )


class TestIsolatedClockDoesNotRunAhead(unittest.TestCase):
    """Requests arriving while the isolated simulator clock has run ahead of
    real time must receive deadlines anchored to their real arrival time, not
    to the far-future accumulated isolated history end.

    Failure mode from the real workload (req_183/184/185 vs req_226/228/…):
      - user_1 has several requests actively decoding in isolation.
      - The isolated clock accumulates to a far-future end timestamp as those
        decodes are simulated many steps ahead.
      - A new request (req_183-equivalent) arrives with a real timestamp T_arr
        that is BEFORE the current isolated clock end T_iso >> T_arr.
      - _current_history_time() previously returned T_iso, so the new request's
        prefill was stamped at T_iso → deadline ~ T_iso + 32*decode_step.
      - Later-arriving req_226-equivalent arrived when user_1 had NO active
        requests, so the clock correctly reset to its arrival time → deadline
        ~ T_arr_226 + ~0.1s.
      - Result: req_226 had an earlier deadline than req_183 even though req_183
        arrived 29 seconds earlier.

    The scenario where memory is full but the user is still "fair":
      - The live system is at high utilisation (memory full) but user_1 is
        within its fair KV share, so its requests are eligible for forced-
        prefill once their isolated deadline is imminent.
      - If the deadline is stamped 22 hours in the future the request never
        becomes urgent and sits in the waiting queue indefinitely.

    Fix: _current_history_time() now returns the earliest real arrival time
    among waiting states when any waiting state's arrival is before the
    accumulated history end — regardless of whether active states exist.
    """

    def _make_simulator(self):
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            AlternateHistorySimulator,
        )
        return AlternateHistorySimulator(
            fairinf_n=N_USERS,
            max_kv_tokens_per_user=None,  # no KV limit so retraction never fires
            enable_timeline_logging=False,
        )

    def _drive_history(self, sim, req, n_decodes, clock, t0_prefill,
                       waiting_queue=None):
        """Drive the simulator's UserTimeline history forward by feeding
        rebuild_from_real_state with real prefill + decode progress.

        UserTimeline is created lazily via get_live_users, so we call that
        first to ensure the timeline exists before each rebuild.
        """
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            RequestDecodeEvent, RequestPrefillEvent,
        )
        wq = waiting_queue or []
        # Ensure user timeline exists
        sim.get_live_users(_batch(req), wq)
        user_timeline = sim.users[req.uid]
        # Feed prefill
        real_statuses = {
            req.rid: RequestPrefillEvent(req_id=req.rid, end_timestamp=t0_prefill),
        }
        user_timeline.rebuild_from_real_state(real_statuses)
        # Feed each decode step
        for k in range(1, n_decodes + 1):
            t_decode = t0_prefill + k * ISOLATED_DECODE_S
            clock.t = t_decode
            req = _with_output(req, k)
            sim.get_live_users(_batch(req), wq)
            user_timeline = sim.users[req.uid]
            real_statuses = {
                req.rid: RequestDecodeEvent(
                    req_id=req.rid,
                    end_timestamp=t_decode,
                    completion_number=k,
                ),
            }
            user_timeline.rebuild_from_real_state(real_statuses)
        return req

    def test_late_arrival_deadline_anchored_to_arrival_not_history_end(self):
        """req_early arrives, decodes a few steps (building up isolated history),
        then req_late arrives while req_early is still active.  req_late's
        isolated prefill deadline must be close to its real arrival time, NOT
        close to the end of the accumulated isolated history.

        Concretely, with the fake time constants:
          T=0.00  req_early arrives and is prefilled
          T=0.05  prefill done (ISOLATED_PREFILL_S)
          T=0.05–0.35  req_early decodes 3 steps (3 × ISOLATED_DECODE_S)
                        isolated history end ≈ 0.35
          T=0.20  req_late arrives (while req_early still decoding)
          → isolated history end (0.35) >> req_late arrival (0.20)
          → req_late prefill must be stamped close to T=0.20, not T=0.35+

        Memory is full but the user is still fair (no KV limit in the sim),
        so the request is eligible — the only thing that should hold it back
        is its own deadline, which must be close to its arrival.
        """
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            RequestStartEvent, RequestDecodeEvent, RequestPrefillEvent,
        )
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S):

            sim = self._make_simulator()

            # --- req_early: arrives at T=0, prefilled, then decodes 3 steps ---
            clock.t = 0.0
            early = _mk_req("user_1", "rid_early", GOOD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
            sim.process_new_request(early)
            K_EARLY = 3
            early = self._drive_history(
                sim, early, K_EARLY, clock, t0_prefill=ISOLATED_PREFILL_S
            )

            # Isolated history end ≈ ISOLATED_PREFILL_S + K_EARLY * ISOLATED_DECODE_S
            history_end_approx = ISOLATED_PREFILL_S + K_EARLY * ISOLATED_DECODE_S  # 0.35s

            # --- req_late: arrives at T=0.20, WHILE req_early is still active ---
            T_LATE_ARRIVAL = ISOLATED_PREFILL_S + 1.5 * ISOLATED_DECODE_S  # 0.20s
            self.assertLess(
                T_LATE_ARRIVAL, history_end_approx,
                "Precondition: req_late must arrive before the isolated history end.",
            )
            clock.t = T_LATE_ARRIVAL
            late = _mk_req("user_1", "rid_late", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)
            sim.process_new_request(late)

            # Rebuild with req_early still decoding + req_late just arrived.
            # get_live_users ensures both requests are tracked in the user timeline.
            sim.get_live_users(_batch(early), [late])
            user_timeline = sim.users["user_1"]
            real_statuses = {
                early.rid: RequestDecodeEvent(
                    req_id=early.rid,
                    end_timestamp=history_end_approx,
                    completion_number=K_EARLY,
                ),
                late.rid: RequestStartEvent(
                    req_id=late.rid,
                    end_timestamp=T_LATE_ARRIVAL,
                ),
            }
            user_timeline.rebuild_from_real_state(real_statuses)

            tracked_late = sim.requests.get(late.rid)
            self.assertIsNotNone(tracked_late, "rid_late must be tracked")
            anticipated = tracked_late.alternate_history_timeline.anticipated_future_events
            self.assertTrue(anticipated,
                            "rid_late must have an anticipated future event after rebuild.")
            prefill_event = next(
                (e for e in anticipated if isinstance(e, RequestPrefillEvent)), None
            )
            self.assertIsNotNone(prefill_event,
                                 "rid_late must have an anticipated prefill event.")

            # The prefill is scheduled after the isolated decoder advances from
            # the arrival time until early's anticipated decode is recorded.
            # In the worst case that takes ceil((history_end-arrival)/decode)+1
            # decode steps from arrival, then one prefill step.
            # Upper bound: arrival + (history_end - arrival) + 2*decode + prefill
            slack = (history_end_approx - T_LATE_ARRIVAL) + 2 * ISOLATED_DECODE_S + ISOLATED_PREFILL_S
            max_acceptable = T_LATE_ARRIVAL + slack

            # Buggy case (pre-fix): prefill is stamped starting from history_end,
            # not from the arrival, so it ends up >> max_acceptable.
            # Use 2× history_end as the "clearly wrong" threshold.
            buggy_ts = 2 * history_end_approx

            self.assertLess(
                prefill_event.end_timestamp, buggy_ts,
                f"req_late prefill stamped at {prefill_event.end_timestamp:.4f}s "
                f"which is at or beyond the inflated history_end-based value "
                f"({buggy_ts:.4f}s).  _current_history_time must collapse the "
                f"clock to the arrival when waiting arrivals are in the past.",
            )
            self.assertLessEqual(
                prefill_event.end_timestamp, max_acceptable,
                f"req_late prefill stamped at {prefill_event.end_timestamp:.4f}s, "
                f"expected ≤ {max_acceptable:.4f}s "
                f"(arrival {T_LATE_ARRIVAL:.4f}s + bounded drain of isolated history).",
            )

    def test_earlier_arriving_request_gets_earlier_or_equal_deadline(self):
        """When two requests arrive in order, the earlier-arriving one must
        receive an isolated prefill deadline no later than the later-arriving one.

        This is the direct invariant violated by the req_183 vs req_226 bug:
        req_183 arrived first but got a deadline 22 hours later than req_226.

        Scenario (memory full, user still fair):
          - req_running is actively decoding; isolated history end ≈ 0.35s.
          - req_A arrives at T=0.10 (before history end).
          - req_B arrives at T=0.30 (also before history end, but after req_A).
          - req_A must receive an isolated prefill deadline ≤ req_B's deadline.
        """
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            RequestStartEvent, RequestDecodeEvent, RequestPrefillEvent,
        )
        clock = _FakeClock(start=0.0)

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(doc_policy_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S):

            sim = self._make_simulator()

            # Running request decodes 3 steps, building isolated history to 0.35s
            clock.t = 0.0
            running = _mk_req("user_1", "rid_running",
                               GOOD_PROMPT_TOKENS, BAD_MAX_NEW_TOKENS)
            sim.process_new_request(running)
            K = 3
            running = self._drive_history(
                sim, running, K, clock, t0_prefill=ISOLATED_PREFILL_S
            )
            history_end_approx = ISOLATED_PREFILL_S + K * ISOLATED_DECODE_S  # 0.35s

            # req_A arrives early, req_B arrives later — both before history end
            T_A = ISOLATED_PREFILL_S + 0.5 * ISOLATED_DECODE_S   # 0.10s
            T_B = ISOLATED_PREFILL_S + 2.5 * ISOLATED_DECODE_S   # 0.30s
            self.assertLess(T_A, T_B)
            self.assertLess(T_B, history_end_approx)

            clock.t = T_A
            req_A = _mk_req("user_1", "rid_A", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)
            sim.process_new_request(req_A)

            clock.t = T_B
            req_B = _mk_req("user_1", "rid_B", GOOD_PROMPT_TOKENS, GOOD_MAX_NEW_TOKENS)
            sim.process_new_request(req_B)

            sim.get_live_users(_batch(running), [req_A, req_B])
            user_timeline = sim.users["user_1"]
            real_statuses = {
                running.rid: RequestDecodeEvent(
                    req_id=running.rid,
                    end_timestamp=history_end_approx,
                    completion_number=K,
                ),
                req_A.rid: RequestStartEvent(req_id=req_A.rid, end_timestamp=T_A),
                req_B.rid: RequestStartEvent(req_id=req_B.rid, end_timestamp=T_B),
            }
            user_timeline.rebuild_from_real_state(real_statuses)

            def _prefill_ts(rid):
                tracked = sim.requests.get(rid)
                anticipated = tracked.alternate_history_timeline.anticipated_future_events
                ev = next((e for e in anticipated if isinstance(e, RequestPrefillEvent)), None)
                return ev.end_timestamp if ev is not None else float("inf")

            ts_A = _prefill_ts(req_A.rid)
            ts_B = _prefill_ts(req_B.rid)

            self.assertLessEqual(
                ts_A, ts_B,
                f"req_A (arrived T={T_A:.3f}s) got prefill deadline {ts_A:.4f}s "
                f"but req_B (arrived T={T_B:.3f}s) got {ts_B:.4f}s. "
                f"Earlier-arriving requests must not receive later isolated deadlines.",
            )


class TestFinishingRequestKVTracking(unittest.TestCase):
    """Requests that have left request_timelines (mark_request_finished) but
    still have completion tokens left to generate must continue to occupy KV
    memory in the isolated simulator.

    Failure mode without the fix:
      - Bad user finishes prefill; simulator removes it from request_timelines.
      - Isolated scheduler now thinks KV is free.
      - New good-user request is admitted immediately.
      - In reality the bad user still holds KV for its remaining decode tokens,
        so the admission was premature.

    Fix: mark_request_finished records a _FinishingRequest ghost in
    UserTimeline.finishing_requests.  The ghost contributes to
    _current_active_kv_tokens() until its simulated_decode_count reaches
    total_completion_tokens, delaying new prefills correctly.
    """

    def _make_tight_simulator(self, max_kv_tokens: int):
        """Return a simulator with a per-user KV token limit."""
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            AlternateHistorySimulator,
        )
        return AlternateHistorySimulator(
            fairinf_n=1,
            max_kv_tokens_per_user=max_kv_tokens,
            min_new_token_ratio=0.0,
            enable_timeline_logging=False,
        )

    def test_finishing_request_blocks_new_prefill_until_decoded(self):
        """A just-finished request with many remaining completion tokens must
        hold KV memory and delay the next waiting request's prefill deadline.

        Setup:
          - PROMPT_TOKENS = 50, COMPLETION_TOKENS = 20 (total = 70 tokens in KV
            at the peak of decoding).
          - max_kv_tokens = 80 — tight enough that a second 50-prompt request
            cannot be admitted while the finishing request still holds its
            50 + 20 = 70 tokens.
          - finishing_req: mark_request_finished called with 20 completion tokens,
            simulated_decode_count at finish = 0 (none yet consumed).
          - waiting_req: arrives AFTER mark_request_finished, prompt = 50 tokens,
            max_new_tokens = 5.

        Without the fix: the waiting_req's anticipated prefill event would be
        stamped at arrival + ISOLATED_PREFILL_S (KV looks empty → admitted right
        away in the loop).

        With the fix: the anticipated prefill must be stamped AFTER the simulator
        has ticked the finishing ghost through all 20 decode steps (freeing KV).
        The prefill end_timestamp must therefore be strictly greater than
          arrival + ISOLATED_PREFILL_S + 20 * ISOLATED_DECODE_S.
        """
        from sglang.srt.delta_fairness.doc_policy_simulator import (
            RequestStartEvent, RequestPrefillEvent,
        )
        clock = _FakeClock(start=0.0)

        PROMPT_TOKENS = 50
        COMPLETION_TOKENS = 20
        # KV limit: just above prompt+1 (so one active request fits) but below
        # prompt + completion (so the finishing ghost + new prefill don't both fit)
        MAX_KV = PROMPT_TOKENS + COMPLETION_TOKENS + 5  # = 75; new 50-token req fits alone

        with patch.object(sim_mod.time, "time", side_effect=clock), \
             patch.object(sim_mod, "isolated_decode_time_estimation",
                          return_value=ISOLATED_DECODE_S), \
             patch.object(sim_mod, "isolated_prefill_time_estimation",
                          return_value=ISOLATED_PREFILL_S):

            sim = self._make_tight_simulator(MAX_KV)

            # --- Finishing request: arrives, prefills, finishes after 0 decodes ---
            clock.t = 0.0
            finish_req = _mk_req("user_1", "rid_finish",
                                 PROMPT_TOKENS, COMPLETION_TOKENS + 10,
                                 output_tokens=COMPLETION_TOKENS)
            sim.process_new_request(finish_req)
            sim.get_live_users(None, [finish_req])

            # Prefill the finishing request in the simulator so it's "active"
            sim.finished_prefill(_batch(finish_req))

            # Now mark it finished — this records the ghost with 20 completion
            # tokens remaining.
            sim.mark_request_finished(finish_req)

            # Verify the ghost is recorded
            user_timeline = sim.users.get("user_1")
            self.assertIsNotNone(user_timeline,
                                 "user_1 timeline must exist after mark_request_finished.")
            self.assertEqual(
                len(user_timeline.finishing_requests), 1,
                f"Expected 1 finishing ghost, got {len(user_timeline.finishing_requests)}.",
            )
            ghost = user_timeline.finishing_requests[0]
            self.assertEqual(ghost.total_completion_tokens, COMPLETION_TOKENS)
            self.assertFalse(ghost.is_done(),
                             "Ghost must not be done immediately (simulated_decode_count=0).")

            # --- Waiting request: arrives after the finishing request is gone ---
            T_WAIT_ARRIVAL = 0.001
            clock.t = T_WAIT_ARRIVAL
            wait_req = _mk_req("user_1", "rid_wait", PROMPT_TOKENS, 5)
            sim.process_new_request(wait_req)

            # Rebuild — the finishing ghost is in UserTimeline.finishing_requests.
            # The KV budget is: ghost uses PROMPT_TOKENS + 1 = 51 tokens (initial).
            # That leaves only MAX_KV - 51 = 24 tokens, which is < PROMPT_TOKENS=50,
            # so the new prefill cannot be admitted until ghost decodes enough.
            sim.get_live_users(None, [wait_req])
            user_timeline = sim.users["user_1"]
            real_statuses = {
                wait_req.rid: RequestStartEvent(
                    req_id=wait_req.rid, end_timestamp=T_WAIT_ARRIVAL
                ),
            }
            user_timeline.rebuild_from_real_state(real_statuses)

            tracked_wait = sim.requests.get(wait_req.rid)
            self.assertIsNotNone(tracked_wait, "waiting request must be tracked")
            anticipated = tracked_wait.alternate_history_timeline.anticipated_future_events
            self.assertTrue(
                anticipated,
                "waiting request must have an anticipated event after rebuild.",
            )
            prefill_ev = next(
                (e for e in anticipated if isinstance(e, RequestPrefillEvent)), None
            )
            self.assertIsNotNone(
                prefill_ev,
                "waiting request must have an anticipated prefill event.",
            )

            # The prefill must be delayed past the point where the ghost has
            # generated enough tokens to free KV.
            # Ghost starts at decode 0 and needs (PROMPT + completion) = 70 tokens.
            # New req needs 50 tokens.  Budget = 75.
            # Ghost frees KV only once simulated_decode_count ≥ COMPLETION_TOKENS.
            # Each decode step advances the ghost by `rounds` (= 1 here).
            # So at minimum 20 decode steps must pass before the prefill fires.
            # Each step takes ISOLATED_DECODE_S seconds.
            # earliest_possible = T_WAIT_ARRIVAL + 20 * ISOLATED_DECODE_S + ISOLATED_PREFILL_S
            min_delay = COMPLETION_TOKENS * ISOLATED_DECODE_S + ISOLATED_PREFILL_S
            earliest_no_delay = T_WAIT_ARRIVAL + ISOLATED_PREFILL_S  # buggy: admitted right away

            self.assertGreater(
                prefill_ev.end_timestamp,
                earliest_no_delay,
                f"Prefill deadline {prefill_ev.end_timestamp:.4f}s is at or before "
                f"the no-delay baseline {earliest_no_delay:.4f}s.  The finishing "
                f"ghost is not blocking the new prefill as it should.",
            )
            self.assertGreaterEqual(
                prefill_ev.end_timestamp,
                T_WAIT_ARRIVAL + min_delay,
                f"Prefill deadline {prefill_ev.end_timestamp:.4f}s is earlier than "
                f"the minimum delay {T_WAIT_ARRIVAL + min_delay:.4f}s.  "
                f"The ghost ({COMPLETION_TOKENS} decode steps at {ISOLATED_DECODE_S}s each) "
                f"must be fully drained before the new prefill can be admitted.",
            )


if __name__ == "__main__":
    unittest.main()
