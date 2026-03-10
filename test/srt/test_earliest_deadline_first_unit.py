import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.delta_fairness.earliest_deadline_first as edf_mod
from sglang.srt.delta_fairness.earliest_deadline_first import EarliestDeltaFirst
from sglang.srt.managers.schedule_batch import Req


def _mk_req(uid: str, rid: str, n_tokens: int) -> Req:
    req = Req(uid=uid, rid=rid, origin_input_text="", origin_input_ids=[1] * n_tokens)
    req.fill_ids = list(req.origin_input_ids)
    req.output_ids = []
    return req


class TestEarliestDeadlineFirstUnit(unittest.TestCase):
    def test_edf_should_force_decode_when_decode_deadline_is_earlier(self):
        """
        This test intentionally reflects expected EDF behavior:
        if the earliest deadline among tracked requests is a decode, force decode.

        It currently fails because decode candidates can be filtered out by fairness
        checks during deadline construction, leaving only prefill events.
        """

        now = {"t": 100.0}

        def fake_time():
            return now["t"]

        with patch.object(edf_mod.time, "time", side_effect=fake_time):
            # Match experiment-scale max running requests.
            policy = EarliestDeltaFirst(delta_fairness_n=2, max_running_requests=256)
            policy._write_fairinf_log = lambda *args, **kwargs: None

            running_req = _mk_req("u_running", "rid_running", 8)
            waiting_req = _mk_req("u_waiting", "rid_waiting", 8)

            # Build tracked timeline for running request:
            # start -> prefill -> decode(1) with anticipated decode(2).
            policy.process_new_request(running_req)
            now["t"] = 101.0
            policy.finished_prefill(SimpleNamespace(reqs=[running_req]))
            running_req.output_ids = [42]
            now["t"] = 102.0
            policy.finished_decode(SimpleNamespace(reqs=[running_req]))

            # Build tracked timeline for waiting request:
            # start with anticipated prefill at a later time.
            now["t"] = 103.0
            policy.process_new_request(waiting_req)

            running_batch = SimpleNamespace(reqs=[running_req])
            deltas = {"prefill": 0, "first_decode": 0, "decode": 0}

            force_prefill = policy.fairinf_force_prefill_any_waiting(
                [waiting_req],
                running_batch=running_batch,
                delta_fairness_deltas_microseconds=deltas,
            )
            force_decode, _ = policy.fairinf_force_decode(
                running_batch,
                delta_fairness_deltas_microseconds=deltas,
            )

            # Expected EDF behavior for this setup:
            # decode deadline(102.0) < prefill deadline(103.0), so decode should win.
            self.assertFalse(force_prefill)
            self.assertTrue(force_decode)

    def test_rehydrated_waiting_requests_are_chunked_by_runtime_max_prefill_tokens(self):
        now = {"t": 100.0}

        def fake_time():
            return now["t"]

        with patch.object(edf_mod.time, "time", side_effect=fake_time):
            policy = EarliestDeltaFirst(
                delta_fairness_n=1,
                max_running_requests=256,
                max_prefill_tokens=5,
            )
            policy._write_fairinf_log = lambda *args, **kwargs: None

            waiting = [
                _mk_req("u_wait", "rid_wait_1", 4),
                _mk_req("u_wait", "rid_wait_2", 4),
                _mk_req("u_wait", "rid_wait_3", 4),
            ]

            now["t"] = 100.0
            for req in waiting:
                policy.process_new_request(req)

            # Rehydrate the user as fair at a later pass. Since max_prefill_tokens=5,
            # alternate-history prefill scheduling should admit these as three separate
            # prefill batches rather than one large batch.
            now["t"] = 101.0
            policy.start_of_pass(SimpleNamespace(reqs=[]), waiting)

            tracked = [policy.event_queue.requests[req.rid] for req in waiting]
            prefill_events = []
            for tracked_req in tracked:
                req_prefills = [
                    event
                    for event in tracked_req.alternate_history_timeline.history
                    if isinstance(event, edf_mod.RequestPrefillEvent)
                ]
                self.assertEqual(len(req_prefills), 1)
                prefill_events.append(req_prefills[0])

            user_prefills = [
                event
                for event in policy.event_queue.users["u_wait"].history
                if isinstance(event, edf_mod.UserPrefillEvent)
            ]
            self.assertEqual(len(user_prefills), 3)


if __name__ == "__main__":
    unittest.main()
