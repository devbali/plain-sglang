from __future__ import annotations

"""Design.md policy implementation."""

import logging
import multiprocessing as mp
import os
import pickle
import select
import socket
import struct
import time
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from sglang.global_config import global_config
from sglang.srt.request_timeline import TIMELINE_WRITER
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

from .delta_fairness_policy import DeltaFairnessPolicy
from .doc_policy_simulator import (
    AlternateHistorySimulator,
    DeadlineCandidate,
    RequestDecodeEvent,
    RequestEvent,
    RequestStartEvent,
    RequestPrefillEvent,
    TrackedRequest,
)
from .time_estimation import (
    isolated_decode_time_estimation,
    pooled_decode_time_estimation,
    pooled_prefill_time_estimation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedSnapshot:
    task_seq: int
    mutation_seq: int
    waiting_sig: Tuple[str, ...]
    running_sig: Tuple[str, ...]
    deadline_queue: Tuple[object, ...]
    waiting_prefill_deadlines: Dict[str, float]
    safe_waiting_queue: Tuple[Req, ...]
    safe_waiting_rids: frozenset[str]
    forced_prefill_queue: Tuple[Req, ...]
    forced_prefill_rids: frozenset[str]
    max_safe_prefill_tokens: Optional[int]
    has_fair_waiting: bool
    has_decode_deadline: bool
    earliest_decode_start_deadline: Optional[float]
    safe_prefix_now: Optional[float]
    breakdown_items: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class _FrozenPrepareCacheState:
    total_user_tokens: Dict[str, int]
    evictable_user_tokens: Dict[str, int]
    fairinf_max_per_user: Optional[int]
    unevictable_limit: Optional[int]


@dataclass(frozen=True)
class _FrozenPrepareInputs:
    deltas_us: Dict[str, int]
    no_retraction_cap: Optional[int]
    new_token_ratio: float
    fairinf_n: int
    now_s: float


class _DocPolicyPrepareWorker:
    def __init__(
        self,
        owner: "DocPolicy",
        *,
        isolated_kv_tokens_per_user: Optional[int],
        fairinf_n: int,
        min_new_token_ratio: float,
    ) -> None:
        self._owner = owner
        self._simulator = AlternateHistorySimulator(
            max_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=fairinf_n,
            min_new_token_ratio=min_new_token_ratio,
            enable_timeline_logging=False,
        )
        self._published_snapshot: Optional[_PreparedSnapshot] = None
        self._last_applied_mutation_seq = 0
        self._thread_exception: Optional[BaseException] = None
        self._task_seq = 0
        self._mutation_seq = 0
        self._pending_mutation_queue_backpressure_wait_ms = 0.0
        self._pending_prepare_task_queue_backpressure_wait_ms = 0.0
        self._pending_mutation_queue_drain_wait_ms = 0.0
        self._pending_duplicate_state_wait_ms = 0.0
        self._queue_csv_path = os.path.join(os.getcwd(), "doc_policy_prepare_queue.csv")
        self._queue_csv_header_written = False
        (
            self._mutation_parent_sock,
            self._mutation_child_sock,
        ) = socket.socketpair()
        (
            self._task_parent_sock,
            self._task_child_sock,
        ) = socket.socketpair()
        (
            self._status_parent_sock,
            self._status_child_sock,
        ) = socket.socketpair()
        self._status_parent_sock.setblocking(False)
        ctx = mp.get_context("fork")
        self._process = ctx.Process(
            target=self._process_loop,
            name="doc-policy-prepare",
            daemon=True,
        )
        self._process.start()
        self._mutation_child_sock.close()
        self._task_child_sock.close()
        self._status_child_sock.close()
        self._closed = False

    def snapshot_req(self, req: Req) -> dict:
        return req.to_prepare_dict()

    def snapshot_batch(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        if batch is None:
            return None
        return SimpleNamespace(reqs=[self.snapshot_req(req) for req in batch.reqs])

    def _restore_req(self, req_data: dict) -> Req:
        req = Req(
            req_data["uid"],
            req_data["rid"],
            req_data.get("origin_input_text"),
            list(req_data["origin_input_ids"]),
        )
        req.output_ids = list(req_data.get("output_ids", []))
        fill_ids = req_data.get("fill_ids")
        req.fill_ids = None if fill_ids is None else list(fill_ids)
        req.extend_input_len = int(req_data.get("extend_input_len", 0))
        req.prefix_indices = list(req_data.get("prefix_indices", []))
        req.waiting_time_in_decodes = int(req_data.get("waiting_time_in_decodes", 0))
        req.first_time_in_waiting_queue = bool(
            req_data.get("first_time_in_waiting_queue", False)
        )
        max_new_tokens = req_data.get("max_new_tokens")
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens or 0)
        return req

    def _restore_batch(
        self, batch_data: Optional[SimpleNamespace]
    ) -> Optional[SimpleNamespace]:
        if batch_data is None:
            return None
        return SimpleNamespace(reqs=[self._restore_req(req) for req in batch_data.reqs])

    def raise_exception_if_any(self) -> None:
        if self._thread_exception is not None:
            raise RuntimeError("doc policy prepare worker failed") from self._thread_exception

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sock in (
            self._mutation_parent_sock,
            self._task_parent_sock,
            self._status_parent_sock,
        ):
            try:
                sock.close()
            except Exception:
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    def latest_snapshot(self) -> Optional[_PreparedSnapshot]:
        self._drain_status_socket()
        return self._published_snapshot

    def _send_message(self, sock: socket.socket, message) -> None:
        self._validate_socket_payload(message)
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(struct.pack("!I", len(payload)) + payload)

    def _validate_socket_payload(self, obj, path: str = "payload") -> None:
        primitive_types = (type(None), bool, int, float, str, bytes)
        if isinstance(obj, primitive_types):
            return
        if isinstance(obj, tuple):
            for i, item in enumerate(obj):
                self._validate_socket_payload(item, f"{path}[{i}]")
            return
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                self._validate_socket_payload(item, f"{path}[{i}]")
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, primitive_types[:-1]):
                    raise TypeError(
                        f"non-plain socket payload key at {path}: {type(key).__name__}"
                    )
                self._validate_socket_payload(value, f"{path}[{key!r}]")
            return
        if isinstance(obj, (set, frozenset)):
            for i, item in enumerate(obj):
                self._validate_socket_payload(item, f"{path}[set:{i}]")
            return
        if isinstance(obj, SimpleNamespace):
            self._validate_socket_payload(vars(obj), f"{path}.__dict__")
            return
        if is_dataclass(obj):
            for field in fields(obj):
                self._validate_socket_payload(
                    getattr(obj, field.name), f"{path}.{field.name}"
                )
            return
        raise TypeError(
            f"non-plain socket payload at {path}: {type(obj).__module__}.{type(obj).__name__}"
        )

    def _recv_message_blocking(self, sock: socket.socket):
        header = bytearray()
        while len(header) < 4:
            chunk = sock.recv(4 - len(header))
            if not chunk:
                return None
            header.extend(chunk)
        size = struct.unpack("!I", header)[0]
        payload = bytearray()
        while len(payload) < size:
            chunk = sock.recv(size - len(payload))
            if not chunk:
                return None
            payload.extend(chunk)
        return pickle.loads(payload)

    def _recv_message_nonblocking(self, sock: socket.socket):
        try:
            header = sock.recv(4, socket.MSG_PEEK)
        except BlockingIOError:
            return None
        if len(header) < 4:
            return None
        size = struct.unpack("!I", header)[0]
        needed = 4 + size
        try:
            available = sock.recv(needed, socket.MSG_PEEK)
        except BlockingIOError:
            return None
        if len(available) < needed:
            return None
        data = sock.recv(needed)
        return pickle.loads(data[4:])

    def _drain_status_socket(self) -> None:
        while True:
            message = self._recv_message_nonblocking(self._status_parent_sock)
            if message is None:
                break
            kind = message[0]
            if kind == "snapshot":
                snapshot = self._restore_snapshot(message[1])
                self._published_snapshot = snapshot
                self._last_applied_mutation_seq = max(
                    self._last_applied_mutation_seq, snapshot.mutation_seq
                )
                self._owner._simulator_rebuild_prepared = True
                self._owner._last_prepare_breakdown_ms = dict(snapshot.breakdown_items)
            elif kind == "mutation_ack":
                self._last_applied_mutation_seq = max(
                    self._last_applied_mutation_seq, int(message[1])
                )
            elif kind == "error":
                self._thread_exception = RuntimeError(message[1])
            else:
                raise RuntimeError(f"unknown prepare worker status message: {kind}")

    def _log_queue_backpressure(
        self, *, queue_name: str, queue_len: int, wait_ms: float
    ) -> None:
        if queue_name == "mutation":
            self._pending_mutation_queue_backpressure_wait_ms += wait_ms
        else:
            self._pending_prepare_task_queue_backpressure_wait_ms += wait_ms
        line = f"{time.time()},{queue_name},{queue_len},{wait_ms}\n"
        if not self._queue_csv_header_written:
            with open(self._queue_csv_path, "a") as f:
                if f.tell() == 0:
                    f.write("timestamp,queue_name,queue_len,wait_ms\n")
                f.write(line)
            self._queue_csv_header_written = True
            return
        with open(self._queue_csv_path, "a") as f:
            f.write(line)

    def _maybe_wait_for_queue_capacity(
        self, pending_len: int, *, queue_name: str
    ) -> None:
        if pending_len < 100:
            return
        wait_start = time.perf_counter()
        while pending_len >= 100:
            self._drain_status_socket()
            if queue_name == "mutation":
                pending_len = max(0, self._mutation_seq - self._last_applied_mutation_seq)
            else:
                latest_task_seq = self._published_snapshot.task_seq if self._published_snapshot else 0
                pending_len = max(0, self._task_seq - latest_task_seq)
            time.sleep(0.001)
        self._log_queue_backpressure(
            queue_name=queue_name,
            queue_len=pending_len,
            wait_ms=(time.perf_counter() - wait_start) * 1000.0,
        )

    def enqueue_mutation(self, kind: str, payload) -> None:
        self._drain_status_socket()
        self._maybe_wait_for_queue_capacity(
            max(0, self._mutation_seq - self._last_applied_mutation_seq),
            queue_name="mutation",
        )
        self._mutation_seq += 1
        self._send_message(
            self._mutation_parent_sock,
            ("mutation", self._mutation_seq, kind, payload),
        )

    def enqueue_task(self, task: tuple) -> int:
        self._drain_status_socket()
        latest_task_seq = self._published_snapshot.task_seq if self._published_snapshot else 0
        self._maybe_wait_for_queue_capacity(
            max(0, self._task_seq - latest_task_seq),
            queue_name="prepare",
        )
        self._task_seq += 1
        seq = self._task_seq
        self._send_message(self._task_parent_sock, ("task", seq, task))
        return seq

    def wait_for_mutation_queue_below_limit(self, limit: int = 100) -> None:
        self._drain_status_socket()
        if self._mutation_seq - self._last_applied_mutation_seq < limit:
            return
        wait_start = time.perf_counter()
        while self._mutation_seq - self._last_applied_mutation_seq >= limit:
            self.raise_exception_if_any()
            self._drain_status_socket()
            time.sleep(0.001)
        self._pending_mutation_queue_drain_wait_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0

    @property
    def mutation_seq(self) -> int:
        return self._mutation_seq

    def request_live_snapshot(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        last_consumed_snapshot_seq: int,
    ) -> Tuple[int, bool]:
        del waiting_queue, running_batch
        self._drain_status_socket()
        snapshot = self.latest_snapshot()
        if snapshot is not None and snapshot.task_seq > last_consumed_snapshot_seq:
            return snapshot.task_seq, True
        return last_consumed_snapshot_seq, False

    def wait_for_snapshot(
        self,
        *,
        min_task_seq: int,
        record_duplicate_wait: bool = False,
        timeout_s: float = 0.050,
    ) -> bool:
        if min_task_seq <= 0:
            return False
        deadline = time.perf_counter() + timeout_s
        wait_start = time.perf_counter()
        while time.perf_counter() < deadline:
            self.raise_exception_if_any()
            self._drain_status_socket()
            snapshot = self._published_snapshot
            if snapshot is not None and snapshot.task_seq >= min_task_seq:
                if record_duplicate_wait:
                    self._pending_duplicate_state_wait_ms += (
                        time.perf_counter() - wait_start
                    ) * 1000.0
                return True
            time.sleep(0.001)
        if record_duplicate_wait:
            self._pending_duplicate_state_wait_ms += (
                time.perf_counter() - wait_start
            ) * 1000.0
        return False

    def consume_wait_metrics(self) -> Dict[str, float]:
        metrics = {
            "prepare_mutation_queue_backpressure_wait_ms": self._pending_mutation_queue_backpressure_wait_ms,
            "prepare_task_queue_backpressure_wait_ms": self._pending_prepare_task_queue_backpressure_wait_ms,
            "prepare_mutation_queue_drain_wait_ms": self._pending_mutation_queue_drain_wait_ms,
            "prepare_duplicate_state_wait_ms": self._pending_duplicate_state_wait_ms,
        }
        self._pending_mutation_queue_backpressure_wait_ms = 0.0
        self._pending_prepare_task_queue_backpressure_wait_ms = 0.0
        self._pending_mutation_queue_drain_wait_ms = 0.0
        self._pending_duplicate_state_wait_ms = 0.0
        return metrics

    def _apply_mutation(self, kind: str, payload) -> None:
        simulator = self._simulator
        if kind == "process_new_request":
            req, deltas_us = payload
            simulator.process_new_request(self._restore_req(req), deltas_us)
            return
        if kind == "note_retracted_reqs":
            reqs, deltas_us = payload
            for req in reqs:
                simulator.process_new_request(self._restore_req(req), deltas_us)
            return
        if kind == "finished_prefill":
            simulator.finished_prefill(
                SimpleNamespace(reqs=[self._restore_req(req) for req in payload])
            )
            return
        if kind == "finished_decode":
            reqs, decode_rounds = payload
            simulator.finished_decode(
                SimpleNamespace(reqs=[self._restore_req(req) for req in reqs]),
                decode_rounds=decode_rounds,
            )
            return
        if kind == "mark_request_finished":
            simulator.mark_request_finished(self._restore_req(payload))
            return
        if kind == "note_scheduled_prefill_batch":
            return
        raise ValueError(f"unknown prepare mutation kind: {kind}")

    def _serialize_event(self, event: RequestEvent) -> dict:
        event_type = "start"
        extra = {}
        if isinstance(event, RequestDecodeEvent):
            event_type = "decode"
            extra["completion_number"] = int(event.completion_number)
        elif isinstance(event, RequestPrefillEvent):
            event_type = "prefill"
        elif isinstance(event, RequestStartEvent):
            event_type = "start"
        return {
            "event_type": event_type,
            "req_id": event.req_id,
            "duration": float(event.duration),
            "end_timestamp": float(event.end_timestamp),
            **extra,
        }

    def _restore_event(self, event_data: dict) -> RequestEvent:
        event_type = event_data["event_type"]
        kwargs = {
            "req_id": event_data["req_id"],
            "duration": float(event_data.get("duration", 0.0)),
            "end_timestamp": float(event_data["end_timestamp"]),
        }
        if event_type == "decode":
            return RequestDecodeEvent(
                completion_number=int(event_data.get("completion_number", 0)),
                **kwargs,
            )
        if event_type == "prefill":
            return RequestPrefillEvent(**kwargs)
        if event_type == "start":
            return RequestStartEvent(**kwargs)
        raise ValueError(f"unknown serialized event type: {event_type}")

    def _serialize_deadline_candidate(self, candidate: DeadlineCandidate) -> dict:
        return {
            "deadline": float(candidate.deadline),
            "start_deadline": float(candidate.start_deadline),
            "event_type": candidate.event_type,
            "req": self.snapshot_req(candidate.req),
            "event": self._serialize_event(candidate.event),
        }

    def _restore_deadline_candidate(self, candidate_data: dict) -> DeadlineCandidate:
        return DeadlineCandidate(
            deadline=float(candidate_data["deadline"]),
            start_deadline=float(candidate_data["start_deadline"]),
            event_type=candidate_data["event_type"],
            req=self._restore_req(candidate_data["req"]),
            event=self._restore_event(candidate_data["event"]),
        )

    def _serialize_snapshot(self, snapshot: _PreparedSnapshot) -> dict:
        return {
            "task_seq": int(snapshot.task_seq),
            "mutation_seq": int(snapshot.mutation_seq),
            "waiting_sig": list(snapshot.waiting_sig),
            "running_sig": list(snapshot.running_sig),
            "deadline_queue": [
                self._serialize_deadline_candidate(candidate)
                for candidate in snapshot.deadline_queue
            ],
            "waiting_prefill_deadlines": dict(snapshot.waiting_prefill_deadlines),
            "safe_waiting_queue": [
                self.snapshot_req(req) for req in snapshot.safe_waiting_queue
            ],
            "safe_waiting_rids": list(snapshot.safe_waiting_rids),
            "forced_prefill_queue": [
                self.snapshot_req(req) for req in snapshot.forced_prefill_queue
            ],
            "forced_prefill_rids": list(snapshot.forced_prefill_rids),
            "max_safe_prefill_tokens": snapshot.max_safe_prefill_tokens,
            "has_fair_waiting": bool(snapshot.has_fair_waiting),
            "has_decode_deadline": bool(snapshot.has_decode_deadline),
            "earliest_decode_start_deadline": snapshot.earliest_decode_start_deadline,
            "safe_prefix_now": snapshot.safe_prefix_now,
            "breakdown_items": list(snapshot.breakdown_items),
        }

    def _restore_snapshot(self, snapshot_data: dict) -> _PreparedSnapshot:
        return _PreparedSnapshot(
            task_seq=int(snapshot_data["task_seq"]),
            mutation_seq=int(snapshot_data["mutation_seq"]),
            waiting_sig=tuple(snapshot_data["waiting_sig"]),
            running_sig=tuple(snapshot_data["running_sig"]),
            deadline_queue=tuple(
                self._restore_deadline_candidate(candidate)
                for candidate in snapshot_data["deadline_queue"]
            ),
            waiting_prefill_deadlines={
                rid: float(deadline)
                for rid, deadline in snapshot_data["waiting_prefill_deadlines"].items()
            },
            safe_waiting_queue=tuple(
                self._restore_req(req) for req in snapshot_data["safe_waiting_queue"]
            ),
            safe_waiting_rids=frozenset(snapshot_data["safe_waiting_rids"]),
            forced_prefill_queue=tuple(
                self._restore_req(req) for req in snapshot_data["forced_prefill_queue"]
            ),
            forced_prefill_rids=frozenset(snapshot_data["forced_prefill_rids"]),
            max_safe_prefill_tokens=snapshot_data["max_safe_prefill_tokens"],
            has_fair_waiting=bool(snapshot_data["has_fair_waiting"]),
            has_decode_deadline=bool(snapshot_data["has_decode_deadline"]),
            earliest_decode_start_deadline=snapshot_data["earliest_decode_start_deadline"],
            safe_prefix_now=snapshot_data["safe_prefix_now"],
            breakdown_items=tuple(tuple(item) for item in snapshot_data["breakdown_items"]),
        )

    def _process_loop(self) -> None:
        self._mutation_parent_sock.close()
        self._task_parent_sock.close()
        self._status_parent_sock.close()
        mutation_sock = self._mutation_child_sock
        task_sock = self._task_child_sock
        status_sock = self._status_child_sock
        applied_mutation_seq = 0
        while True:
            try:
                readable, _, _ = select.select([mutation_sock, task_sock], [], [], 0.001)
                if not readable:
                    continue
                if mutation_sock in readable:
                    message = self._recv_message_blocking(mutation_sock)
                    if message is None:
                        return
                    _, mutation_seq, kind, payload = message
                    self._apply_mutation(kind, payload)
                    applied_mutation_seq = mutation_seq
                    self._send_message(status_sock, ("mutation_ack", applied_mutation_seq))
                    continue
                if task_sock in readable:
                    message = self._recv_message_blocking(task_sock)
                    if message is None:
                        return
                    _, task_seq, task = message
                (
                    event_type,
                    running_batch,
                        waiting_queue,
                        scheduled_batch,
                        selected_rids,
                        prepare_pass_state,
                        decode_steps,
                        _new_token_ratio,
                    frozen_cache_state,
                    frozen_inputs,
                    _requested_mutation_seq,
                ) = task
                running_batch = self._restore_batch(running_batch)
                scheduled_batch = self._restore_batch(scheduled_batch)
                waiting_queue = [self._restore_req(req) for req in waiting_queue]
                if (
                    event_type == "decode"
                    and running_batch is not None
                    and decode_steps > 0
                ):
                    self._owner._apply_logical_decode_updates(
                        self._simulator,
                        running_batch,
                        selected_rids=selected_rids,
                        decode_steps=decode_steps,
                    )
                if not prepare_pass_state:
                    continue
                if event_type == "prefill" and scheduled_batch is not None:
                    scheduled_rids = {req.rid for req in scheduled_batch.reqs}
                    predicted_waiting = [
                        req for req in waiting_queue if req.rid not in scheduled_rids
                    ]
                    predicted_running = self._owner._predicted_running_batch(
                        running_batch, scheduled_batch
                    )
                else:
                    predicted_waiting = list(waiting_queue)
                    predicted_running = running_batch
                breakdown: Dict[str, float] = {
                    "logical_event_update_ms": 0.0,
                    "rebuild_from_real_state_ms": 0.0,
                    "prepare_during_gpu_execution_total_ms": 0.0,
                }
                snapshot = self._owner._build_prepare_snapshot(
                    self._simulator,
                    predicted_waiting,
                    predicted_running,
                    task_seq=task_seq,
                    mutation_seq=applied_mutation_seq,
                    breakdown=breakdown,
                    frozen_cache_state=frozen_cache_state,
                    frozen_inputs=frozen_inputs,
                )
                self._send_message(status_sock, ("snapshot", self._serialize_snapshot(snapshot)))
            except BaseException as exc:
                try:
                    self._send_message(status_sock, ("error", repr(exc)))
                except Exception:
                    pass
                logger.exception("prepare worker process failed")
                return


class DocPolicy(DeltaFairnessPolicy):
    def __init__(self, *args, **kwargs):
        self._pooled_quanta_us = int(
            kwargs.pop(
                "delta_fairness_pooled_quanta_us",
                kwargs.pop("delta_fairness_quanta_us", 0),
            )
            or 0
        )
        self._exclusive_quanta_us = int(
            kwargs.pop("delta_fairness_exclusive_quanta_us", 0) or 0
        )
        kwargs.pop("max_prefill_tokens", 0)
        isolated_kv_tokens_per_user = kwargs.pop("isolated_kv_tokens_per_user", None)
        schedule_conservativeness = float(kwargs.pop("schedule_conservativeness", 1.0))
        super().__init__(*args, **kwargs)
        min_new_token_ratio = min(
            global_config.base_min_new_token_ratio * schedule_conservativeness,
            1.0,
        )
        self.simulator = AlternateHistorySimulator(
            max_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
        )
        self._deltas_us = {"prefill": 0, "first_decode": 0, "decode": 0}
        self._deadline_queue = []
        self._waiting_prefill_start_deadline_by_rid: Dict[str, float] = {}
        self._safe_waiting_queue: List[Req] = []
        self._safe_waiting_rids: set[str] = set()
        self._forced_prefill_queue: List[Req] = []
        self._forced_prefill_rids: set[str] = set()
        self._max_safe_prefill_tokens: Optional[int] = None
        self._has_fair_waiting = False
        self._has_decode_deadline = False
        self._last_pass_breakdown_ms: Dict[str, float] = {}
        self._simulator_rebuild_prepared = False
        self._earliest_decode_start_deadline: Optional[float] = None
        self._safe_prefix_now: Optional[float] = None
        self._debug_first_waiting_rid: Optional[str] = None
        self._debug_earliest_decode_rid: Optional[str] = None
        self._debug_earliest_decode_uid: Optional[str] = None
        self._debug_first_waiting_prompt_tokens: Optional[int] = None
        self._debug_first_candidate_prefill_ms: Optional[float] = None
        self._debug_first_candidate_residual_slack_ms: Optional[float] = None
        self._prepared_deadline_queue = []
        self._prepared_waiting_prefill_start_deadline_by_rid: Dict[str, float] = {}
        self._prepared_safe_waiting_queue: List[Req] = []
        self._prepared_safe_waiting_rids: set[str] = set()
        self._prepared_forced_prefill_queue: List[Req] = []
        self._prepared_forced_prefill_rids: set[str] = set()
        self._prepared_max_safe_prefill_tokens: Optional[int] = None
        self._prepared_has_fair_waiting = False
        self._prepared_has_decode_deadline = False
        self._prepared_earliest_decode_start_deadline: Optional[float] = None
        self._prepared_safe_prefix_now: Optional[float] = None
        self._prepared_running_sig: Optional[Tuple[str, ...]] = None
        self._prepared_waiting_sig: Optional[Tuple[str, ...]] = None
        self._prepared_pass_state = None
        self._current_pass_running_sig: Optional[Tuple[str, ...]] = None
        self._current_pass_waiting_sig: Optional[Tuple[str, ...]] = None
        self._last_prepare_breakdown_ms: Dict[str, float] = {}
        self._last_pass_state_source = "init"
        self._pending_new_requests: List[Req] = []
        self._pending_finished_rids: set[str] = set()
        self._pending_decoded_reqs: Dict[str, Req] = {}
        self._pending_finished_prefill_reqs: Dict[str, Req] = {}
        self._pending_scheduled_prefill_reqs: Dict[str, Req] = {}
        self._prefill_no_retraction_token_cap: Optional[int] = None
        self._last_consumed_prepare_snapshot_seq = 0
        self._prepare_worker = _DocPolicyPrepareWorker(
            self,
            isolated_kv_tokens_per_user=isolated_kv_tokens_per_user,
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            min_new_token_ratio=min_new_token_ratio,
        )

    def __del__(self):
        prepare_worker = getattr(self, "_prepare_worker", None)
        if prepare_worker is not None:
            try:
                prepare_worker.close()
            except Exception:
                pass

    def _has_pending_mutations(self) -> bool:
        return bool(
            self._pending_new_requests
            or self._pending_finished_rids
            or self._pending_decoded_reqs
            or self._pending_finished_prefill_reqs
            or self._pending_scheduled_prefill_reqs
        )

    def _clear_pending_mutations(self) -> None:
        self._pending_new_requests = []
        self._pending_finished_rids.clear()
        self._pending_decoded_reqs.clear()
        self._pending_finished_prefill_reqs.clear()
        self._pending_scheduled_prefill_reqs.clear()

    def _snapshot_req_for_prepare(self, req: Req) -> Req:
        return self._prepare_worker.snapshot_req(req)

    def _snapshot_batch_for_prepare(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[SimpleNamespace]:
        return self._prepare_worker.snapshot_batch(batch)

    def _enqueue_prepare_mutation(self, kind: str, payload) -> None:
        self._prepare_worker.enqueue_mutation(kind, payload)

    def _enqueue_prepare_task(self, task: tuple) -> int:
        return self._prepare_worker.enqueue_task(task)

    def _raise_prepare_thread_exception_if_any(self) -> None:
        self._prepare_worker.raise_exception_if_any()

    def _request_live_prepare_snapshot(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> Tuple[int, bool]:
        return self._prepare_worker.request_live_snapshot(
            waiting_queue,
            running_batch,
            last_consumed_snapshot_seq=self._last_consumed_prepare_snapshot_seq,
        )

    def _wait_for_matching_prepare_snapshot(
        self,
        *,
        min_task_seq: int,
        record_duplicate_wait: bool = False,
        timeout_s: float = 0.050,
    ) -> bool:
        return self._prepare_worker.wait_for_snapshot(
            min_task_seq=min_task_seq,
            record_duplicate_wait=record_duplicate_wait,
            timeout_s=timeout_s,
        )

    def _read_deltas(self, delta_fairness_deltas_microseconds: Optional[Dict[str, int]]) -> None:
        deltas = delta_fairness_deltas_microseconds or {}
        self._deltas_us = {
            "prefill": deltas.get("prefill", deltas.get("prefill_running_batch", 0)),
            "first_decode": deltas.get(
                "first_decode",
                deltas.get("first_decode_running_batch", deltas.get("decode_running_batch", 0)),
            ),
            "decode": deltas.get("decode", deltas.get("decode_running_batch", 0)),
        }

    def _freeze_prepare_cache_state(self) -> Optional[_FrozenPrepareCacheState]:
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None:
            return None
        total_counters = getattr(tree_cache, "total_user_counters", None)
        evictable_counters = getattr(tree_cache, "evictable_total_user_counters", None)
        total_user_tokens = (
            dict(total_counters.snapshot())
            if total_counters is not None and hasattr(total_counters, "snapshot")
            else {}
        )
        evictable_user_tokens = (
            dict(evictable_counters.snapshot())
            if evictable_counters is not None and hasattr(evictable_counters, "snapshot")
            else {}
        )
        unevictable_limit = None
        if hasattr(tree_cache, "calculate_delta_fair_reservation_size") and hasattr(
            tree_cache, "fairinf_delta_unevictable"
        ):
            unevictable_limit = int(
                tree_cache.calculate_delta_fair_reservation_size(
                    tree_cache.fairinf_delta_unevictable
                )
            )
        return _FrozenPrepareCacheState(
            total_user_tokens=total_user_tokens,
            evictable_user_tokens=evictable_user_tokens,
            fairinf_max_per_user=getattr(tree_cache, "fairinf_max_per_user", None),
            unevictable_limit=unevictable_limit,
        )

    def _freeze_prepare_inputs(
        self, *, new_token_ratio: float = 0.0
    ) -> _FrozenPrepareInputs:
        return _FrozenPrepareInputs(
            deltas_us=dict(self._deltas_us),
            no_retraction_cap=self._prefill_no_retraction_token_cap,
            new_token_ratio=float(new_token_ratio),
            fairinf_n=max(int(self.delta_fairness_n or 1), 1),
            now_s=time.time(),
        )

    def _event_delta_seconds(self, tracked_req: TrackedRequest, event: RequestEvent) -> float:
        req_deltas = tracked_req.deltas_in_microseconds or self._deltas_us
        if isinstance(event, RequestPrefillEvent):
            delta_us = req_deltas.get("prefill", self._deltas_us["prefill"])
        elif isinstance(event, RequestDecodeEvent):
            if event.completion_number <= 1:
                delta_us = req_deltas.get("first_decode", self._deltas_us["first_decode"])
            else:
                delta_us = req_deltas.get("decode", self._deltas_us["decode"])
        else:
            delta_us = 0
        return float(delta_us) / 1_000_000.0

    def _user_is_fair_for_tracking(
        self, user_id: str, running_batch: Optional[ScheduleBatch]
    ) -> bool:
        del running_batch
        return super().user_is_fair_prefill(user_id, running_batch=None)

    def _user_is_fair_prefill_from_frozen(
        self,
        user_id: str,
        *,
        this_user_sum: int = 0,
        frozen_cache_state: Optional[_FrozenPrepareCacheState],
    ) -> bool:
        if not self.delta_fairness_n:
            return False
        if frozen_cache_state is None or frozen_cache_state.unevictable_limit is None:
            return True
        total_tokens = frozen_cache_state.total_user_tokens.get(user_id, 0)
        evictable_tokens = frozen_cache_state.evictable_user_tokens.get(user_id, 0)
        unevictable_used = total_tokens - evictable_tokens + this_user_sum
        return unevictable_used < frozen_cache_state.unevictable_limit

    def user_is_fair_prefill(
        self,
        user_id: str,
        *,
        running_batch: Optional[ScheduleBatch],
        this_user_len: int = 0,
        this_user_sum: int = 0,
    ) -> bool:
        del running_batch
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None or hasattr(
            tree_cache, "user_unevictable_kv_is_under_fair_share_reservation"
        ):
            return super().user_is_fair_prefill(
                user_id,
                running_batch=None,
                this_user_len=this_user_len,
                this_user_sum=this_user_sum,
            )
        original_tree_cache = self.tree_cache
        try:
            self.tree_cache = None
            return super().user_is_fair_prefill(
                user_id,
                running_batch=None,
                this_user_len=this_user_len,
                this_user_sum=this_user_sum,
            )
        finally:
            self.tree_cache = original_tree_cache

    def _force_prefill_within_user_headroom_from_frozen(
        self,
        req: Req,
        *,
        running_batch: Optional[ScheduleBatch],
        pending_prefill_tokens: int = 0,
        frozen_cache_state: Optional[_FrozenPrepareCacheState],
        new_token_ratio: float,
    ) -> bool:
        if frozen_cache_state is None or frozen_cache_state.fairinf_max_per_user is None:
            return True
        total_tokens = frozen_cache_state.total_user_tokens.get(req.uid, 0)
        evictable_tokens = frozen_cache_state.evictable_user_tokens.get(req.uid, 0)
        cached_unevictable_tokens = total_tokens - evictable_tokens
        uncached_running_tokens = 0
        if running_batch is not None and getattr(running_batch, "seq_lens", None) is not None:
            seq_lens_cpu = running_batch.seq_lens.cpu().tolist()
            for i, running_req in enumerate(running_batch.reqs):
                if running_req.uid != req.uid:
                    continue
                uncached_running_tokens += max(
                    0, int(seq_lens_cpu[i]) - len(running_req.prefix_indices)
                )
        ratio = max(0.0, float(new_token_ratio))
        decode_headroom = 0
        if running_batch is not None:
            for running_req in running_batch.reqs:
                if running_req.uid != req.uid:
                    continue
                remaining = max(
                    0,
                    running_req.sampling_params.max_new_tokens
                    - len(running_req.output_ids),
                )
                decode_headroom += int(min(remaining, 4096) * ratio)
        protected_tokens = (
            cached_unevictable_tokens + uncached_running_tokens + pending_prefill_tokens
        )
        return (
            protected_tokens + decode_headroom + req.extend_input_len
            <= frozen_cache_state.fairinf_max_per_user
        )

    def _force_prefill_within_user_headroom(
        self,
        req: Req,
        *,
        running_batch: Optional[ScheduleBatch],
        pending_prefill_tokens: int = 0,
    ) -> bool:
        return super()._force_prefill_within_user_headroom(
            req,
            running_batch=running_batch,
            pending_prefill_tokens=pending_prefill_tokens,
        )

    def _memory_pressure_active_for_prefill(self) -> bool:
        return self._prefill_no_retraction_token_cap is not None

    def _no_retraction_prefill_token_cap(
        self, running_batch: Optional[ScheduleBatch]
    ) -> Optional[int]:
        del running_batch
        return self._prefill_no_retraction_token_cap

    def _prefill_deadline_waiting_queue(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> List[Req]:
        if not self._memory_pressure_active_for_prefill():
            return waiting_queue
        return [
            req
            for req in waiting_queue
            if self.user_is_fair_prefill(
                req.uid,
                running_batch=running_batch,
                this_user_sum=req.get_estimated_prefill_impact(),
            )
        ]

    def _pooled_prefill_seconds(self, reqs: List[Req]) -> float:
        if not reqs:
            return 0.0
        prompt_tokens = [len(req.origin_input_ids) for req in reqs]
        return pooled_prefill_time_estimation(
            sum(prompt_tokens),
            max(prompt_tokens),
            len(prompt_tokens),
            max(int(self.delta_fairness_n or 1), 1),
        )

    def _current_pooled_decode_seconds(
        self, running_batch: Optional[ScheduleBatch]
    ) -> float:
        if running_batch is None or not running_batch.reqs:
            return 0.0
        token_counts = [
            len(req.fill_ids) if req.fill_ids is not None else len(req.origin_input_ids) + len(req.output_ids)
            for req in running_batch.reqs
        ]
        decode_s = pooled_decode_time_estimation(
            sum(token_counts),
            max(token_counts),
            len(token_counts),
            max(int(self.delta_fairness_n or 1), 1),
        )
        return decode_s

    def _compute_safe_prefix_state(
        self,
        deadline_queue,
        waiting_prefill_start_deadline_by_rid: Dict[str, float],
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        ordered_waiting_queue: Optional[Tuple[Req, ...]] = None,
        frozen_cache_state: Optional[_FrozenPrepareCacheState] = None,
        frozen_inputs: Optional[_FrozenPrepareInputs] = None,
    ) -> Dict[str, object]:
        if ordered_waiting_queue is None:
            indexed = list(enumerate(waiting_queue))
            indexed.sort(
                key=lambda item: (
                    0 if item[1].rid in waiting_prefill_start_deadline_by_rid else 1,
                    waiting_prefill_start_deadline_by_rid.get(item[1].rid, float("inf")),
                    item[0],
                )
            )
            safe_waiting_queue = [req for _, req in indexed]
        else:
            safe_waiting_queue = list(ordered_waiting_queue)
        safe_waiting_rids = {req.rid for req in safe_waiting_queue}
        forced_prefill_queue: List[Req] = []
        forced_prefill_rids: set[str] = set()
        max_safe_prefill_tokens: Optional[int] = None
        has_fair_waiting = bool(safe_waiting_queue)
        has_decode_deadline = False
        earliest_decode_start_deadline: Optional[float] = None
        safe_prefix_now: Optional[float] = None

        earliest_decode_candidate = min(
            (
                candidate
                for candidate in deadline_queue
                if candidate.event_type == "decode"
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        if earliest_decode_candidate is None:
            return {
                "safe_waiting_queue": safe_waiting_queue,
                "safe_waiting_rids": safe_waiting_rids,
                "forced_prefill_queue": forced_prefill_queue,
                "forced_prefill_rids": forced_prefill_rids,
                "max_safe_prefill_tokens": max_safe_prefill_tokens,
                "has_fair_waiting": has_fair_waiting,
                "has_decode_deadline": has_decode_deadline,
                "earliest_decode_start_deadline": earliest_decode_start_deadline,
                "safe_prefix_now": safe_prefix_now,
            }

        has_decode_deadline = True
        earliest_decode_start_deadline = earliest_decode_candidate.start_deadline
        now = frozen_inputs.now_s if frozen_inputs is not None else time.time()
        safe_prefix_now = now
        selected_batch: List[Req] = []
        safe_prompt_tokens = 0
        no_retraction_cap = (
            frozen_inputs.no_retraction_cap
            if frozen_inputs is not None
            else self._no_retraction_prefill_token_cap(running_batch)
        )
        pending_prefill_sum_by_user: Dict[str, int] = {}
        pending_prefill_len_by_user: Dict[str, int] = {}
        fair_user_by_uid: Dict[str, bool] = {}
        for req in safe_waiting_queue:
            req_prefill_tokens = len(req.origin_input_ids)
            next_safe_prompt_tokens = safe_prompt_tokens + req_prefill_tokens
            beyond_no_retraction_cap = (
                no_retraction_cap is not None
                and next_safe_prompt_tokens > no_retraction_cap
            )
            if beyond_no_retraction_cap:
                is_fair = fair_user_by_uid.get(req.uid)
                if is_fair is None:
                    this_user_sum = pending_prefill_sum_by_user.get(req.uid, 0)
                    is_fair = self._user_is_fair_prefill_from_frozen(
                        req.uid,
                        this_user_sum=req.get_estimated_prefill_impact() + this_user_sum,
                        frozen_cache_state=frozen_cache_state,
                    )
                    fair_user_by_uid[req.uid] = is_fair
                if not is_fair:
                    continue
                if not self._force_prefill_within_user_headroom_from_frozen(
                    req,
                    running_batch=running_batch,
                    pending_prefill_tokens=pending_prefill_sum_by_user.get(req.uid, 0),
                    frozen_cache_state=frozen_cache_state,
                    new_token_ratio=(
                        frozen_inputs.new_token_ratio if frozen_inputs is not None else 0.0
                    ),
                ):
                    continue
            candidate_batch = selected_batch + [req]
            prompt_tokens = [len(item.origin_input_ids) for item in candidate_batch]
            pooled_prefill_s = pooled_prefill_time_estimation(
                sum(prompt_tokens),
                max(prompt_tokens),
                len(prompt_tokens),
                (
                    frozen_inputs.fairinf_n
                    if frozen_inputs is not None
                    else max(int(self.delta_fairness_n or 1), 1)
                ),
            )
            if now + pooled_prefill_s <= earliest_decode_start_deadline:
                safe_prompt_tokens = next_safe_prompt_tokens
                selected_batch.append(req)
                forced_prefill_queue.append(req)
                forced_prefill_rids.add(req.rid)
                pending_prefill_sum_by_user[req.uid] = (
                    pending_prefill_sum_by_user.get(req.uid, 0) + req_prefill_tokens
                )
                pending_prefill_len_by_user[req.uid] = (
                    pending_prefill_len_by_user.get(req.uid, 0) + 1
                )
                continue
            break

        max_safe_prefill_tokens = safe_prompt_tokens
        return {
            "safe_waiting_queue": safe_waiting_queue,
            "safe_waiting_rids": safe_waiting_rids,
            "forced_prefill_queue": forced_prefill_queue,
            "forced_prefill_rids": forced_prefill_rids,
            "max_safe_prefill_tokens": max_safe_prefill_tokens,
            "has_fair_waiting": has_fair_waiting,
            "has_decode_deadline": has_decode_deadline,
            "earliest_decode_start_deadline": earliest_decode_start_deadline,
            "safe_prefix_now": safe_prefix_now,
        }

    def _recompute_safe_prefix(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        phase_start = time.perf_counter()
        indexed = list(enumerate(waiting_queue))
        indexed.sort(
            key=lambda item: (
                0
                if item[1].rid in self._waiting_prefill_start_deadline_by_rid
                else 1,
                self._waiting_prefill_start_deadline_by_rid.get(
                    item[1].rid, float("inf")
                ),
                item[0],
            )
        )
        after_waiting_sort = time.perf_counter()
        self._safe_waiting_queue = [req for _, req in indexed]
        self._safe_waiting_rids = {req.rid for req in self._safe_waiting_queue}
        self._has_fair_waiting = bool(self._safe_waiting_queue)
        self._forced_prefill_queue = []
        self._max_safe_prefill_tokens = None
        self._forced_prefill_rids = set()
        self._has_decode_deadline = False
        self._earliest_decode_start_deadline = None
        self._safe_prefix_now = None
        self._debug_first_waiting_rid = None
        self._debug_earliest_decode_rid = None
        self._debug_earliest_decode_uid = None
        self._debug_first_waiting_prompt_tokens = None
        self._debug_first_candidate_prefill_ms = None
        self._debug_first_candidate_residual_slack_ms = None

        earliest_decode_candidate = min(
            (
                candidate
                for candidate in self._deadline_queue
                if candidate.event_type == "decode"
            ),
            key=lambda candidate: candidate.start_deadline,
            default=None,
        )
        if earliest_decode_candidate is None:
            if timing_breakdown is not None:
                timing_breakdown["sort_waiting_prefills_ms"] = (
                    after_waiting_sort - phase_start
                ) * 1000.0
                timing_breakdown["safe_prefix_scan_ms"] = 0.0
                timing_breakdown["build_pass_state_ms"] = (
                    after_waiting_sort - phase_start
                ) * 1000.0
            return
        self._has_decode_deadline = True
        earliest_decode_deadline = earliest_decode_candidate.start_deadline
        self._earliest_decode_start_deadline = earliest_decode_deadline
        self._debug_earliest_decode_rid = earliest_decode_candidate.req.rid
        self._debug_earliest_decode_uid = earliest_decode_candidate.req.uid

        now = time.time()
        self._safe_prefix_now = now
        selected_batch: List[Req] = []
        safe_prompt_tokens = 0
        no_retraction_cap = self._no_retraction_prefill_token_cap(running_batch)
        pending_prefill_sum_by_user: Dict[str, int] = {}
        pending_prefill_len_by_user: Dict[str, int] = {}
        for req in self._safe_waiting_queue:
            req_prefill_tokens = len(req.origin_input_ids)
            next_safe_prompt_tokens = safe_prompt_tokens + req_prefill_tokens
            beyond_no_retraction_cap = (
                no_retraction_cap is not None
                and next_safe_prompt_tokens > no_retraction_cap
            )
            if beyond_no_retraction_cap:
                this_user_sum = pending_prefill_sum_by_user.get(req.uid, 0)
                this_user_len = pending_prefill_len_by_user.get(req.uid, 0)
                if not self.user_is_fair_prefill(
                    req.uid,
                    running_batch=running_batch,
                    this_user_len=this_user_len,
                    this_user_sum=this_user_sum,
                ):
                    continue
                if not self._force_prefill_within_user_headroom(
                    req,
                    running_batch=running_batch,
                    pending_prefill_tokens=this_user_sum,
                ):
                    continue
            candidate_batch = selected_batch + [req]
            pooled_prefill_s = self._pooled_prefill_seconds(candidate_batch)
            if len(selected_batch) == 0:
                self._debug_first_waiting_rid = req.rid
                self._debug_first_waiting_prompt_tokens = len(req.origin_input_ids)
                self._debug_first_candidate_prefill_ms = pooled_prefill_s * 1000.0
                self._debug_first_candidate_residual_slack_ms = (
                    earliest_decode_deadline - (now + pooled_prefill_s)
                ) * 1000.0
            if now + pooled_prefill_s <= earliest_decode_deadline:
                safe_prompt_tokens = next_safe_prompt_tokens
                selected_batch.append(req)
                self._forced_prefill_queue.append(req)
                self._forced_prefill_rids.add(req.rid)
                pending_prefill_sum_by_user[req.uid] = (
                    pending_prefill_sum_by_user.get(req.uid, 0) + req_prefill_tokens
                )
                pending_prefill_len_by_user[req.uid] = (
                    pending_prefill_len_by_user.get(req.uid, 0) + 1
                )
                continue
            break

        self._max_safe_prefill_tokens = safe_prompt_tokens
        after_safe_scan = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["sort_waiting_prefills_ms"] = (
                after_waiting_sort - phase_start
            ) * 1000.0
            timing_breakdown["safe_prefix_scan_ms"] = (
                after_safe_scan - after_waiting_sort
            ) * 1000.0
            timing_breakdown["build_pass_state_ms"] = (
                after_safe_scan - phase_start
            ) * 1000.0

    def _build_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]],
        timing_breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        phase_start = time.perf_counter()
        self._read_deltas(delta_fairness_deltas_microseconds)
        self.simulator.sync_live_user_tracking(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=self._deltas_us,
        )
        deadline_waiting_queue = self._prefill_deadline_waiting_queue(waiting_queue, running_batch)
        deadline_result = self.simulator.build_deadline_candidates(
                deadline_waiting_queue,
                running_batch,
                include_ordered_waiting_queue=True,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
                event_delta_seconds=self._event_delta_seconds,
                pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                    rb
                ),
            )
        if len(deadline_result) == 3:
            (
                self._deadline_queue,
                self._waiting_prefill_start_deadline_by_rid,
                ordered_waiting_queue,
            ) = deadline_result
        else:
            self._deadline_queue, self._waiting_prefill_start_deadline_by_rid = deadline_result
            ordered_waiting_queue = None
        after_deadline_build = time.perf_counter()
        if timing_breakdown is not None:
            timing_breakdown["build_deadline_candidates_ms"] = (
                after_deadline_build - phase_start
            ) * 1000.0
        self._recompute_safe_prefix(waiting_queue, running_batch, timing_breakdown=timing_breakdown)

    def _queue_sig(self, reqs: List[Req]) -> Tuple[str, ...]:
        return tuple(req.rid for req in reqs)

    def _queue_sig_excluding_pending_new(self, reqs: List[Req]) -> Tuple[str, ...]:
        pending_new_rids = {req.rid for req in self._pending_new_requests}
        if not pending_new_rids:
            return self._queue_sig(reqs)
        return tuple(req.rid for req in reqs if req.rid not in pending_new_rids)

    def _running_sig(self, running_batch: Optional[ScheduleBatch]) -> Tuple[str, ...]:
        if running_batch is None:
            return ()
        return tuple(req.rid for req in running_batch.reqs)

    def _predicted_decode_running_batch(
        self,
        running_batch: ScheduleBatch,
        selected_rids: Optional[set[str]],
    ) -> SimpleNamespace:
        predicted_reqs = []
        for req in running_batch.reqs:
            predicted_req = copy(req)
            predicted_req.output_ids = list(req.output_ids)
            if selected_rids is None or req.rid in selected_rids:
                predicted_req.output_ids = list(req.output_ids) + [0]
            predicted_reqs.append(predicted_req)
        return SimpleNamespace(reqs=predicted_reqs)

    def _prepare_deadline_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> None:
        target_simulator = self.simulator if simulator is None else simulator
        target_simulator.sync_live_user_tracking(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=self._deltas_us,
        )
        deadline_waiting_queue = self._prefill_deadline_waiting_queue(waiting_queue, running_batch)
        deadline_result = target_simulator.build_deadline_candidates(
                deadline_waiting_queue,
                running_batch,
                include_ordered_waiting_queue=True,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
                event_delta_seconds=self._event_delta_seconds,
                pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                    rb
                ),
            )
        if len(deadline_result) == 3:
            (
                self._prepared_deadline_queue,
                self._prepared_waiting_prefill_start_deadline_by_rid,
                _prepared_ordered_waiting_queue,
            ) = deadline_result
        else:
            (
                self._prepared_deadline_queue,
                self._prepared_waiting_prefill_start_deadline_by_rid,
            ) = deadline_result
        self._prepared_waiting_sig = self._queue_sig(waiting_queue)
        self._prepared_running_sig = self._running_sig(running_batch)
        old_deadline_queue = self._deadline_queue
        old_waiting_deadlines = self._waiting_prefill_start_deadline_by_rid
        old_safe_waiting_queue = self._safe_waiting_queue
        old_safe_waiting_rids = self._safe_waiting_rids
        old_forced_prefill_queue = self._forced_prefill_queue
        old_forced_prefill_rids = self._forced_prefill_rids
        old_max_safe_prefill_tokens = self._max_safe_prefill_tokens
        old_has_fair_waiting = self._has_fair_waiting
        old_has_decode_deadline = self._has_decode_deadline
        old_earliest_decode_start_deadline = self._earliest_decode_start_deadline
        old_safe_prefix_now = self._safe_prefix_now
        try:
            self._deadline_queue = list(self._prepared_deadline_queue)
            self._waiting_prefill_start_deadline_by_rid = dict(
                self._prepared_waiting_prefill_start_deadline_by_rid
            )
            self._recompute_safe_prefix(waiting_queue, running_batch)
            self._prepared_safe_waiting_queue = list(self._safe_waiting_queue)
            self._prepared_safe_waiting_rids = set(self._safe_waiting_rids)
            self._prepared_forced_prefill_queue = list(self._forced_prefill_queue)
            self._prepared_forced_prefill_rids = set(self._forced_prefill_rids)
            self._prepared_max_safe_prefill_tokens = self._max_safe_prefill_tokens
            self._prepared_has_fair_waiting = self._has_fair_waiting
            self._prepared_has_decode_deadline = self._has_decode_deadline
            self._prepared_earliest_decode_start_deadline = (
                self._earliest_decode_start_deadline
            )
            self._prepared_safe_prefix_now = self._safe_prefix_now
            self._prepared_pass_state = True
        finally:
            self._deadline_queue = old_deadline_queue
            self._waiting_prefill_start_deadline_by_rid = old_waiting_deadlines
            self._safe_waiting_queue = old_safe_waiting_queue
            self._safe_waiting_rids = old_safe_waiting_rids
            self._forced_prefill_queue = old_forced_prefill_queue
            self._forced_prefill_rids = old_forced_prefill_rids
            self._max_safe_prefill_tokens = old_max_safe_prefill_tokens
            self._has_fair_waiting = old_has_fair_waiting
            self._has_decode_deadline = old_has_decode_deadline
            self._earliest_decode_start_deadline = old_earliest_decode_start_deadline
            self._safe_prefix_now = old_safe_prefix_now

    def _build_prepare_snapshot(
        self,
        simulator: AlternateHistorySimulator,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        *,
        task_seq: int,
        mutation_seq: int,
        breakdown: Optional[Dict[str, float]] = None,
        frozen_cache_state: Optional[_FrozenPrepareCacheState] = None,
        frozen_inputs: Optional[_FrozenPrepareInputs] = None,
    ) -> _PreparedSnapshot:
        target_breakdown: Dict[str, float] = {} if breakdown is None else breakdown
        build_start = time.perf_counter()
        simulator.sync_live_user_tracking(
            running_batch,
            waiting_queue,
            deltas_in_microseconds=(
                frozen_inputs.deltas_us if frozen_inputs is not None else self._deltas_us
            ),
        )
        after_sync = time.perf_counter()
        if frozen_inputs is not None and frozen_inputs.no_retraction_cap is not None:
            deadline_waiting_queue = [
                req
                for req in waiting_queue
                if self._user_is_fair_prefill_from_frozen(
                    req.uid,
                    this_user_sum=req.get_estimated_prefill_impact(),
                    frozen_cache_state=frozen_cache_state,
                )
            ]
        else:
            deadline_waiting_queue = waiting_queue
        fairinf_n = (
            frozen_inputs.fairinf_n
            if frozen_inputs is not None
            else max(int(self.delta_fairness_n or 1), 1)
        )
        deadline_result = simulator.build_deadline_candidates(
            deadline_waiting_queue,
            running_batch,
            include_ordered_waiting_queue=True,
            req_is_fair_prefill=lambda req, rb: self._user_is_fair_prefill_from_frozen(
                req.uid,
                this_user_sum=req.get_estimated_prefill_impact(),
                frozen_cache_state=frozen_cache_state,
            ),
            req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                req, running_batch=rb
            ),
            event_delta_seconds=lambda tracked_req, event: float(
                (
                    tracked_req.deltas_in_microseconds
                    or (
                        frozen_inputs.deltas_us
                        if frozen_inputs is not None
                        else self._deltas_us
                    )
                ).get(
                    "prefill"
                    if isinstance(event, RequestPrefillEvent)
                    else (
                        "first_decode"
                        if isinstance(event, RequestDecodeEvent)
                        and event.completion_number <= 1
                        else "decode"
                    ),
                    0,
                )
            )
            / 1_000_000.0,
            pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                len(req.origin_input_ids),
                len(req.origin_input_ids),
                1,
                fairinf_n,
            ),
            pooled_decode_estimate_seconds=lambda req, rb: (
                0.0
                if rb is None or not rb.reqs
                else pooled_decode_time_estimation(
                    sum(
                        len(item.fill_ids)
                        if item.fill_ids is not None
                        else len(item.origin_input_ids) + len(item.output_ids)
                        for item in rb.reqs
                    ),
                    max(
                        len(item.fill_ids)
                        if item.fill_ids is not None
                        else len(item.origin_input_ids) + len(item.output_ids)
                        for item in rb.reqs
                    ),
                    len(rb.reqs),
                    fairinf_n,
                )
            ),
        )
        if len(deadline_result) == 3:
            (
                deadline_queue,
                waiting_prefill_deadline_by_rid,
                ordered_waiting_queue,
            ) = deadline_result
        else:
            deadline_queue, waiting_prefill_deadline_by_rid = deadline_result
            ordered_waiting_queue = None
        after_deadline = time.perf_counter()
        safe_state = self._compute_safe_prefix_state(
            deadline_queue,
            waiting_prefill_deadline_by_rid,
            waiting_queue,
            running_batch,
            ordered_waiting_queue=ordered_waiting_queue,
            frozen_cache_state=frozen_cache_state,
            frozen_inputs=frozen_inputs,
        )
        after_safe = time.perf_counter()
        target_breakdown["sync_live_user_tracking_ms"] = (
            after_sync - build_start
        ) * 1000.0
        target_breakdown["build_deadline_candidates_ms"] = (
            after_deadline - after_sync
        ) * 1000.0
        target_breakdown["sort_waiting_prefills_ms"] = 0.0
        target_breakdown["safe_prefix_scan_ms"] = (
            after_safe - after_deadline
        ) * 1000.0
        target_breakdown["build_pass_state_ms"] = (
            after_safe - build_start
        ) * 1000.0
        return _PreparedSnapshot(
            task_seq=task_seq,
            mutation_seq=mutation_seq,
            waiting_sig=self._queue_sig(waiting_queue),
            running_sig=self._running_sig(running_batch),
            deadline_queue=tuple(deadline_queue),
            waiting_prefill_deadlines=dict(waiting_prefill_deadline_by_rid),
            safe_waiting_queue=tuple(safe_state["safe_waiting_queue"]),
            safe_waiting_rids=frozenset(safe_state["safe_waiting_rids"]),
            forced_prefill_queue=tuple(safe_state["forced_prefill_queue"]),
            forced_prefill_rids=frozenset(safe_state["forced_prefill_rids"]),
            max_safe_prefill_tokens=safe_state["max_safe_prefill_tokens"],
            has_fair_waiting=bool(safe_state["has_fair_waiting"]),
            has_decode_deadline=bool(safe_state["has_decode_deadline"]),
            earliest_decode_start_deadline=safe_state["earliest_decode_start_deadline"],
            safe_prefix_now=safe_state["safe_prefix_now"],
            breakdown_items=tuple(target_breakdown.items()),
        )

    def consume_prepare_thread_wait_metrics(self) -> Dict[str, float]:
        return self._prepare_worker.consume_wait_metrics()

    def _decode_candidate_for_req(
        self,
        req: Req,
        running_batch: Optional[ScheduleBatch],
    ):
        tracked = self.simulator.requests.get(req.rid)
        real_event = self.simulator.most_recent_event_real.get(req.rid)
        if tracked is None or real_event is None:
            return None
        if not isinstance(real_event, (RequestPrefillEvent, RequestDecodeEvent)):
            return None
        event = None
        for upcoming in tracked.earliest_events_after_real_time(real_event) or []:
            if isinstance(upcoming, RequestDecodeEvent):
                event = upcoming
                break
        if event is None:
            token_count = len(req.origin_input_ids) + max(1, len(req.output_ids))
            event = RequestDecodeEvent(
                req_id=req.rid,
                end_timestamp=real_event.end_timestamp
                + isolated_decode_time_estimation(
                    token_count,
                    token_count,
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                completion_number=len(req.output_ids) + 1,
            )
        deadline = event.end_timestamp + self._event_delta_seconds(tracked, event)
        return SimpleNamespace(
            deadline=deadline,
            start_deadline=deadline - self._current_pooled_decode_seconds(running_batch),
            event_type="decode",
            req=req,
            event=event,
        )

    def _merge_pending_pass_state_mutations(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> None:
        waiting_rids = {req.rid for req in waiting_queue}
        running_rids = (
            set() if running_batch is None else {req.rid for req in running_batch.reqs}
        )
        finished_rids = set(self._pending_finished_rids)
        scheduled_prefill_rids = set(self._pending_scheduled_prefill_reqs)
        self._deadline_queue = [
            candidate
            for candidate in self._deadline_queue
            if (
                candidate.event_type != "prefill"
                or candidate.req.rid in waiting_rids
            )
            and (
                candidate.event_type != "decode"
                or candidate.req.rid in running_rids
            )
            if candidate.req.rid not in finished_rids
            and candidate.req.rid not in scheduled_prefill_rids
            and not (
                candidate.event_type == "decode"
                and candidate.req.rid in self._pending_decoded_reqs
            )
            and not (
                candidate.event_type == "decode"
                and candidate.req.rid in self._pending_finished_prefill_reqs
            )
        ]
        self._waiting_prefill_start_deadline_by_rid = {
            rid: deadline
            for rid, deadline in self._waiting_prefill_start_deadline_by_rid.items()
            if rid in waiting_rids
            and rid not in finished_rids
            and rid not in scheduled_prefill_rids
        }
        if self._pending_new_requests:
            deadline_result = self.simulator.build_deadline_candidates(
                self._pending_new_requests,
                running_batch,
                req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                    req.uid,
                    running_batch=rb,
                    this_user_sum=req.get_estimated_prefill_impact(),
                ),
                req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                    req, running_batch=rb
                ),
                event_delta_seconds=self._event_delta_seconds,
                pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                    len(req.origin_input_ids),
                    len(req.origin_input_ids),
                    1,
                    max(int(self.delta_fairness_n or 1), 1),
                ),
                pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                    rb
                ),
            )
            if len(deadline_result) == 3:
                new_candidates, new_waiting_deadlines, _ = deadline_result
            else:
                new_candidates, new_waiting_deadlines = deadline_result
            self._deadline_queue.extend(new_candidates)
            self._waiting_prefill_start_deadline_by_rid.update(new_waiting_deadlines)
        for req in self._pending_finished_prefill_reqs.values():
            candidate = self._decode_candidate_for_req(req, running_batch)
            if candidate is not None:
                self._deadline_queue.append(candidate)
        for req in self._pending_decoded_reqs.values():
            candidate = self._decode_candidate_for_req(req, running_batch)
            if candidate is not None:
                self._deadline_queue.append(candidate)
        latest_decode = {}
        others = []
        for candidate in self._deadline_queue:
            if candidate.event_type != "decode":
                others.append(candidate)
                continue
            prev = latest_decode.get(candidate.req.rid)
            if prev is None or candidate.event.completion_number >= prev.event.completion_number:
                latest_decode[candidate.req.rid] = candidate
        self._deadline_queue = others + list(latest_decode.values())
        self._deadline_queue.sort(
            key=lambda candidate: (
                candidate.start_deadline,
                0 if candidate.event_type == "decode" else 1,
                candidate.deadline,
            )
        )
        self._recompute_safe_prefix(waiting_queue, running_batch)
        self._clear_pending_mutations()

    def _async_snapshot_req(self, req: Req, *, output_delta: int = 0) -> Req:
        cloned = copy(req)
        cloned.origin_input_ids = list(req.origin_input_ids)
        cloned.output_ids = list(req.output_ids) + ([0] * output_delta)
        cloned.fill_ids = None if req.fill_ids is None else list(req.fill_ids)
        return cloned

    def _async_prepare_decode_epoch(
        self,
        *,
        running_batch: ScheduleBatch,
        waiting_queue: List[Req],
        selected_rids: Optional[set[str]],
        decode_steps: int,
    ) -> None:
        try:
            predicted_running = SimpleNamespace(reqs=list(running_batch.reqs))
            predicted_waiting = list(waiting_queue)
            self.simulator.sync_live_user_tracking(
                predicted_running,
                predicted_waiting,
                deltas_in_microseconds=self._deltas_us,
            )
            affected_user_ids = sorted(
                {
                    req.uid
                    for req in running_batch.reqs
                    if selected_rids is None or req.rid in selected_rids
                }
            )
            self._apply_logical_decode_updates(
                self.simulator,
                predicted_running,
                selected_rids=selected_rids,
                decode_steps=decode_steps,
            )
            deadline_start = time.perf_counter()
            deadline_result = self.simulator.build_deadline_candidates(
                    predicted_waiting,
                    predicted_running,
                    req_is_fair_prefill=lambda req, rb: self.user_is_fair_prefill(
                        req.uid,
                        running_batch=rb,
                        this_user_sum=req.get_estimated_prefill_impact(),
                    ),
                    req_is_fair_decode=lambda req, rb: self.req_is_fair_decode(
                        req, running_batch=rb
                    ),
                    event_delta_seconds=self._event_delta_seconds,
                    pooled_prefill_estimate_seconds=lambda req: pooled_prefill_time_estimation(
                        len(req.origin_input_ids),
                        len(req.origin_input_ids),
                        1,
                        max(int(self.delta_fairness_n or 1), 1),
                    ),
                    pooled_decode_estimate_seconds=lambda req, rb: self._current_pooled_decode_seconds(
                        rb
                    ),
                )
            if len(deadline_result) == 3:
                deadline_queue, waiting_prefill_start_deadline_by_rid, _ = deadline_result
            else:
                deadline_queue, waiting_prefill_start_deadline_by_rid = deadline_result
            deadline_elapsed_ms = (time.perf_counter() - deadline_start) * 1000.0
            result = {
                "deadline_queue": deadline_queue,
                "waiting_prefill_start_deadline_by_rid": waiting_prefill_start_deadline_by_rid,
                "waiting_sig": self._queue_sig(predicted_waiting),
                "running_sig": self._running_sig(predicted_running),
                "breakdown": {
                    "sync_live_user_tracking_ms": 0.0,
                    "logical_event_update_ms": 0.0,
                    "rebuild_from_real_state_ms": 0.0,
                    "rebuild_live_states_ms": 0.0,
                    "rebuild_state_setup_ms": 0.0,
                    "rebuild_scheduler_loop_ms": 0.0,
                    "rebuild_scheduler_step_count": 0,
                    "rebuild_prefill_step_count": 0,
                    "rebuild_decode_step_count": 0,
                    "build_deadline_candidates_ms": deadline_elapsed_ms,
                },
            }
            with self._async_prepare_lock:
                self._async_prepare_result = result
                self._async_prepare_exception = None
        except BaseException as exc:
            with self._async_prepare_lock:
                self._async_prepare_exception = exc
        finally:
            self._async_prepare_request = None

    def launch_async_decode_epoch_prepare(
        self,
        *,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        selected_rids: Optional[set[str]] = None,
        decode_steps: int = 1,
    ) -> bool:
        del running_batch, waiting_queue, selected_rids, decode_steps
        return False

    def wait_for_async_prepare(self) -> float:
        return 0.0

    def _consume_prepared_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> bool:
        snapshot = self._prepare_worker.latest_snapshot()
        if snapshot is None:
            return False
        if snapshot.task_seq <= self._last_consumed_prepare_snapshot_seq:
            return False
        waiting_by_rid = {req.rid: req for req in waiting_queue}
        running_by_rid = (
            {} if running_batch is None else {req.rid: req for req in running_batch.reqs}
        )
        live_req_by_rid = dict(waiting_by_rid)
        live_req_by_rid.update(running_by_rid)

        remapped_deadline_queue = []
        for candidate in snapshot.deadline_queue:
            live_req = live_req_by_rid.get(candidate.req.rid)
            if live_req is None:
                continue
            remapped_deadline_queue.append(
                DeadlineCandidate(
                    deadline=candidate.deadline,
                    start_deadline=candidate.start_deadline,
                    event_type=candidate.event_type,
                    req=live_req,
                    event=candidate.event,
                )
            )

        self._deadline_queue = tuple(remapped_deadline_queue)
        self._waiting_prefill_start_deadline_by_rid = snapshot.waiting_prefill_deadlines
        self._safe_waiting_queue = tuple(
            waiting_by_rid[req.rid]
            for req in snapshot.safe_waiting_queue
            if req.rid in waiting_by_rid
        )
        self._safe_waiting_rids = frozenset(req.rid for req in self._safe_waiting_queue)
        self._forced_prefill_queue = tuple(
            waiting_by_rid[req.rid]
            for req in snapshot.forced_prefill_queue
            if req.rid in waiting_by_rid
        )
        self._forced_prefill_rids = frozenset(
            req.rid for req in self._forced_prefill_queue
        )
        self._max_safe_prefill_tokens = snapshot.max_safe_prefill_tokens
        self._has_fair_waiting = snapshot.has_fair_waiting
        self._has_decode_deadline = snapshot.has_decode_deadline
        self._earliest_decode_start_deadline = snapshot.earliest_decode_start_deadline
        self._safe_prefix_now = snapshot.safe_prefix_now
        self._simulator_rebuild_prepared = False
        self._last_consumed_prepare_snapshot_seq = snapshot.task_seq
        return True

    def _ensure_current_pass_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
    ) -> None:
        del delta_fairness_deltas_microseconds
        waiting_sig = self._queue_sig(waiting_queue)
        running_sig = self._running_sig(running_batch)
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "ensure_consume_prepared"
            self._current_pass_waiting_sig = waiting_sig
            self._current_pass_running_sig = running_sig
            return
        if (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
            and not self._has_pending_mutations()
        ):
            self._last_pass_state_source = "ensure_current_reuse"
            return
        task_seq, duplicate_wait = self._request_live_prepare_snapshot(
            waiting_queue, running_batch
        )
        if self._wait_for_matching_prepare_snapshot(
            min_task_seq=task_seq,
            record_duplicate_wait=duplicate_wait,
        ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "ensure_wait_prepare"
        else:
            self._last_pass_state_source = "ensure_no_prepared"
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig

    def refresh_decode_hot_path_state(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch],
    ) -> None:
        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "hot_path_consume_prepared"
            return
        task_seq, duplicate_wait = self._request_live_prepare_snapshot(
            waiting_queue, running_batch
        )
        if self._wait_for_matching_prepare_snapshot(
            min_task_seq=task_seq,
            record_duplicate_wait=duplicate_wait,
        ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "hot_path_wait_prepare"

    def start_of_pass(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        *,
        new_token_ratio: float = 0.0,
        max_running_requests: Optional[int] = None,
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        self._prepare_worker.wait_for_mutation_queue_below_limit(100)
        super().start_of_pass(
            running_batch,
            waiting_queue,
            new_token_ratio=new_token_ratio,
            max_running_requests=max_running_requests,
        )
        breakdown: Dict[str, float] = {}
        pass_start = time.perf_counter()
        waiting_sig = self._queue_sig(waiting_queue)
        running_sig = self._running_sig(running_batch)

        if self._consume_prepared_pass_state(waiting_queue, running_batch):
            self._last_pass_state_source = "start_of_pass_consume_prepared"
        elif (
            self._current_pass_waiting_sig == waiting_sig
            and self._current_pass_running_sig == running_sig
            and not self._has_pending_mutations()
        ):
            self._last_pass_state_source = "start_of_pass_current_reuse"
        else:
            task_seq, duplicate_wait = self._request_live_prepare_snapshot(
                waiting_queue, running_batch
            )
            if self._wait_for_matching_prepare_snapshot(
                min_task_seq=task_seq,
                record_duplicate_wait=duplicate_wait,
            ) and self._consume_prepared_pass_state(waiting_queue, running_batch):
                self._last_pass_state_source = "start_of_pass_wait_prepare"
            else:
                self._last_pass_state_source = "start_of_pass_no_prepared"

        after_simulator = time.perf_counter()
        self._current_pass_waiting_sig = waiting_sig
        self._current_pass_running_sig = running_sig
        after_build = time.perf_counter()
        breakdown["doc_policy_after_super_ms"] = (after_simulator - pass_start) * 1000.0
        breakdown["doc_policy_start_of_pass_total_ms"] = (
            after_build - pass_start
        ) * 1000.0
        self._last_pass_breakdown_ms = breakdown

    def _logical_event_timestamp(
        self,
        req: Req,
        *,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> float:
        target = self.simulator if simulator is None else simulator
        real_event = target.most_recent_event_real.get(req.rid)
        if real_event is not None:
            return float(real_event.end_timestamp)
        tracked = target.requests.get(req.rid)
        if tracked is not None:
            return float(tracked.arrival_timestamp)
        return 0.0

    def _logical_next_event_timestamp(
        self,
        req: Req,
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        simulator: Optional[AlternateHistorySimulator] = None,
    ) -> float:
        target = self.simulator if simulator is None else simulator
        tracked = target.requests.get(req.rid)
        real_event = target.most_recent_event_real.get(req.rid)
        if tracked is not None and real_event is not None:
            upcoming = tracked.earliest_events_after_real_time(real_event) or []
            for event in upcoming:
                if event_type == "prefill" and isinstance(event, RequestPrefillEvent):
                    return float(event.end_timestamp)
                if (
                    event_type == "decode"
                    and isinstance(event, RequestDecodeEvent)
                    and (
                        completion_number is None
                        or event.completion_number == completion_number
                    )
                ):
                    return float(event.end_timestamp)
        return self._logical_event_timestamp(req, simulator=target)

    def _apply_logical_decode_updates(
        self,
        simulator: AlternateHistorySimulator,
        running_batch,
        *,
        selected_rids: Optional[set[str]],
        decode_steps: int,
    ) -> None:
        chosen = selected_rids
        for req in running_batch.reqs:
            if chosen is not None and req.rid not in chosen:
                continue
            tracked = simulator.requests.get(req.rid)
            if tracked is None:
                continue
            final_logical_ts = None
            previous_logical_ts = self._logical_event_timestamp(req, simulator=simulator)
            most_recent_event = tracked.most_recent_event()
            if most_recent_event is not None:
                previous_logical_ts = max(
                    previous_logical_ts, float(most_recent_event.end_timestamp)
                )
            if tracked.alternate_history_timeline.anticipated_future_events:
                previous_logical_ts = max(
                    previous_logical_ts,
                    max(
                        float(event.end_timestamp)
                        for event in tracked.alternate_history_timeline.anticipated_future_events
                    ),
                )
            start_completion_number = len(req.output_ids) + 1
            for completion_number in range(
                start_completion_number, start_completion_number + decode_steps
            ):
                logical_ts = self._logical_next_event_timestamp(
                    req,
                    event_type="decode",
                    completion_number=completion_number,
                    simulator=simulator,
                )
                if logical_ts <= previous_logical_ts:
                    context_tokens = len(req.origin_input_ids) + completion_number
                    logical_ts = previous_logical_ts + isolated_decode_time_estimation(
                        context_tokens,
                        context_tokens,
                        1,
                        max(int(self.delta_fairness_n or 1), 1),
                    )
                simulator.most_recent_event_real[req.rid] = RequestDecodeEvent(
                    req_id=req.rid,
                    end_timestamp=logical_ts,
                    completion_number=completion_number,
                )
                final_logical_ts = logical_ts
                previous_logical_ts = logical_ts
            if final_logical_ts is None:
                continue
            next_completion_number = start_completion_number + decode_steps
            context_tokens = len(req.origin_input_ids) + next_completion_number
            decode_duration = isolated_decode_time_estimation(
                context_tokens,
                context_tokens,
                1,
                max(int(self.delta_fairness_n or 1), 1),
            )
            tracked.alternate_history_timeline.anticipated_future_events = [
                RequestDecodeEvent(
                    req_id=req.rid,
                    duration=decode_duration,
                    end_timestamp=final_logical_ts + decode_duration,
                    completion_number=next_completion_number,
                )
            ]

    def _predicted_running_batch(
        self,
        running_batch: Optional[ScheduleBatch],
        scheduled_batch: Optional[ScheduleBatch],
    ) -> Optional[SimpleNamespace]:
        running_reqs = [] if running_batch is None else list(running_batch.reqs)
        if scheduled_batch is not None:
            running_reqs = running_reqs + list(scheduled_batch.reqs)
        if not running_reqs:
            return None
        return SimpleNamespace(reqs=running_reqs)

    def prepare_during_gpu_execution(
        self,
        *,
        event_type: str,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: List[Req],
        scheduled_batch: Optional[ScheduleBatch] = None,
        selected_rids: Optional[set[str]] = None,
        prepare_pass_state: bool = True,
        decode_steps: int = 1,
        new_token_ratio: float = 0.0,
    ) -> None:
        self._raise_prepare_thread_exception_if_any()
        phase_start = time.perf_counter()
        if event_type == "decode" and running_batch is not None and decode_steps > 0:
            self._apply_logical_decode_updates(
                self.simulator,
                running_batch,
                selected_rids=selected_rids,
                decode_steps=decode_steps,
            )
            chosen = selected_rids
            for req in running_batch.reqs:
                if chosen is None or req.rid in chosen:
                    self._pending_decoded_reqs[req.rid] = req
        running_snapshot = self._snapshot_batch_for_prepare(running_batch)
        waiting_snapshot = [
            self._snapshot_req_for_prepare(req) for req in waiting_queue
        ]
        scheduled_snapshot = self._snapshot_batch_for_prepare(scheduled_batch)
        self._enqueue_prepare_task(
            (
                event_type,
                running_snapshot,
                waiting_snapshot,
                scheduled_snapshot,
                None if selected_rids is None else set(selected_rids),
                prepare_pass_state,
                decode_steps,
                new_token_ratio,
                self._freeze_prepare_cache_state(),
                self._freeze_prepare_inputs(new_token_ratio=new_token_ratio),
                self._prepare_worker.mutation_seq,
            )
        )
        elapsed_ms = (time.perf_counter() - phase_start) * 1000.0
        self._last_prepare_breakdown_ms = {
            "sync_live_user_tracking_ms": 0.0,
            "logical_event_update_ms": 0.0,
            "rebuild_from_real_state_ms": 0.0,
            "build_deadline_candidates_ms": 0.0,
            "prepare_during_gpu_execution_total_ms": elapsed_ms,
        }

    def process_new_request(self, req: Req) -> None:
        super().process_new_request(req)
        self.simulator.process_new_request(req, self._deltas_us)
        self._pending_new_requests.append(req)
        self._enqueue_prepare_mutation(
            "process_new_request",
            (self._snapshot_req_for_prepare(req), dict(self._deltas_us)),
        )

    def note_scheduled_prefill_batch(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            self._pending_scheduled_prefill_reqs[req.rid] = req
        self._enqueue_prepare_mutation(
            "note_scheduled_prefill_batch",
            [self._snapshot_req_for_prepare(req) for req in batch.reqs],
        )

    def note_retracted_reqs(self, reqs) -> None:
        for req in reqs:
            self.simulator.process_new_request(req, self._deltas_us)
            self._pending_new_requests.append(req)
        self._enqueue_prepare_mutation(
            "note_retracted_reqs",
            (
                [self._snapshot_req_for_prepare(req) for req in reqs],
                dict(self._deltas_us),
            ),
        )

    def fairinf_force_prefill(
        self,
        req: Req,
        token_counters_by_user: Dict[str, List[int]],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional[ScheduleBatch] = None,
        decode_time_us: int = 20000,
    ) -> bool:
        del delta_fairness_deltas_microseconds, decode_time_us
        if req.rid not in self._forced_prefill_rids:
            return False
        this_users_extras = token_counters_by_user.get(req.uid, [])
        extra_sum = sum(this_users_extras)
        return self.user_is_fair_prefill(
            req.uid,
            running_batch=running_batch,
            this_user_len=len(this_users_extras),
            this_user_sum=extra_sum,
        ) and self._force_prefill_within_user_headroom(
            req,
            running_batch=running_batch,
            pending_prefill_tokens=extra_sum,
        )

    def fairinf_prioritize_force_prefill(self):
        return True

    def fairinf_force_prefill_any_waiting(
        self,
        waiting_queue: List[Req],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        running_batch: Optional[ScheduleBatch] = None,
    ) -> bool:
        del waiting_queue, delta_fairness_deltas_microseconds, running_batch
        return bool(self._forced_prefill_rids)

    def fairinf_force_decode(
        self,
        running_batch: Optional[ScheduleBatch],
        *,
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        decode_time_us: int = 20000,
    ) -> Tuple[bool, Optional[int]]:
        del decode_time_us
        if running_batch is None:
            return False, self._max_safe_prefill_tokens
        if self._has_decode_deadline and (self._max_safe_prefill_tokens or 0) <= 0:
            return True, 0
        return False, self._max_safe_prefill_tokens

    def force_prefill_reservations(
        self,
        waiting_queue: List[Req],
        *,
        token_counters_by_user: Dict[str, List[int]],
        adder,
        token_to_kv_pool=None,
        running_batch: Optional[ScheduleBatch],
        delta_fairness_deltas_microseconds: Optional[Dict[str, int]] = None,
        max_input_size: Optional[int] = None,
        prefix_computed: bool = False,
        max_running_requests: Optional[int] = None,
    ) -> Tuple[int, Optional[List[Req]]]:
        if not self._forced_prefill_rids:
            return 0, None
        prioritized_waiting = []
        safe_prefill_cap = self._max_safe_prefill_tokens or 0
        used_safe_prefill_tokens = 0
        local_token_counters_by_user = {
            user_id: list(tokens) for user_id, tokens in token_counters_by_user.items()
        }
        for req in self._forced_prefill_queue:
            if req.rid not in self._forced_prefill_rids:
                continue
            req_prefill_tokens = getattr(req, "extend_input_len", len(req.origin_input_ids))
            if (
                safe_prefill_cap > 0
                and used_safe_prefill_tokens + req_prefill_tokens > safe_prefill_cap
            ):
                break
            this_users_extras = local_token_counters_by_user.get(req.uid, [])
            extra_sum = sum(this_users_extras)
            if not self.user_is_fair_prefill(
                req.uid,
                running_batch=running_batch,
                this_user_len=len(this_users_extras),
                this_user_sum=extra_sum,
            ):
                continue
            if not self._force_prefill_within_user_headroom(
                req,
                running_batch=running_batch,
                pending_prefill_tokens=extra_sum,
            ):
                continue
            prioritized_waiting.append(req)
            used_safe_prefill_tokens += req_prefill_tokens
            local_token_counters_by_user.setdefault(req.uid, []).append(
                req_prefill_tokens
            )
        if not prioritized_waiting:
            return 0, None
        return super().force_prefill_reservations(
            prioritized_waiting,
            token_counters_by_user=token_counters_by_user,
            adder=adder,
            token_to_kv_pool=token_to_kv_pool,
            running_batch=running_batch,
            delta_fairness_deltas_microseconds=delta_fairness_deltas_microseconds,
            max_input_size=max_input_size,
            prefix_computed=prefix_computed,
            max_running_requests=max_running_requests,
            exact_forced_prefills=True,
            reservation_token_cap=used_safe_prefill_tokens,
        )

    def fairinf_overdue_decode_subset_rids(
        self,
        running_batch: Optional[ScheduleBatch],
    ) -> Optional[set[str]]:
        del running_batch
        return None

    def sorted_waiting_queue(self, waiting_queue: List[Req]):
        if not self._safe_waiting_queue:
            return waiting_queue

        prioritized = {req.rid: i for i, req in enumerate(self._safe_waiting_queue)}
        indexed = list(enumerate(waiting_queue))
        indexed.sort(
            key=lambda item: (
                0 if item[1].rid in prioritized else 1,
                prioritized.get(item[1].rid, 0),
                item[0],
            )
        )
        return [req for _, req in indexed]

    def _mark_violation_if_executed_after_deadline(
        self,
        req: Req,
        *,
        event_type: str,
        completion_number: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        tracked = self.simulator.requests.get(req.rid)
        real_event = self.simulator.most_recent_event_real.get(req.rid)
        if tracked is None or real_event is None:
            return

        now_ts = time.time() if now is None else now
        upcoming_events = tracked.earliest_events_after_real_time(real_event) or []
        matched_event = None
        for event in upcoming_events:
            if event_type == "prefill" and isinstance(event, RequestPrefillEvent):
                matched_event = event
                break
            if (
                event_type == "decode"
                and isinstance(event, RequestDecodeEvent)
                and (completion_number is None or event.completion_number == completion_number)
            ):
                matched_event = event
                break

        if matched_event is None:
            return

        deadline = matched_event.end_timestamp + self._event_delta_seconds(
            tracked, matched_event
        )
        if now_ts > deadline:
            TIMELINE_WRITER.mark_delta_violation(req.rid, req.uid, event_type=event_type)

    def finished_prefill(self, batch: ScheduleBatch) -> None:
        super().finished_prefill(batch)
        now = time.time()
        for req in batch.reqs:
            self._mark_violation_if_executed_after_deadline(
                req,
                event_type="prefill",
                now=now,
            )
        self.simulator.finished_prefill(batch)
        for req in batch.reqs:
            self._pending_finished_prefill_reqs[req.rid] = req
            self._pending_scheduled_prefill_reqs.pop(req.rid, None)
        self._enqueue_prepare_mutation(
            "finished_prefill",
            [self._snapshot_req_for_prepare(req) for req in batch.reqs],
        )

    def finished_decode(self, batch: ScheduleBatch, decode_rounds: int = 1) -> None:
        super().finished_decode(batch, decode_rounds=decode_rounds)
        needs_simulator_update = []
        for req in batch.reqs:
            self._pending_decoded_reqs[req.rid] = req
            real_event = self.simulator.most_recent_event_real.get(req.rid)
            if not isinstance(real_event, RequestDecodeEvent) or (
                real_event.completion_number < len(req.output_ids)
            ):
                needs_simulator_update.append(req)
        if needs_simulator_update:
            self.simulator.finished_decode(
                SimpleNamespace(reqs=needs_simulator_update),
                decode_rounds=decode_rounds,
            )
        self._enqueue_prepare_mutation(
            "finished_decode",
            (
                [self._snapshot_req_for_prepare(req) for req in batch.reqs],
                decode_rounds,
            ),
        )

    def mark_request_finished(self, req: Req) -> None:
        super().mark_request_finished(req)
        self.simulator.mark_request_finished(req)
        self._pending_finished_rids.add(req.rid)
        self._pending_decoded_reqs.pop(req.rid, None)
        self._pending_finished_prefill_reqs.pop(req.rid, None)
        self._pending_scheduled_prefill_reqs.pop(req.rid, None)
        self._enqueue_prepare_mutation(
            "mark_request_finished", self._snapshot_req_for_prepare(req)
        )
