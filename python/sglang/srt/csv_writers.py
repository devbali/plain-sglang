from __future__ import annotations

"""Async CSV writers for fairinf telemetry.

Each writer buffers rows in memory and flushes to disk on a dedicated I/O
thread (or at process exit), so callers on the GPU main thread or the
simulator worker thread are never blocked on disk I/O.

Set ENABLE_REQUEST_TIMELINE_WRITES = False to skip all RequestTimelineWriter
calls (mark_* methods become no-ops). Useful when the writes are too heavy.
"""

ENABLE_REQUEST_TIMELINE_WRITES: bool = True

import atexit
import csv
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional


_RID_PATTERN = re.compile(r"^user_(?P<user_id>\d+)_req_(?P<request_num>\d+)_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_request_ids(request_id: str, uid: Optional[str]) -> tuple[str, str]:
    match = _RID_PATTERN.match(request_id or "")
    if match:
        return match.group("user_id"), match.group("request_num")
    if uid and uid.startswith("user_"):
        return uid[len("user_"):], ""
    return uid or "", ""


# ---------------------------------------------------------------------------
# Shared base for append-only buffered writers
# ---------------------------------------------------------------------------

class _BufferedAppendWriter:
    """Buffers rows in memory; a background thread flushes to disk every 0.5s."""

    _HEADER: list[str] = []

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._header_written = False
        self._write_failed = False
        self._pending_rows: list = []
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name=f"csv-writer-{os.path.basename(csv_path)}",
            daemon=True,
        )
        self._flush_thread.start()
        atexit.register(self.flush)

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            self.flush()

    def _ensure_header_locked(self) -> None:
        if self._header_written or self._write_failed:
            return
        try:
            need_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            with open(self.csv_path, "a", newline="") as f:
                if need_header:
                    csv.writer(f).writerow(self._HEADER)
            self._header_written = True
        except OSError:
            self._write_failed = True

    def _enqueue(self, rows: list) -> None:
        if not rows or self._write_failed:
            return
        with self._lock:
            self._pending_rows.extend(rows)

    def flush(self) -> None:
        with self._lock:
            if self._write_failed or not self._pending_rows:
                return
            self._ensure_header_locked()
            if self._write_failed:
                return
            rows = self._pending_rows
            self._pending_rows = []
        try:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerows(rows)
        except OSError:
            self._write_failed = True
            with self._lock:
                self._pending_rows = rows + self._pending_rows


# ---------------------------------------------------------------------------
# RequestTimelineWriter — one row per request, rewritten atomically at exit
# ---------------------------------------------------------------------------

@dataclass
class TimelineRow:
    request_id: str
    user_id: str
    user_request_number: str
    queue_enter_ts: str = ""
    prefill_start_ts: str = ""
    prefill_done_ts: str = ""
    running_batch_removed_ts: str = ""
    running_batch_removed_count: int = 0
    retraction_count: int = 0
    delta_violation_count: int = 0
    prefill_delta_violation_count: int = 0
    decode_delta_violation_count: int = 0
    total_request_event_count: int = 0
    prefill_request_event_count: int = 0
    decode_request_event_count: int = 0
    first_decode_start_ts: str = ""
    isolated_start_ts: str = ""
    isolated_prefill_done_ts: str = ""
    isolated_first_decode_done_ts: str = ""
    isolated_latest_decode_done_ts: str = ""
    isolated_completed_ts: str = ""
    completed_ts: str = ""


_TIMELINE_HEADER = [
    "request_id", "user_id", "user_request_number",
    "queue_enter_ts", "prefill_start_ts", "prefill_done_ts",
    "running_batch_removed_ts", "running_batch_removed_count", "retraction_count",
    "delta_violation_count", "prefill_delta_violation_count", "decode_delta_violation_count",
    "total_request_event_count", "prefill_request_event_count", "decode_request_event_count",
    "first_decode_start_ts",
    "isolated_start_ts", "isolated_prefill_done_ts",
    "isolated_first_decode_done_ts", "isolated_latest_decode_done_ts",
    "isolated_completed_ts", "completed_ts",
]


class RequestTimelineWriter:
    """In-memory row store; written atomically to disk periodically and at process exit."""

    _FLUSH_INTERVAL_S: float = 30.0

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.csv_path = csv_path or os.path.join(os.getcwd(), "fairinf_request_timeline.csv")
        self._lock = threading.Lock()
        self._rows: Dict[str, TimelineRow] = {}
        self._write_failed = False
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="timeline-writer-flush",
            daemon=True,
        )
        self._flush_thread.start()
        atexit.register(self.flush)

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(self._FLUSH_INTERVAL_S):
            self.flush()

    def _get_or_create(self, request_id: str, uid: Optional[str]) -> TimelineRow:
        row = self._rows.get(request_id)
        if row is None:
            user_id, user_request_number = parse_request_ids(request_id, uid)
            row = TimelineRow(request_id=request_id, user_id=user_id, user_request_number=user_request_number)
            self._rows[request_id] = row
        return row

    def _update(self, request_id: str, uid: Optional[str], fn) -> None:
        if not ENABLE_REQUEST_TIMELINE_WRITES or not request_id:
            return
        with self._lock:
            fn(self._get_or_create(request_id, uid))

    def flush(self) -> None:
        # Disk writes are handled by COMPLETION_WRITER (append-only).
        # This method is kept as a no-op so atexit and the flush thread
        # don't error, but it no longer overwrites fairinf_request_timeline.csv.
        pass

    def mark_queue_enter(self, rid: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(rid, uid, lambda r: None if r.queue_enter_ts else setattr(r, "queue_enter_ts", now))

    def mark_prefill_start(self, rid: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(rid, uid, lambda r: None if r.prefill_start_ts else setattr(r, "prefill_start_ts", now))

    def mark_prefill_done(self, rid: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(rid, uid, lambda r: None if r.prefill_done_ts else setattr(r, "prefill_done_ts", now))

    def mark_running_batch_removed(self, rid: str, uid: Optional[str]) -> None:
        now = _now_iso()
        def _apply(r: TimelineRow) -> None:
            if not r.running_batch_removed_ts:
                r.running_batch_removed_ts = now
            r.running_batch_removed_count += 1
            r.retraction_count += 1
        self._update(rid, uid, _apply)

    def mark_first_decode_start(self, rid: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(rid, uid, lambda r: None if r.first_decode_start_ts else setattr(r, "first_decode_start_ts", now))

    def mark_isolated_start(self, rid: str, uid: Optional[str], *, timestamp_iso: str) -> None:
        self._update(rid, uid, lambda r: None if r.isolated_start_ts == timestamp_iso else setattr(r, "isolated_start_ts", timestamp_iso))

    def mark_delta_violation(self, rid: str, uid: Optional[str], *, event_type: str) -> None:
        if not ENABLE_REQUEST_TIMELINE_WRITES or not rid:
            return
        with self._lock:
            r = self._get_or_create(rid, uid)
            r.delta_violation_count += 1
            if event_type == "prefill":
                r.prefill_delta_violation_count += 1
            elif event_type == "decode":
                r.decode_delta_violation_count += 1

    def mark_request_event(self, rid: str, uid: Optional[str], *, event_type: str) -> None:
        if not ENABLE_REQUEST_TIMELINE_WRITES or not rid:
            return
        with self._lock:
            r = self._get_or_create(rid, uid)
            r.total_request_event_count += 1
            if event_type == "prefill":
                r.prefill_request_event_count += 1
            elif event_type == "decode":
                r.decode_request_event_count += 1

    def mark_completed(self, rid: str, uid: Optional[str], *, completion_writer=None) -> None:
        now = _now_iso()
        if not ENABLE_REQUEST_TIMELINE_WRITES or not rid:
            return
        with self._lock:
            r = self._get_or_create(rid, uid)
            if not r.completed_ts:
                r.completed_ts = now
                # Always write a row now (may lack isolated fields); a second
                # row with complete isolation data will be appended by
                # mark_isolated_completed.
                writer = completion_writer if completion_writer is not None else COMPLETION_WRITER
                writer.write_completion(r)

    def mark_isolated_prefill_done(self, rid: str, uid: Optional[str], *, timestamp_iso: str) -> None:
        self._update(rid, uid, lambda r: None if r.isolated_prefill_done_ts == timestamp_iso else setattr(r, "isolated_prefill_done_ts", timestamp_iso))

    def mark_isolated_decode_done(self, rid: str, uid: Optional[str], *, timestamp_iso: str, completion_number: int) -> None:
        def _apply(r: TimelineRow) -> None:
            if completion_number <= 1 and not r.isolated_first_decode_done_ts:
                r.isolated_first_decode_done_ts = timestamp_iso
            if r.isolated_latest_decode_done_ts != timestamp_iso:
                r.isolated_latest_decode_done_ts = timestamp_iso
        self._update(rid, uid, _apply)

    def mark_isolated_completed(self, rid: str, uid: Optional[str], *, timestamp_iso: str, completion_writer=None) -> None:
        if not ENABLE_REQUEST_TIMELINE_WRITES or not rid:
            return
        with self._lock:
            r = self._get_or_create(rid, uid)
            if r.isolated_completed_ts != timestamp_iso:
                r.isolated_completed_ts = timestamp_iso
            # Append a final row with all isolated fields filled in.
            if r.completed_ts and completion_writer is not None:
                completion_writer.write_completion(r)


# ---------------------------------------------------------------------------
# RunningBatchSnapshotWriter — append-only, buffered
# ---------------------------------------------------------------------------

class RunningBatchSnapshotWriter(_BufferedAppendWriter):
    _HEADER = [
        "timestamp", "batch_type", "running_batch_size",
        "request_id", "user_id", "user_request_number",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "waiting_time_in_decodes",
    ]

    def __init__(self, csv_path: Optional[str] = None) -> None:
        super().__init__(csv_path or os.path.join(os.getcwd(), "fairinf_running_batch.csv"))

    def write_snapshot(self, *, batch_type: str, running_reqs: Iterable[object]) -> None:
        now = _now_iso()
        reqs = list(running_reqs)
        rows = []
        for req in reqs:
            rid = getattr(req, "rid", "")
            uid = getattr(req, "uid", None)
            user_id, user_req_num = parse_request_ids(rid, uid)
            prompt_tokens = len(getattr(req, "origin_input_ids", []) or [])
            completion_tokens = len(getattr(req, "output_ids", []) or [])
            rows.append([
                now, batch_type, len(reqs),
                rid, user_id, user_req_num,
                prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
                getattr(req, "waiting_time_in_decodes", 0),
            ])
        self._enqueue(rows)


# ---------------------------------------------------------------------------
# IsolatedSimTimelineWriter — append-only, buffered
# ---------------------------------------------------------------------------

class IsolatedSimTimelineWriter(_BufferedAppendWriter):
    _HEADER = [
        "snapshot_ts", "request_id", "user_id", "user_request_number",
        "arrival_ts", "real_prefill_done", "real_decode_count",
        "anticipated_event_type", "anticipated_event_ts", "anticipated_completion_number",
    ]

    def __init__(self, csv_path: Optional[str] = None) -> None:
        super().__init__(csv_path or os.path.join(os.getcwd(), "fairinf_isolated_sim_timeline.csv"))

    def write_snapshot(self, simulator) -> None:
        now_iso = _now_iso()
        rows = []
        for uid, user_timeline in simulator.users.items():
            real_statuses = user_timeline.requests_real
            for rid, tracked in user_timeline.request_timelines.items():
                user_id, user_req_num = parse_request_ids(rid, uid)
                h = tracked.timeline.history
                arrival_ts = h[0].end_timestamp if h else tracked.arrival_timestamp
                arrival_iso = (
                    datetime.fromtimestamp(arrival_ts, timezone.utc).isoformat(timespec="milliseconds")
                    if arrival_ts != float("inf") else ""
                )
                status = real_statuses.get(rid)
                real_prefill_done = int(status.prefill_done) if status is not None else ""
                real_decode_count = status.decode_count if status is not None else ""

                ant = tracked.timeline.next_anticipated_event
                if ant is None:
                    event_type, event_ts, completion_number = "none", "", ""
                else:
                    from sglang.srt.delta_fairness.doc_policy_simulator import (
                        RequestPrefillEvent, RequestDecodeEvent,
                    )
                    if isinstance(ant, RequestPrefillEvent):
                        event_type = "prefill"
                    elif isinstance(ant, RequestDecodeEvent):
                        event_type = "decode"
                    else:
                        event_type = "other"
                    ts = ant.end_timestamp
                    event_ts = (
                        datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")
                        if ts != float("inf") else "inf"
                    )
                    completion_number = getattr(ant, "completion_number", "")

                rows.append([
                    now_iso, rid, user_id, user_req_num,
                    arrival_iso, real_prefill_done, real_decode_count,
                    event_type, event_ts, completion_number,
                ])
        self._enqueue(rows)


# ---------------------------------------------------------------------------
# RequestCompletionWriter — append-only, one row per completed request
# ---------------------------------------------------------------------------

class RequestCompletionWriter(_BufferedAppendWriter):
    _HEADER = _TIMELINE_HEADER

    def __init__(self, csv_path: Optional[str] = None) -> None:
        super().__init__(csv_path or os.path.join(os.getcwd(), "fairinf_request_completions.csv"))

    def write_completion(self, row: TimelineRow) -> None:
        if not ENABLE_REQUEST_TIMELINE_WRITES:
            return
        self._enqueue([[
            row.request_id, row.user_id, row.user_request_number,
            row.queue_enter_ts, row.prefill_start_ts, row.prefill_done_ts,
            row.running_batch_removed_ts, row.running_batch_removed_count,
            row.retraction_count, row.delta_violation_count,
            row.prefill_delta_violation_count, row.decode_delta_violation_count,
            row.total_request_event_count, row.prefill_request_event_count,
            row.decode_request_event_count, row.first_decode_start_ts,
            row.isolated_start_ts, row.isolated_prefill_done_ts,
            row.isolated_first_decode_done_ts, row.isolated_latest_decode_done_ts,
            row.isolated_completed_ts, row.completed_ts,
        ]])


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

TIMELINE_WRITER = RequestTimelineWriter()
COMPLETION_WRITER = RequestCompletionWriter()
RUNNING_BATCH_WRITER = RunningBatchSnapshotWriter()
ISOLATED_SIM_TIMELINE_WRITER = IsolatedSimTimelineWriter()
