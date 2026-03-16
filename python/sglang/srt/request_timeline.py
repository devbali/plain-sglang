from __future__ import annotations

import csv
import os
import re
import threading
import atexit
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional


_RID_PATTERN = re.compile(r"^user_(?P<user_id>\d+)_req_(?P<request_num>\d+)_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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
    completed_ts: str = ""


class RequestTimelineWriter:
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or os.path.join(os.getcwd(), "fairinf_request_timeline.csv")
        self._lock = threading.Lock()
        self._rows: Dict[str, TimelineRow] = {}
        self._write_failed = False
        atexit.register(self.flush)

    def _parse_ids(self, request_id: str, uid: Optional[str]) -> tuple[str, str]:
        match = _RID_PATTERN.match(request_id or "")
        if match:
            return match.group("user_id"), match.group("request_num")

        if uid and uid.startswith("user_"):
            return uid[len("user_") :], ""
        return uid or "", ""

    def _get_or_create(self, request_id: str, uid: Optional[str]) -> TimelineRow:
        row = self._rows.get(request_id)
        if row is None:
            user_id, user_request_number = self._parse_ids(request_id, uid)
            row = TimelineRow(
                request_id=request_id,
                user_id=user_id,
                user_request_number=user_request_number,
            )
            self._rows[request_id] = row
        return row

    def _flush_locked(self) -> None:
        if self._write_failed:
            return

        header = [
            "request_id",
            "user_id",
            "user_request_number",
            "queue_enter_ts",
            "prefill_start_ts",
            "prefill_done_ts",
            "running_batch_removed_ts",
            "running_batch_removed_count",
            "retraction_count",
            "delta_violation_count",
            "prefill_delta_violation_count",
            "decode_delta_violation_count",
            "total_request_event_count",
            "prefill_request_event_count",
            "decode_request_event_count",
            "first_decode_start_ts",
            "completed_ts",
        ]
        tmp_path = f"{self.csv_path}.tmp"
        try:
            with open(tmp_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for request_id in sorted(self._rows):
                    row = self._rows[request_id]
                    writer.writerow(
                        [
                            row.request_id,
                            row.user_id,
                            row.user_request_number,
                            row.queue_enter_ts,
                            row.prefill_start_ts,
                            row.prefill_done_ts,
                            row.running_batch_removed_ts,
                            row.running_batch_removed_count,
                            row.retraction_count,
                            row.delta_violation_count,
                            row.prefill_delta_violation_count,
                            row.decode_delta_violation_count,
                            row.total_request_event_count,
                            row.prefill_request_event_count,
                            row.decode_request_event_count,
                            row.first_decode_start_ts,
                            row.completed_ts,
                        ]
                    )
            os.replace(tmp_path, self.csv_path)
        except OSError:
            self._write_failed = True
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def _update(self, request_id: str, uid: Optional[str], update_fn) -> None:
        if not request_id:
            return
        with self._lock:
            row = self._get_or_create(request_id, uid)
            changed = update_fn(row)
            if changed:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def mark_queue_enter(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: False if row.queue_enter_ts else setattr(row, "queue_enter_ts", now) or True,
        )

    def mark_prefill_start(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: False if row.prefill_start_ts else setattr(row, "prefill_start_ts", now) or True,
        )

    def mark_prefill_done(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: False if row.prefill_done_ts else setattr(row, "prefill_done_ts", now) or True,
        )

    def mark_running_batch_removed(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()

        def _apply(row: TimelineRow) -> None:
            if not row.running_batch_removed_ts:
                row.running_batch_removed_ts = now
            row.running_batch_removed_count += 1
            row.retraction_count += 1
            return True

        self._update(request_id, uid, _apply)

    def mark_first_decode_start(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: False
            if row.first_decode_start_ts
            else setattr(row, "first_decode_start_ts", now) or True,
        )

    def mark_delta_violation(
        self, request_id: str, uid: Optional[str], *, event_type: str
    ) -> None:
        if not request_id:
            return
        with self._lock:
            row = self._get_or_create(request_id, uid)
            row.delta_violation_count += 1
            if event_type == "prefill":
                row.prefill_delta_violation_count += 1
            elif event_type == "decode":
                row.decode_delta_violation_count += 1

    def mark_request_event(
        self, request_id: str, uid: Optional[str], *, event_type: str
    ) -> None:
        if not request_id:
            return
        with self._lock:
            row = self._get_or_create(request_id, uid)
            row.total_request_event_count += 1
            if event_type == "prefill":
                row.prefill_request_event_count += 1
            elif event_type == "decode":
                row.decode_request_event_count += 1

    def mark_completed(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: False if row.completed_ts else setattr(row, "completed_ts", now) or True,
        )


TIMELINE_WRITER = RequestTimelineWriter()


class RunningBatchSnapshotWriter:
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or os.path.join(os.getcwd(), "fairinf_running_batch.csv")
        self._lock = threading.Lock()
        self._header_written = False
        self._write_failed = False

    def _ensure_header_locked(self) -> None:
        if self._header_written or self._write_failed:
            return
        try:
            file_exists = os.path.exists(self.csv_path)
            need_header = True
            if file_exists and os.path.getsize(self.csv_path) > 0:
                need_header = False
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "batch_type",
                            "running_batch_size",
                            "request_id",
                            "user_id",
                            "user_request_number",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "waiting_time_in_decodes",
                        ]
                    )
            self._header_written = True
        except OSError:
            self._write_failed = True

    def write_snapshot(
        self,
        *,
        batch_type: str,
        running_reqs: Iterable[object],
    ) -> None:
        if self._write_failed:
            return
        now = _now_iso()
        reqs = list(running_reqs)
        with self._lock:
            self._ensure_header_locked()
            if self._write_failed:
                return
            try:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    running_batch_size = len(reqs)
                    for req in reqs:
                        request_id = getattr(req, "rid", "")
                        uid = getattr(req, "uid", None)
                        user_id, user_request_number = TIMELINE_WRITER._parse_ids(
                            request_id, uid
                        )
                        prompt_tokens = len(getattr(req, "origin_input_ids", []) or [])
                        completion_tokens = len(getattr(req, "output_ids", []) or [])
                        total_tokens = prompt_tokens + completion_tokens
                        writer.writerow(
                            [
                                now,
                                batch_type,
                                running_batch_size,
                                request_id,
                                user_id,
                                user_request_number,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                getattr(req, "waiting_time_in_decodes", 0),
                            ]
                        )
            except OSError:
                self._write_failed = True


RUNNING_BATCH_WRITER = RunningBatchSnapshotWriter()
