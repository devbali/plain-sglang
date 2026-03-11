from __future__ import annotations

import csv
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


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
    first_decode_start_ts: str = ""
    completed_ts: str = ""


class RequestTimelineWriter:
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or os.path.join(os.getcwd(), "fairinf_request_timeline.csv")
        self._lock = threading.Lock()
        self._rows: Dict[str, TimelineRow] = {}
        self._write_failed = False

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
            update_fn(row)
            self._flush_locked()

    def mark_queue_enter(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: setattr(row, "queue_enter_ts", row.queue_enter_ts or now),
        )

    def mark_prefill_start(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: setattr(row, "prefill_start_ts", row.prefill_start_ts or now),
        )

    def mark_prefill_done(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: setattr(row, "prefill_done_ts", row.prefill_done_ts or now),
        )

    def mark_running_batch_removed(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()

        def _apply(row: TimelineRow) -> None:
            if not row.running_batch_removed_ts:
                row.running_batch_removed_ts = now
            row.running_batch_removed_count += 1

        self._update(request_id, uid, _apply)

    def mark_first_decode_start(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: setattr(
                row, "first_decode_start_ts", row.first_decode_start_ts or now
            ),
        )

    def mark_completed(self, request_id: str, uid: Optional[str]) -> None:
        now = _now_iso()
        self._update(
            request_id,
            uid,
            lambda row: setattr(row, "completed_ts", row.completed_ts or now),
        )


TIMELINE_WRITER = RequestTimelineWriter()
