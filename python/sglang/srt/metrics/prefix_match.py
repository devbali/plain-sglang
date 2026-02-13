"""Tracking helpers for prefix match hit rates."""

from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime
from typing import List, Union

_METRICS_BUFFER: List[dict] = []
_METRICS_LOCK = threading.Lock()
_CSV_PATH = os.path.abspath("sglang_prefix_match.csv")
logger = logging.getLogger(__name__)

_CSV_HEADER = [
    "timestamp",
    "prompt_tokens",
    "prefix_matched_tokens",
    "request_id",
    "request_uid",
]


def record_prefix_match_metric(
    prompt_tokens: int, prefix_tokens: int, rid: Union[str, int], uid: Union[str, int]
) -> None:
    """Append a prefix hit measurement to the buffer."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_tokens": prompt_tokens,
        "prefix_matched_tokens": prefix_tokens,
        "request_id": rid,
        "request_uid": uid,
    }
    with _METRICS_LOCK:
        _METRICS_BUFFER.append(entry)


def flush_prefix_match_metrics() -> None:
    """Persist buffered prefix hit metrics if any exist."""
    with _METRICS_LOCK:
        if not _METRICS_BUFFER:
            return
        rows = list(_METRICS_BUFFER)
        _METRICS_BUFFER.clear()

    dir_path = os.path.dirname(_CSV_PATH)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    file_exists = os.path.exists(_CSV_PATH)
    try:
        with open(_CSV_PATH, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    except Exception:
        logger.exception("Failed to flush prefix match metrics.")
