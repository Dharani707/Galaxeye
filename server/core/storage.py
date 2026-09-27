from __future__ import annotations

import csv
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server.core import config

_TRUTHY = {"1", "true", "yes", "y", "t"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def new_prediction_id() -> str:
    return uuid.uuid4().hex[:16]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in _TRUTHY


def ensure_predictions_file(path: Path | None = None) -> Path:
    target = Path(path) if path is not None else config.PREDICTIONS_PATH
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(config.PREDICTION_COLUMNS)).writeheader()
    return target


def append_prediction(record: dict[str, Any], path: Path | None = None) -> str:
    target = ensure_predictions_file(path)
    row: dict[str, Any] = {column: record.get(column, "") for column in config.PREDICTION_COLUMNS}
    if not row["prediction_id"]:
        row["prediction_id"] = new_prediction_id()
    if not row["created_at"]:
        row["created_at"] = utc_now()
    row["needs_review"] = _as_bool(row["needs_review"])
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(config.PREDICTION_COLUMNS))
        writer.writerow(row)
        handle.flush()
    return str(row["prediction_id"])


def query_predictions(
    label: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    needs_review: bool | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    target = ensure_predictions_file(path)
    rows: list[dict[str, Any]] = []
    with target.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if label is not None and raw.get("label") != label:
                continue
            if needs_review is not None and _as_bool(raw.get("needs_review")) is not needs_review:
                continue
            if min_confidence is not None or max_confidence is not None:
                try:
                    confidence = float(raw.get("confidence", ""))
                except ValueError:
                    continue
                if min_confidence is not None and confidence < min_confidence:
                    continue
                if max_confidence is not None and confidence > max_confidence:
                    continue
            created = str(raw.get("created_at", ""))
            if since is not None and created < since:
                continue
            if until is not None and created > until:
                continue
            rows.append(dict(raw))
    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    if limit and limit > 0:
        return rows[:limit]
    return rows


def count_predictions(path: Path | None = None) -> int:
    target = ensure_predictions_file(path)
    with target.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def label_distribution(path: Path | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in query_predictions(limit=0, path=path):
        label = str(row.get("label", ""))
        counts[label] = counts.get(label, 0) + 1
    return counts
