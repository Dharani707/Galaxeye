from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from server.core import config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonLogger:
    def __init__(
        self,
        name: str,
        path: Path | None = None,
        run_id: str | None = None,
        echo: bool = True,
    ) -> None:
        self.name = name
        self.path = Path(path) if path is not None else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.echo = echo
        self._stream: TextIO | None = None

    def _handle(self) -> TextIO:
        if self._stream is None:
            if self.path is None:
                raise RuntimeError(f"logger {self.name!r} has no path configured")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        return self._stream

    def _write(self, level: str, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": utc_now(),
            "level": level,
            "logger": self.name,
            "run_id": self.run_id,
            "event": event,
        }
        record.update(fields)
        if self.path is not None:
            stream = self._handle()
            stream.write(json.dumps(record, default=str) + "\n")
            stream.flush()
        if self.echo:
            extras = " ".join(f"{key}={value}" for key, value in fields.items())
            suffix = f" {extras}" if extras else ""
            print(f"[{level.upper():<7}] {event}{suffix}", file=sys.stderr, flush=True)

    def info(self, event: str, **fields: Any) -> None:
        self._write("info", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._write("warning", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._write("error", event, **fields)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def get_logger(
    name: str,
    filename: str | None = None,
    run_id: str | None = None,
    echo: bool = True,
) -> JsonLogger:
    path = config.LOG_DIR / filename if filename else None
    return JsonLogger(name, path=path, run_id=run_id, echo=echo)
