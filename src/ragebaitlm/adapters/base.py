"""Harness adapter protocol and helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

from ..model import NormalizedSession


class Adapter(Protocol):
    name: str
    default_path: Path

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]: ...


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_iso(value: str | None) -> int | None:
    """Parse an ISO-8601 timestamp into epoch milliseconds."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def to_ms(value) -> int | None:
    """Best-effort conversion of a timestamp (ms, s, or ISO string) to ms."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e12:  # already milliseconds
            return int(number)
        if number > 1e9:  # seconds
            return int(number * 1000)
        return int(number)
    if isinstance(value, str):
        return parse_iso(value)
    return None


def collect_files(root: Path, pattern: str, recursive: bool = True) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob(pattern) if recursive else root.glob(pattern))


PathFilter = Callable[[Path], bool]
