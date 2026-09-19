"""Pi adapter.

Reads ``~/.pi/agent/sessions/--<cwd>--/<timestamp>_<uuid>.jsonl`` as documented
in ``docs/session-format.md`` (installed with the pi-coding-agent package).

Session entries form a tree via ``id``/``parentId``. We walk the active branch
from the leaf to the root and ignore messages on abandoned branches.
``model_change`` entries are used as explicit model markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt, text_from_content
from ..model import Kind, NormalizedMessage, NormalizedSession
from .base import collect_files, iter_jsonl, parse_iso

ROLE_KIND = {"user": Kind.HUMAN, "assistant": Kind.ASSISTANT}


class PiAdapter:
    name = "pi"

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else Path.home() / ".pi"
        self.default_path = self.home / "agent" / "sessions"

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        roots = [Path(p) for p in paths] if paths else [self.default_path]
        for root in roots:
            for file in collect_files(root, "*.jsonl"):
                session = self._parse(file, since)
                if session and session.messages:
                    yield session.finalize()

    def _parse(self, path: Path, since: int | None) -> NormalizedSession | None:
        records = list(iter_jsonl(path))
        if not records:
            return None
        header = next((r for r in records if r.get("type") == "session"), {})
        entries = [r for r in records if r.get("type") != "session"]
        if not entries:
            return None

        session_id = str(header.get("id") or path.stem)
        started = parse_iso(header.get("timestamp"))
        if since is not None and started is not None and started < since:
            return None

        session = NormalizedSession(
            id=session_id,
            harness=self.name,
            project_path=header.get("cwd"),
            started_at=started,
            raw_path=str(path),
        )

        active = self._active_path(entries)
        model = None
        provider = None
        seq = 0

        for entry in entries:
            etype = entry.get("type")
            ts = parse_iso(entry.get("timestamp"))

            if etype == "model_change":
                model = entry.get("modelId") or model
                provider = entry.get("provider") or provider
                if len(active) == 0 or entry.get("id") in active:
                    session.add(
                        NormalizedMessage(
                            session_id=session_id,
                            seq=seq,
                            ts=ts,
                            role="system",
                            kind=Kind.MODEL_CHANGE,
                            model=model,
                            provider=provider,
                            source_event_id=entry.get("id"),
                        )
                    )
                    seq += 1
                continue

            if etype == "session_info" and entry.get("name"):
                session.title = entry["name"]
                continue

            if etype == "branch_summary":
                continue

            if etype != "message":
                continue

            on_active = not active or entry.get("id") in active
            message = entry.get("message") or {}
            role = message.get("role")
            if role not in ROLE_KIND:
                continue

            if not on_active:
                continue

            text = text_from_content(message.get("content"))
            if role == "assistant":
                kind = Kind.ASSISTANT
                model = message.get("model") or model
                provider = message.get("provider") or provider
            else:
                if not text.strip():
                    continue
                if is_human_prompt(text):
                    kind = Kind.HUMAN
                    if session.title is None:
                        session.title = text.strip().splitlines()[0][:120]
                else:
                    kind = Kind.CONTEXT

            session.add(
                NormalizedMessage(
                    session_id=session_id,
                    seq=seq,
                    ts=ts,
                    role=role,
                    kind=kind,
                    text=text,
                    model=model,
                    provider=provider,
                    stop_reason=message.get("stopReason"),
                    source_event_id=entry.get("id"),
                )
            )
            seq += 1

        return session

    @staticmethod
    def _active_path(entries: list[dict]) -> set[str]:
        """Return ids on the active branch (leaf -> root), or empty if linear."""
        with_id = [e for e in entries if e.get("id")]
        if not with_id or not all("parentId" in e for e in with_id):
            return set()
        by_id = {e["id"]: e for e in with_id}
        referenced = {e.get("parentId") for e in with_id if e.get("parentId")}
        leaves = [e for e in with_id if e["id"] not in referenced]
        if not leaves:
            return set()
        leaf = max(leaves, key=lambda e: e.get("timestamp") or "")
        path: set[str] = set()
        current = leaf.get("id")
        while current and current in by_id:
            path.add(current)
            current = by_id[current].get("parentId")
        return path
