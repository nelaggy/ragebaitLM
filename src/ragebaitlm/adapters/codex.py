"""Codex adapter.

Reads rollout JSONL files under ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.

Codex records injected context (plugins, environment blocks) as ``user``
messages in ``response_item`` records, but only real user messages appear as
``UserMessage`` items in ``event_msg.item_completed``. We therefore prefer the
latter and fall back to ``response_item`` for older sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..filters import has_annotation, is_human_prompt, text_from_content
from ..model import Kind, NormalizedMessage, NormalizedSession
from .base import collect_files, iter_jsonl, parse_iso

ITEM_KIND = {
    "UserMessage": Kind.HUMAN,
    "AgentMessage": Kind.ASSISTANT,
}


class CodexAdapter:
    name = "codex"

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else Path.home() / ".codex"
        self.default_path = self.home / "sessions"

    def _titles(self) -> dict[str, str]:
        titles: dict[str, str] = {}
        index = self.home / "session_index.jsonl"
        for row in iter_jsonl(index):
            sid = row.get("id")
            if sid and row.get("thread_name"):
                titles[sid] = row["thread_name"]
        return titles

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        roots = [Path(p) for p in paths] if paths else [self.default_path]
        titles = self._titles()
        for root in roots:
            for file in collect_files(root, "rollout-*.jsonl"):
                session = self._parse(file, titles, since)
                if session and session.messages:
                    yield session.finalize()

    def _parse(
        self, path: Path, titles: dict[str, str], since: int | None
    ) -> NormalizedSession | None:
        records = list(iter_jsonl(path))
        if not records:
            return None

        meta = next(
            (r["payload"] for r in records if r.get("type") == "session_meta"), {}
        )
        session_id = meta.get("session_id") or meta.get("id") or path.stem
        cwd = meta.get("cwd")

        started = parse_iso(meta.get("timestamp"))
        if since is not None and started is not None and started < since:
            return None

        session = NormalizedSession(
            id=str(session_id),
            harness=self.name,
            project_path=cwd,
            title=titles.get(str(session_id)),
            started_at=started,
            is_subagent=bool(meta.get("agent_path"))
            or meta.get("thread_source") not in (None, "user"),
            raw_path=str(path),
        )

        has_items = any(
            r.get("type") == "event_msg"
            and r.get("payload", {}).get("type") == "item_completed"
            for r in records
        )
        builder = self._from_items if has_items else self._from_response_items
        builder(records, session)
        return session

    # -- message sources -------------------------------------------------
    def _from_items(self, records: list[dict], session: NormalizedSession) -> None:
        model: str | None = None
        provider: str | None = None
        seq = 0
        for record in records:
            rtype = record.get("type")
            payload = record.get("payload", {})
            ts = parse_iso(record.get("timestamp"))

            if rtype == "event_msg" and payload.get("type") == "thread_settings_applied":
                settings = payload.get("thread_settings", {})
                model = settings.get("model") or model
                provider = settings.get("model_provider_id") or provider
                continue

            if rtype == "event_msg" and payload.get("type") in {
                "turn_aborted",
                "user_interrupted",
            }:
                continue

            if rtype != "event_msg" or payload.get("type") != "item_completed":
                continue

            item = payload.get("item", {})
            item_type = item.get("type")
            if item_type not in ITEM_KIND:
                continue
            text = text_from_content(item.get("content"))
            if not text.strip():
                continue

            if item_type == "UserMessage":
                if has_annotation(text):
                    kind = Kind.CONTEXT
                elif is_human_prompt(text):
                    kind = Kind.HUMAN
                else:
                    kind = Kind.CONTEXT
            else:
                kind = Kind.ASSISTANT

            session.add(
                NormalizedMessage(
                    session_id=session.id,
                    seq=seq,
                    ts=ts,
                    role="user" if item_type == "UserMessage" else "assistant",
                    kind=kind,
                    text=text,
                    model=model,
                    provider=provider,
                    source_event_id=item.get("id"),
                )
            )
            seq += 1

    def _from_response_items(self, records: list[dict], session: NormalizedSession) -> None:
        model: str | None = None
        provider: str | None = None
        seq = 0
        for record in records:
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            text = text_from_content(payload.get("content"))
            if not text.strip():
                continue
            ts = parse_iso(record.get("timestamp"))

            if role == "user":
                if has_annotation(text):
                    kind = Kind.CONTEXT
                elif is_human_prompt(text):
                    kind = Kind.HUMAN
                else:
                    kind = Kind.CONTEXT
            else:
                kind = Kind.ASSISTANT

            session.add(
                NormalizedMessage(
                    session_id=session.id,
                    seq=seq,
                    ts=ts,
                    role=role,
                    kind=kind,
                    text=text,
                    model=model,
                    provider=provider,
                    source_event_id=payload.get("id"),
                )
            )
            seq += 1
