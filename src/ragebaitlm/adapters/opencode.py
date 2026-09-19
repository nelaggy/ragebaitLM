"""OpenCode adapter.

Reads the SQLite database at ``~/.local/share/opencode/opencode.db`` directly
(read-only). Only ``text`` parts are read, so multi-gigabyte tool output is
never loaded.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt
from ..model import Kind, NormalizedMessage, NormalizedSession


class OpenCodeAdapter:
    name = "opencode"

    def __init__(self, home: Path | None = None) -> None:
        base = Path(home) if home else Path.home() / ".local" / "share" / "opencode"
        self.default_path = base / "opencode.db"

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        db = Path(next(iter(paths))) if paths else self.default_path
        if not db.exists():
            return
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT * FROM session ORDER BY time_created"):
                if since is not None and (row["time_created"] or 0) < since:
                    continue
                session = self._build(conn, row)
                if session and session.messages:
                    yield session.finalize()
        finally:
            conn.close()

    def _session_model(self, raw: str | None) -> tuple[str | None, str | None]:
        if not raw:
            return None, None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        return data.get("id"), data.get("providerID")

    def _build(self, conn: sqlite3.Connection, row: sqlite3.Row) -> NormalizedSession:
        session_id = row["id"]
        is_subagent = bool(row["parent_id"])
        session_model, session_provider = self._session_model(row["model"])

        session = NormalizedSession(
            id=session_id,
            harness=self.name,
            project_path=row["directory"],
            title=row["title"],
            started_at=row["time_created"],
            ended_at=row["time_updated"],
            is_subagent=is_subagent,
            parent_session_id=row["parent_id"],
            raw_path=str(self.default_path),
        )

        # Group text parts by message id. The LIKE pre-filter avoids scanning
        # huge tool-output payloads with json_extract before discarding them.
        parts: dict[str, list[str]] = {}
        for part in conn.execute(
            """SELECT message_id, data FROM part
               WHERE session_id=?
                 AND (data LIKE '%"type":"text"%' OR data LIKE '%"type": "text"%')
               ORDER BY time_created, id""",
            (session_id,),
        ):
            try:
                data = json.loads(part["data"])
            except json.JSONDecodeError:
                continue
            if data.get("type") != "text":
                continue
            text = data.get("text")
            if text:
                parts.setdefault(part["message_id"], []).append(text)

        seq = 0
        for message in conn.execute(
            """SELECT id, time_created, data FROM message
               WHERE session_id=? ORDER BY time_created, id""",
            (session_id,),
        ):
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError:
                continue
            role = data.get("role")
            if role not in ("user", "assistant"):
                continue
            text = "\n".join(parts.get(message["id"], []))
            model = data.get("modelID") or session_model
            provider = data.get("providerID") or session_provider
            ts = (data.get("time") or {}).get("created") or message["time_created"]

            if role == "user":
                if not text.strip():
                    continue
                if is_subagent:
                    kind = Kind.SUBAGENT_PROMPT
                elif is_human_prompt(text):
                    kind = Kind.HUMAN
                else:
                    kind = Kind.CONTEXT
                user_model = (data.get("model") or {})
                model = user_model.get("modelID") or model
                provider = user_model.get("providerID") or provider
            else:
                kind = Kind.ASSISTANT

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
                    stop_reason=data.get("finish"),
                    source_event_id=message["id"],
                    is_subagent=is_subagent,
                )
            )
            seq += 1

        return session
