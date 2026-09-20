"""Cursor adapter.

Reads Cursor's two local session stores:

* **IDE composers** — SQLite ``state.vscdb`` databases under the Cursor ``User``
  directory (``globalStorage/state.vscdb`` and
  ``workspaceStorage/<hash>/state.vscdb``). Session headers live in
  ``cursorDiskKV`` under ``composerData:<id>`` and each message under
  ``bubbleId:<composer>:<bubble>``. Older builds cached messages inline in the
  composer's ``conversationMap``.
* **cursor-agent CLI transcripts** — JSONL files under
  ``~/.cursor/projects/<encoded-cwd>/agent-transcripts/`` in the Anthropic
  message shape.

Cursor's schema is undocumented and shifts between releases, so parsing is
best-effort: malformed rows are skipped rather than aborting a sync.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt, text_from_content
from ..model import Kind, NormalizedMessage, NormalizedSession
from .base import app_data_dir, iter_jsonl, to_ms, uri_to_path

BUBBLE_USER = 1
COMPOSER_INDEX_KEYS = ("composer.composerData", "composer.composerHeaders")


def _decode(raw) -> object | None:
    """Decode a Cursor KV value, tolerating bytes and double-encoded JSON."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8", "replace")
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _connect(db: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _composer_id(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


class CursorAdapter:
    name = "cursor"

    def __init__(self, home: Path | None = None) -> None:
        if home is not None:
            root = Path(home)
            self.user_dir = root / "User"
            self.cli_root = root / "projects"
        else:
            self.user_dir = app_data_dir("Cursor") / "User"
            self.cli_root = Path.home() / ".cursor" / "projects"
        self.default_path = self.user_dir

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        roots = [Path(p) for p in paths] if paths else [self.user_dir, self.cli_root]
        yield from self._ide_sessions(roots, since)
        yield from self._cli_sessions(roots, since)

    # -- IDE (state.vscdb) ------------------------------------------------
    def _ide_sessions(
        self, roots: list[Path], since: int | None
    ) -> Iterator[NormalizedSession]:
        dbs = self._db_files(roots)
        if not dbs:
            return
        projects = self._project_index(dbs)
        conns: list[tuple[Path, sqlite3.Connection]] = []
        for db in dbs:
            conn = _connect(db)
            if conn is not None:
                conns.append((db, conn))
        if not conns:
            return
        try:
            seen: dict[str, int] = {}
            for db, conn in conns:
                for cid, data in self._composers(conn):
                    updated = to_ms(data.get("lastUpdatedAt")) or 0
                    if cid in seen and updated <= seen[cid]:
                        continue
                    seen[cid] = updated
                    session = self._build_session(db, cid, data, projects, conns, since)
                    if session and session.messages:
                        yield session.finalize()
        finally:
            for _, conn in conns:
                conn.close()

    def _db_files(self, roots: list[Path]) -> list[Path]:
        files: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root.is_file():
                candidates = [root] if root.suffix == ".vscdb" else []
            elif root.is_dir():
                candidates = sorted(root.rglob("state.vscdb"))
            else:
                candidates = []
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    files.append(candidate)
        return files

    def _project_index(self, dbs: list[Path]) -> dict[str, str]:
        projects: dict[str, str] = {}
        for db in dbs:
            conn = _connect(db)
            if conn is None:
                continue
            try:
                folder = self._workspace_folder(db)
                if folder:
                    for cid in self._index_composer_ids(conn, "composer.composerData"):
                        projects.setdefault(cid, folder)
                for cid, path in self._header_projects(conn):
                    projects.setdefault(cid, path)
            finally:
                conn.close()
        return projects

    @staticmethod
    def _workspace_folder(db: Path) -> str | None:
        manifest = db.parent / "workspace.json"
        try:
            text = manifest.read_text(encoding="utf-8")
        except OSError:
            return None
        data = _decode(text)
        if not isinstance(data, dict):
            return None
        for key in ("folder", "workspace"):
            path = uri_to_path(data.get(key))
            if path:
                return path
        return None

    @staticmethod
    def _index_composer_ids(conn: sqlite3.Connection, key: str) -> list[str]:
        try:
            row = conn.execute("SELECT value FROM ItemTable WHERE key=?", (key,)).fetchone()
        except sqlite3.Error:
            return []
        data = _decode(row["value"]) if row else None
        if not isinstance(data, dict):
            return []
        out: list[str] = []
        for entry in data.get("allComposers") or []:
            if isinstance(entry, dict) and entry.get("composerId"):
                out.append(str(entry["composerId"]))
        return out

    @staticmethod
    def _header_projects(conn: sqlite3.Connection) -> Iterator[tuple[str, str]]:
        try:
            row = conn.execute(
                "SELECT value FROM ItemTable WHERE key='composer.composerHeaders'"
            ).fetchone()
        except sqlite3.Error:
            return
        data = _decode(row["value"]) if row else None
        if not isinstance(data, dict):
            return
        for entry in data.get("allComposers") or []:
            if not isinstance(entry, dict) or not entry.get("composerId"):
                continue
            identifier = entry.get("workspaceIdentifier") or {}
            path = uri_to_path(identifier.get("uri") if isinstance(identifier, dict) else None)
            if path:
                yield str(entry["composerId"]), path

    @staticmethod
    def _composers(conn: sqlite3.Connection) -> Iterator[tuple[str, dict]]:
        try:
            rows = conn.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
            )
        except sqlite3.Error:
            return
        for row in rows:
            data = _decode(row["value"])
            if not isinstance(data, dict):
                continue
            cid = data.get("composerId") or _composer_id(row["key"])
            if cid:
                yield str(cid), data

    def _build_session(
        self,
        db: Path,
        cid: str,
        data: dict,
        projects: dict[str, str],
        conns: list[tuple[Path, sqlite3.Connection]],
        since: int | None,
    ) -> NormalizedSession | None:
        started = to_ms(data.get("createdAt"))
        if since is not None and started is not None and started < since:
            return None

        model_config = data.get("modelConfig")
        default_model = (
            model_config.get("modelName") if isinstance(model_config, dict) else None
        )
        session = NormalizedSession(
            id=cid,
            harness=self.name,
            project_path=projects.get(cid),
            title=data.get("name") or None,
            started_at=started,
            ended_at=to_ms(data.get("lastUpdatedAt")),
            raw_path=str(db),
        )

        ordered = sorted(conns, key=lambda item: item[0] != db)
        conversation = data.get("conversationMap")
        headers = data.get("fullConversationHeadersOnly")

        seq = 0
        if isinstance(headers, list) and headers:
            for header in headers:
                if not isinstance(header, dict) or not header.get("bubbleId"):
                    continue
                bubble = self._load_bubble(ordered, cid, header["bubbleId"])
                if bubble is None and isinstance(conversation, dict):
                    bubble = conversation.get(header["bubbleId"])
                if isinstance(bubble, dict):
                    seq = self._add_bubble(session, header, bubble, default_model, seq)
        elif isinstance(conversation, dict):
            items = [(k, v) for k, v in conversation.items() if isinstance(v, dict)]
            items.sort(
                key=lambda item: to_ms(item[1].get("timestamp") or item[1].get("createdAt")) or 0
            )
            for bid, bubble in items:
                seq = self._add_bubble(session, {"bubbleId": bid}, bubble, default_model, seq)
        return session

    @staticmethod
    def _load_bubble(
        conns: list[tuple[Path, sqlite3.Connection]], cid: str, bid: str
    ) -> dict | None:
        key = f"bubbleId:{cid}:{bid}"
        for _, conn in conns:
            try:
                row = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key=?", (key,)
                ).fetchone()
            except sqlite3.Error:
                continue
            if row:
                data = _decode(row["value"])
                return data if isinstance(data, dict) else None
        return None

    @staticmethod
    def _add_bubble(
        session: NormalizedSession,
        header: dict,
        bubble: dict,
        default_model: str | None,
        seq: int,
    ) -> int:
        role = "user" if bubble.get("type", header.get("type")) == BUBBLE_USER else "assistant"
        text = bubble.get("text")
        if not isinstance(text, str) or not text:
            raw = bubble.get("rawText")
            text = raw if isinstance(raw, str) else ""
        model = None
        info = bubble.get("modelInfo")
        if isinstance(info, dict):
            model = info.get("modelName")
        model = model or default_model
        if session.project_path is None:
            project = bubble.get("workspaceProjectDir")
            if isinstance(project, str) and project:
                session.project_path = project

        if role == "user":
            if not text.strip():
                return seq
            kind = Kind.HUMAN if is_human_prompt(text) else Kind.CONTEXT
            if kind == Kind.HUMAN and session.title is None:
                session.title = text.strip().splitlines()[0][:120]
        else:
            if not text.strip() and not model:
                return seq
            kind = Kind.ASSISTANT

        session.add(
            NormalizedMessage(
                session_id=session.id,
                seq=seq,
                ts=to_ms(bubble.get("timestamp") or bubble.get("createdAt")),
                role=role,
                kind=kind,
                text=text,
                model=model,
                source_event_id=header.get("bubbleId"),
            )
        )
        return seq + 1

    # -- cursor-agent CLI transcripts ------------------------------------
    def _cli_sessions(
        self, roots: list[Path], since: int | None
    ) -> Iterator[NormalizedSession]:
        seen: set[Path] = set()
        for file in self._cli_files(roots):
            if file in seen:
                continue
            seen.add(file)
            session = self._parse_transcript(file, since)
            if session and session.messages:
                yield session.finalize()

    @staticmethod
    def _cli_files(roots: list[Path]) -> list[Path]:
        files: list[Path] = []
        for root in roots:
            if root.is_file():
                if root.suffix == ".jsonl" and "agent-transcripts" in root.parts:
                    files.append(root)
                continue
            if not root.is_dir():
                continue
            if root.name == "agent-transcripts":
                files.extend(sorted(root.glob("*.jsonl")))
                continue
            for directory in root.rglob("agent-transcripts"):
                if directory.is_dir():
                    files.extend(sorted(directory.glob("*.jsonl")))
        return files

    def _parse_transcript(self, path: Path, since: int | None) -> NormalizedSession | None:
        records = list(iter_jsonl(path))
        if not records:
            return None
        try:
            started = int(path.stat().st_mtime * 1000)
        except OSError:
            started = None
        if since is not None and started is not None and started < since:
            return None

        session = NormalizedSession(
            id=f"cli:{path.stem}",
            harness=self.name,
            project_path=_decode_project_dir(path),
            started_at=started,
            raw_path=str(path),
        )

        seq = 0
        for record in records:
            role = record.get("role")
            if role not in ("user", "assistant"):
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            text = text_from_content(message.get("content"))
            if not text.strip():
                continue
            if role == "user":
                kind = Kind.HUMAN if is_human_prompt(text) else Kind.CONTEXT
                if kind == Kind.HUMAN and session.title is None:
                    session.title = text.strip().splitlines()[0][:120]
            else:
                kind = Kind.ASSISTANT
            session.add(
                NormalizedMessage(
                    session_id=session.id,
                    seq=seq,
                    ts=started,
                    role=role,
                    kind=kind,
                    text=text,
                    model=message.get("model"),
                    provider=message.get("provider"),
                )
            )
            seq += 1
        return session


def _decode_project_dir(path: Path) -> str | None:
    """Recover the cwd from ``projects/<encoded-cwd>/agent-transcripts``.

    Cursor drops the leading separator and joins segments with ``-``, which is
    ambiguous when a segment contains a hyphen; this is a best-effort decode.
    """
    parts = path.parts
    if "agent-transcripts" not in parts:
        return None
    index = parts.index("agent-transcripts")
    if index == 0:
        return None
    encoded = parts[index - 1]
    return "/" + encoded.replace("-", "/") if encoded else None
