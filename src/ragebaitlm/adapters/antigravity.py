"""Google Antigravity adapter.

Antigravity stores each conversation as a standalone SQLite database under
``<data-root>/conversations/<uuid>.db``. Two surfaces share the format:

* the **Antigravity CLI** (``agy``) at ``~/.gemini/antigravity-cli``, and
* newer builds of the **Antigravity IDE** at ``~/.gemini/antigravity`` /
  ``antigravity-ide``.

The database is undocumented and reverse-engineered. Each row of ``steps`` is a
protobuf-encoded ``gemini_coder.Step`` (type tag, a ``CortexStepMetadata``
envelope at field 5, and one payload per step kind):

* ``step_type`` 14 ``CortexStepUserInput`` — human prompt, text at payload
  field 19 → field 2 (falling back to 19 → 3 → 1);
* ``step_type`` 15 ``CortexStepPlannerResponse`` — model turn, visible reply at
  payload field 20 → field 1;
* ``step_type`` 23 ``Checkpoint`` — carries the conversation title.

Timestamps live at metadata field 1 (``{seconds, nanos}``); the model is a
numeric enum at metadata field 11, surfaced as ``model-<n>`` unless
``gen_metadata`` maps it to a name. Workspace and creation time come from the
``main`` row of ``trajectory_metadata_blob``.

Legacy IDE conversations are AES-GCM encrypted ``.pb`` files that cannot be
read offline; they are skipped, as are databases with an unexpected schema.
Parsing is deliberately tolerant and degrades to fewer messages rather than
aborting a sync.
"""

from __future__ import annotations

import re
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt
from ..model import Kind, NormalizedMessage, NormalizedSession

STEP_USER_INPUT = 14
STEP_PLANNER_RESPONSE = 15
STEP_CHECKPOINT = 23

DRIVE_RE = re.compile(r"^/[A-Za-z]:")


# -- minimal protobuf wire reader -----------------------------------------
def _pb_varint(data: bytes, index: int) -> tuple[int | None, int]:
    value = 0
    shift = 0
    while index < len(data):
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift > 63:
            return None, index
    return None, index


def _pb_iter(data: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield ``(field, wire_type, value)`` for a protobuf message.

    Only varint and length-delimited fields are decoded; anything else stops the
    scan, since unknown wire types make the remaining boundaries ambiguous.
    """
    index = 0
    while index < len(data):
        tag, index = _pb_varint(data, index)
        if tag is None:
            return
        field = tag >> 3
        wire = tag & 7
        if field == 0:
            return
        if wire == 0:
            value, index = _pb_varint(data, index)
            if value is None:
                return
            yield field, wire, value
        elif wire == 2:
            length, index = _pb_varint(data, index)
            if length is None or index + length > len(data):
                return
            yield field, wire, data[index : index + length]
            index += length
        elif wire == 1:
            index += 8
        elif wire == 5:
            index += 4
        else:
            return


def _pb_bytes(data: bytes, field: int) -> bytes | None:
    """First length-delimited value for ``field``."""
    for f, wire, value in _pb_iter(data):
        if f == field and wire == 2:
            return value
    return None


def _pb_all_bytes(data: bytes, field: int) -> Iterator[bytes]:
    for f, wire, value in _pb_iter(data):
        if f == field and wire == 2:
            yield value


def _pb_uint(data: bytes, field: int) -> int | None:
    for f, wire, value in _pb_iter(data):
        if f == field and wire == 0:
            return value
    return None


def _pb_str(data: bytes, field: int) -> str | None:
    raw = _pb_bytes(data, field)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _pb_timestamp(data: bytes) -> int | None:
    """Milliseconds since the epoch from a ``google.protobuf.Timestamp``."""
    seconds = _pb_uint(data, 1)
    if seconds is None:
        return None
    nanos = _pb_uint(data, 2) or 0
    if nanos > 999_999_999:
        nanos = 0
    return seconds * 1000 + nanos // 1_000_000


def _path_from_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    if uri.startswith("file:"):
        path = urllib.parse.unquote(urllib.parse.urlparse(uri).path)
        if DRIVE_RE.match(path):
            path = path[1:]
        return path or None
    return uri


class AntigravityAdapter:
    name = "antigravity"

    def __init__(self, home: Path | None = None) -> None:
        base = Path(home) if home is not None else Path.home() / ".gemini"
        self.roots = self._roots(base)
        self.default_path = self.roots[0]

    @staticmethod
    def _roots(base: Path) -> list[Path]:
        candidates = [
            base / "antigravity-cli" / "conversations",
            base / "antigravity-ide" / "conversations",
            base / "antigravity" / "conversations",
            base / "conversations",
        ]
        existing = [path for path in candidates if path.exists()]
        return existing or [candidates[0]]

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        for file in self._db_files(list(paths) if paths else self.roots):
            session = self._parse(file, since)
            if session and session.messages:
                yield session.finalize()

    @staticmethod
    def _db_files(candidates: list[Path]) -> list[Path]:
        files: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate.is_file():
                found = [candidate] if candidate.suffix == ".db" else []
            elif candidate.is_dir():
                found = sorted(candidate.glob("*.db"))
                if not found:
                    found = sorted(candidate.glob("conversations/*.db"))
            else:
                found = []
            for file in found:
                if file not in seen:
                    seen.add(file)
                    files.append(file)
        return files

    def _parse(self, path: Path, since: int | None) -> NormalizedSession | None:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            return None
        conn.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "steps" not in tables:
                return None
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(steps)")
            }
            if "step_type" not in columns:
                return None

            def column(name: str) -> str:
                return name if name in columns else f"NULL AS {name}"

            main = self._main_metadata(conn) if "trajectory_metadata_blob" in tables else b""
            sid = self._session_id(conn, path, tables, main)
            started = _pb_timestamp(_pb_bytes(main, 2)) if main else None
            if started is None:
                started = self._mtime(path)
            if since is not None and started is not None and started < since:
                return None

            session = NormalizedSession(
                id=sid,
                harness=self.name,
                project_path=self._project_path(main),
                started_at=started,
                raw_path=str(path),
            )

            names = self._generation_models(conn, tables)
            checkpoint_title = None
            seq = 0
            ended = started
            for row in conn.execute(
                f"SELECT idx, step_type, {column('metadata')}, "
                f"{column('step_payload')} FROM steps ORDER BY idx"
            ):
                step_type = row["step_type"]
                payload = row["step_payload"] or b""
                metadata = row["metadata"] or _pb_bytes(payload, 5) or b""
                ts = self._step_timestamp(metadata)
                if ts:
                    ended = ts if ended is None else max(ended, ts)

                if step_type == STEP_CHECKPOINT:
                    checkpoint_title = self._checkpoint_title(payload) or checkpoint_title
                    continue
                if step_type == STEP_USER_INPUT:
                    seq = self._add_user(session, payload, ts, seq)
                elif step_type == STEP_PLANNER_RESPONSE:
                    seq = self._add_assistant(session, payload, metadata, names, ts, seq)

            session.ended_at = ended
            if checkpoint_title and session.title is None:
                session.title = checkpoint_title
            return session
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    @staticmethod
    def _mtime(path: Path) -> int | None:
        try:
            return int(path.stat().st_mtime * 1000)
        except OSError:
            return None

    @staticmethod
    def _main_metadata(conn: sqlite3.Connection) -> bytes:
        try:
            row = conn.execute(
                "SELECT data FROM trajectory_metadata_blob WHERE id='main' "
                "OR id IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return b""
        if row is None or row["data"] is None:
            return b""
        return bytes(row["data"])

    @staticmethod
    def _session_id(
        conn: sqlite3.Connection, path: Path, tables: set[str], main: bytes
    ) -> str:
        if "trajectory_meta" in tables:
            try:
                row = conn.execute(
                    "SELECT cascade_id FROM trajectory_meta LIMIT 1"
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row is not None and row["cascade_id"]:
                return str(row["cascade_id"])
        sid = _pb_str(main, 6) if main else None
        return sid or path.stem

    @staticmethod
    def _project_path(main: bytes) -> str | None:
        if not main:
            return None
        workspace = _pb_bytes(main, 1)
        raw = _pb_str(workspace, 1) if workspace else None
        if not raw:
            raw = _pb_str(main, 7)
        return _path_from_uri(raw)

    @staticmethod
    def _step_timestamp(metadata: bytes) -> int | None:
        if not metadata:
            return None
        return _pb_timestamp(_pb_bytes(metadata, 1))

    @staticmethod
    def _checkpoint_title(payload: bytes) -> str | None:
        checkpoint = _pb_bytes(payload, 30)
        if not checkpoint:
            return None
        for field in (10, 4):
            title = _pb_str(checkpoint, field)
            if title and title.strip():
                return title.strip().splitlines()[0][:120]
        return None

    def _add_user(
        self, session: NormalizedSession, payload: bytes, ts: int | None, seq: int
    ) -> int:
        text = self._user_text(payload)
        if not text.strip():
            return seq
        kind = Kind.HUMAN if is_human_prompt(text) else Kind.CONTEXT
        if kind == Kind.HUMAN and session.title is None:
            session.title = text.strip().splitlines()[0][:120]
        session.add(
            NormalizedMessage(
                session_id=session.id,
                seq=seq,
                ts=ts,
                role="user",
                kind=kind,
                text=text,
            )
        )
        return seq + 1

    def _add_assistant(
        self,
        session: NormalizedSession,
        payload: bytes,
        metadata: bytes,
        names: dict[str, str],
        ts: int | None,
        seq: int,
    ) -> int:
        text = self._assistant_text(payload)
        model = self._model(metadata, names)
        if not text.strip() and not model:
            return seq
        session.add(
            NormalizedMessage(
                session_id=session.id,
                seq=seq,
                ts=ts,
                role="assistant",
                kind=Kind.ASSISTANT,
                text=text,
                model=model,
                provider="antigravity" if model else None,
            )
        )
        return seq + 1

    @staticmethod
    def _user_text(payload: bytes) -> str:
        body = _pb_bytes(payload, 19)
        if not body:
            return ""
        text = _pb_str(body, 2)
        if text and text.strip():
            return text
        for item in _pb_all_bytes(body, 3):
            inner = _pb_str(item, 1)
            if inner and inner.strip():
                return inner
        return _pb_str(body, 1) or ""

    @staticmethod
    def _assistant_text(payload: bytes) -> str:
        planner = _pb_bytes(payload, 20)
        if not planner:
            return ""
        return _pb_str(planner, 1) or ""

    @staticmethod
    def _model(metadata: bytes, names: dict[str, str]) -> str | None:
        if not metadata:
            return None
        response_id = AntigravityAdapter._response_id(metadata)
        if response_id and response_id in names:
            return names[response_id]
        enum = _pb_uint(metadata, 11)
        if enum is None:
            stats = _pb_bytes(metadata, 9)
            if stats:
                enum = _pb_uint(stats, 1)
        if enum is not None:
            return f"model-{enum}"
        return response_id

    @staticmethod
    def _response_id(metadata: bytes) -> str | None:
        stats = _pb_bytes(metadata, 9)
        if not stats:
            return None
        return _pb_str(stats, 11)

    @staticmethod
    def _generation_models(
        conn: sqlite3.Connection, tables: set[str]
    ) -> dict[str, str]:
        """Map generation response ids to model names from ``gen_metadata``."""
        if "gen_metadata" not in tables:
            return {}
        try:
            rows = conn.execute("SELECT data FROM gen_metadata ORDER BY idx")
        except sqlite3.Error:
            return {}
        names: dict[str, str] = {}
        for row in rows:
            blob = row["data"]
            if not blob:
                continue
            chat = _pb_bytes(blob, 1) or bytes(blob)
            name = _pb_str(chat, 21) or _pb_str(chat, 19)
            usage = _pb_bytes(chat, 4)
            response_id = _pb_str(usage, 11) if usage else None
            if response_id is None:
                response_id = _pb_str(chat, 11)
            if response_id and name:
                names[response_id] = name
        return names
