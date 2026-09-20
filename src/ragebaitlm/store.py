"""SQLite storage for structured session statistics.

Only derived statistics are persisted by default: message text is hashed, not
stored. ``--cache-text`` opts in to keeping raw text in a separate table for
local experimentation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .model import NormalizedMessage, NormalizedSession
from .sentiment.mood import MoodResult
from .sentiment.types import SentimentResult

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS ingest_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    harness TEXT NOT NULL,
    path TEXT,
    engine TEXT,
    n_sessions INTEGER DEFAULT 0,
    n_messages INTEGER DEFAULT 0,
    n_human INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    error TEXT
);

CREATE TABLE IF NOT EXISTS harness_session (
    session_id TEXT PRIMARY KEY,
    harness TEXT NOT NULL,
    project_path TEXT,
    title TEXT,
    started_at INTEGER,
    ended_at INTEGER,
    is_subagent INTEGER DEFAULT 0,
    parent_session_id TEXT,
    models_json TEXT,
    raw_path TEXT,
    synced_at INTEGER
);

CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    harness TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts INTEGER,
    role TEXT,
    kind TEXT,
    model TEXT,
    provider TEXT,
    source_event_id TEXT,
    text_sha256 TEXT,
    char_len INTEGER,
    stop_reason TEXT,
    is_subagent INTEGER DEFAULT 0,
    UNIQUE (harness, session_id, seq)
);
CREATE INDEX IF NOT EXISTS event_session_idx ON event (session_id, seq);
CREATE INDEX IF NOT EXISTS event_kind_idx ON event (kind);

CREATE TABLE IF NOT EXISTS event_text (
    event_id INTEGER PRIMARY KEY REFERENCES event(id) ON DELETE CASCADE,
    text TEXT
);

CREATE TABLE IF NOT EXISTS message_stat (
    event_id INTEGER PRIMARY KEY REFERENCES event(id) ON DELETE CASCADE,
    sentiment_negative REAL,
    sentiment_neutral REAL,
    sentiment_positive REAL,
    sentiment_label TEXT,
    sentiment_compound REAL,
    mood_score REAL,
    mood_signals_json TEXT,
    engine TEXT,
    engine_version TEXT,
    scored_at INTEGER
);

CREATE TABLE IF NOT EXISTS turn_attribution (
    user_event_id INTEGER PRIMARY KEY REFERENCES event(id) ON DELETE CASCADE,
    prev_assistant_seq INTEGER,
    prev_model TEXT,
    prev_provider TEXT,
    prev_stop_reason TEXT,
    prev_latency_ms INTEGER,
    current_model_at_turn TEXT
);

"""


@dataclass
class Attribution:
    user_seq: int
    prev_assistant_seq: int | None = None
    prev_model: str | None = None
    prev_provider: str | None = None
    prev_stop_reason: str | None = None
    prev_latency_ms: int | None = None
    current_model_at_turn: str | None = None


@dataclass
class EventRow:
    message: NormalizedMessage
    sentiment: SentimentResult | None = None
    mood: MoodResult | None = None
    attribution: Attribution | None = None


@dataclass
class SessionRows:
    session: NormalizedSession
    rows: list[EventRow] = field(default_factory=list)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class Store:
    def __init__(self, path: str | Path, cache_text: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_text = cache_text
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def init_schema(self) -> None:
        has_meta = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        existing = (
            self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if has_meta
            else None
        )
        if existing is None or int(existing["value"]) != SCHEMA_VERSION:
            # Derived data only: rebuild from scratch on incompatible schema.
            for table in (
                "revision_signal",
                "turn_attribution",
                "message_stat",
                "event_text",
                "event",
                "harness_session",
                "ingest_run",
                "meta",
            ):
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    # -- ingest runs -----------------------------------------------------
    def start_run(self, harness: str, path: str | None, engine: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO ingest_run (started_at, harness, path, engine) VALUES (?, ?, ?, ?)",
            (int(time.time() * 1000), harness, path, engine),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        n_sessions: int,
        n_messages: int,
        n_human: int,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """UPDATE ingest_run
               SET finished_at=?, n_sessions=?, n_messages=?, n_human=?, status=?, error=?
               WHERE id=?""",
            (
                int(time.time() * 1000),
                n_sessions,
                n_messages,
                n_human,
                status,
                error,
                run_id,
            ),
        )
        self.conn.commit()

    # -- session persistence --------------------------------------------
    def persist_session(self, data: SessionRows, run_id: int | None = None) -> dict:
        session = data.session
        stats = {"messages": 0, "human": 0}
        now = int(time.time() * 1000)
        with self.conn:
            self.conn.execute(
                """INSERT INTO harness_session
                   (session_id, harness, project_path, title, started_at, ended_at,
                    is_subagent, parent_session_id, models_json, raw_path, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     harness=excluded.harness, project_path=excluded.project_path,
                     title=excluded.title, started_at=excluded.started_at,
                     ended_at=excluded.ended_at,
                     is_subagent=excluded.is_subagent,
                     parent_session_id=excluded.parent_session_id,
                     models_json=excluded.models_json, raw_path=excluded.raw_path,
                     synced_at=excluded.synced_at""",
                (
                    session.id,
                    session.harness,
                    session.project_path,
                    session.title,
                    session.started_at,
                    session.ended_at,
                    int(session.is_subagent),
                    session.parent_session_id,
                    json.dumps(session.models),
                    session.raw_path,
                    now,
                ),
            )
            self.conn.execute(
                "DELETE FROM event WHERE harness=? AND session_id=?",
                (session.harness, session.id),
            )

            for row in data.rows:
                msg = row.message
                cur = self.conn.execute(
                    """INSERT INTO event
                       (session_id, harness, seq, ts, role, kind, model, provider,
                        source_event_id, text_sha256, char_len, stop_reason, is_subagent)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        msg.session_id,
                        session.harness,
                        msg.seq,
                        msg.ts,
                        msg.role,
                        msg.kind.value,
                        msg.model,
                        msg.provider,
                        msg.source_event_id,
                        sha256(msg.text) if msg.text else None,
                        len(msg.text),
                        msg.stop_reason,
                        int(msg.is_subagent),
                    ),
                )
                event_id = int(cur.lastrowid)
                stats["messages"] += 1
                if msg.is_human:
                    stats["human"] += 1

                if self.cache_text and msg.text:
                    self.conn.execute(
                        "INSERT INTO event_text (event_id, text) VALUES (?, ?)",
                        (event_id, msg.text),
                    )
                if row.sentiment is not None:
                    self.conn.execute(
                        """INSERT INTO message_stat
                           (event_id, sentiment_negative, sentiment_neutral,
                            sentiment_positive, sentiment_label, sentiment_compound,
                            mood_score, mood_signals_json, engine,
                            engine_version, scored_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            event_id,
                            row.sentiment.negative,
                            row.sentiment.neutral,
                            row.sentiment.positive,
                            row.sentiment.label,
                            row.sentiment.compound,
                            row.mood.score if row.mood else None,
                            json.dumps(row.mood.signals) if row.mood else None,
                            row.mood.engine if row.mood else None,
                            row.sentiment.engine_version,
                            now,
                        ),
                    )
                if row.attribution is not None:
                    a = row.attribution
                    self.conn.execute(
                        """INSERT INTO turn_attribution
                           (user_event_id, prev_assistant_seq, prev_model, prev_provider,
                            prev_stop_reason, prev_latency_ms, current_model_at_turn)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            event_id,
                            a.prev_assistant_seq,
                            a.prev_model,
                            a.prev_provider,
                            a.prev_stop_reason,
                            a.prev_latency_ms,
                            a.current_model_at_turn,
                        ),
                    )

        return stats

    # -- queries ---------------------------------------------------------
    def scored_messages(self) -> list[sqlite3.Row]:
        """Human messages joined with stats + attribution."""
        return list(
            self.conn.execute(
                """SELECT e.session_id, e.harness, e.seq, e.ts, e.model AS event_model,
                          e.char_len,
                          ss.session_id AS s_session, ss.title, ss.project_path,
                          ms.sentiment_negative, ms.sentiment_neutral,
                          ms.sentiment_positive, ms.sentiment_label,
                          ms.sentiment_compound, ms.mood_score,
                          ms.mood_signals_json,
                          ta.prev_model, ta.prev_provider,
                          ta.prev_stop_reason, ta.prev_latency_ms,
                          ta.prev_assistant_seq,
                          ta.current_model_at_turn
                   FROM event e
                   LEFT JOIN harness_session ss ON ss.session_id = e.session_id
                   LEFT JOIN message_stat ms ON ms.event_id = e.id
                   LEFT JOIN turn_attribution ta ON ta.user_event_id = e.id
                   WHERE e.kind = 'human_prompt'
                   ORDER BY e.ts"""
            )
        )

    def model_markers(self) -> dict[str, list[dict]]:
        """Model change timeline per session (collapses consecutive duplicates)."""
        markers: dict[str, list[dict]] = {}
        for row in self.conn.execute(
            """SELECT session_id, ts, model FROM event
               WHERE model IS NOT NULL
                 AND kind IN ('assistant_text', 'model_change')
               ORDER BY session_id, ts, seq"""
        ):
            timeline = markers.setdefault(row["session_id"], [])
            if not timeline or timeline[-1]["model"] != row["model"]:
                timeline.append({"ts": row["ts"], "model": row["model"]})
        return markers

    def session_counts(self) -> dict[str, int]:
        row = self.conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN is_subagent THEN 1 ELSE 0 END) AS subagents
               FROM harness_session"""
        ).fetchone()
        return {"sessions": int(row["total"] or 0), "subagents": int(row["subagents"] or 0)}

    def session_summary(self) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM harness_session ORDER BY started_at DESC"
            )
        )
