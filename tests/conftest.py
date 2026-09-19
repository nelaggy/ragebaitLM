import json
import sqlite3
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def opencode_db(tmp_path: Path) -> Path:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, parent_id TEXT, slug TEXT, directory TEXT,
            title TEXT, version TEXT, agent TEXT, model TEXT,
            time_created INTEGER, time_updated INTEGER, revert TEXT
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
            time_created INTEGER, time_updated INTEGER, data TEXT
        );
        CREATE TABLE event (
            id TEXT PRIMARY KEY, aggregate_id TEXT, seq INTEGER, type TEXT, data TEXT
        );
        """
    )

    def add_session(sid, title, directory, parent=None, revert=None, model=None):
        conn.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, parent, "slug", directory, title, "1.0", "build",
             model, 1789814123588, 1789814193588, revert),
        )

    def add_message(mid, sid, role, ts, model=None, provider=None, agent=None, finish=None):
        data = {"role": role, "time": {"created": ts}}
        if role == "assistant":
            data.update({"modelID": model, "providerID": provider, "agent": agent})
            if finish:
                data["finish"] = finish
        elif role == "user" and model:
            data["model"] = {"modelID": model, "providerID": provider}
            data["agent"] = agent
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?)",
            (mid, sid, ts, ts, json.dumps(data)),
        )

    def add_text(pid, mid, sid, text, ts):
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (pid, mid, sid, ts, ts, json.dumps({"type": "text", "text": text})),
        )

    add_session("ses_main", "main task", "/tmp/proj",
                model=json.dumps({"id": "gpt-6-astra", "providerID": "opencode"}))
    add_message("m1", "ses_main", "user", 1000, model="gpt-6-astra", provider="opencode")
    add_text("p1", "m1", "ses_main", "please fix the parser", 1000)
    add_message("m2", "ses_main", "assistant", 1010,
                model="gpt-6-astra", provider="opencode", agent="build", finish="stop")
    add_text("p2", "m2", "ses_main", "Done.", 1010)
    add_message("m3", "ses_main", "user", 1020, model="gpt-6-astra", provider="opencode")
    add_text("p3", "m3", "ses_main", "no still broken, why???", 1020)
    add_message("m4", "ses_main", "assistant", 1030,
                model="claude-opus-4-6", provider="anthropic", agent="build", finish="stop")
    add_text("p4", "m4", "ses_main", "Sorry.", 1030)
    add_message("m5", "ses_main", "user", 1040, model="claude-opus-4-6", provider="anthropic")
    add_text("p5", "m5", "ses_main", "ok that works now thanks", 1040)

    add_session("ses_sub", "subagent task", "/tmp/proj", parent="ses_main",
                model=json.dumps({"id": "gpt-6-astra", "providerID": "opencode"}))
    add_message("s1", "ses_sub", "user", 1015, model="gpt-6-astra", provider="opencode")
    add_text("sp1", "s1", "ses_sub", "Explore the codebase thoroughly.", 1015)
    add_message("s2", "ses_sub", "assistant", 1016,
                model="gpt-6-astra", provider="opencode", agent="explore", finish="stop")
    add_text("sp2", "s2", "ses_sub", "Found things.", 1016)

    conn.execute(
        "INSERT INTO event VALUES (?,?,?,?,?)",
        ("e1", "ses_main", 1, "message.removed.1",
         json.dumps({"sessionID": "ses_main", "messageID": "m3"})),
    )
    conn.commit()
    conn.close()
    return db
