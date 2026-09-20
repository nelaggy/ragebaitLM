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


@pytest.fixture
def cursor_root(tmp_path: Path) -> Path:
    root = tmp_path / "cursor"
    user = root / "User"

    global_db = user / "globalStorage" / "state.vscdb"
    global_db.parent.mkdir(parents=True)
    conn = sqlite3.connect(global_db)
    conn.executescript(
        """
        CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB);
        CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB);
        """
    )
    composer = {
        "composerId": "comp-1",
        "name": "",
        "createdAt": 1789814123000,
        "lastUpdatedAt": 1789814193000,
        "modelConfig": {"modelName": "composer-1"},
        "fullConversationHeadersOnly": [
            {"bubbleId": "b1", "type": 1},
            {"bubbleId": "b2", "type": 2},
            {"bubbleId": "b3", "type": 1},
            {"bubbleId": "b4", "type": 2},
        ],
    }
    conn.execute(
        "INSERT INTO cursorDiskKV VALUES (?,?)",
        ("composerData:comp-1", json.dumps(composer)),
    )
    bubbles = {
        "b1": {"type": 1, "text": "please fix the parser", "timestamp": 1789814123100,
               "modelInfo": {"modelName": "claude-4.5-sonnet"}},
        "b2": {"type": 2, "text": "Done.", "timestamp": 1789814133100,
               "modelInfo": {"modelName": "claude-4.5-sonnet"}},
        "b3": {"type": 1, "text": "no still broken, why???", "timestamp": 1789814143100,
               "modelInfo": {"modelName": "gpt-5"}},
        "b4": {"type": 2, "text": "Sorry.", "timestamp": 1789814153100,
               "modelInfo": {"modelName": "gpt-5"}},
    }
    for bid, bubble in bubbles.items():
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?,?)",
            (f"bubbleId:comp-1:{bid}", json.dumps(bubble)),
        )
    conn.commit()
    conn.close()

    ws = user / "workspaceStorage" / "hash1"
    ws.mkdir(parents=True)
    (ws / "workspace.json").write_text(
        json.dumps({"folder": "file:///tmp/proj"}), encoding="utf-8"
    )
    wconn = sqlite3.connect(ws / "state.vscdb")
    wconn.executescript(
        """
        CREATE TABLE ItemTable (key TEXT UNIQUE, value BLOB);
        CREATE TABLE cursorDiskKV (key TEXT UNIQUE, value BLOB);
        """
    )
    wconn.execute(
        "INSERT INTO ItemTable VALUES (?,?)",
        ("composer.composerData",
         json.dumps({"allComposers": [{"composerId": "comp-1", "name": "Fix parser"}]})),
    )
    wconn.commit()
    wconn.close()

    transcripts = root / "projects" / "tmp-proj" / "agent-transcripts"
    transcripts.mkdir(parents=True)
    records = [
        {"role": "user",
         "message": {"content": [{"type": "text", "text": "please fix the parser"}]}},
        {"role": "assistant",
         "message": {"model": "claude-4.5-sonnet",
                     "content": [{"type": "text", "text": "Done."}]}},
        {"type": "turn_ended"},
        {"role": "user",
         "message": {"content": [{"type": "text", "text": "no still broken, why???"}]}},
        {"role": "assistant",
         "message": {"model": "gpt-5", "content": [{"type": "text", "text": "Sorry."}]}},
    ]
    (transcripts / "cli-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return root


def _pb_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _pb_key(field: int, wire: int) -> bytes:
    return _pb_varint((field << 3) | wire)


def _pb_uint(field: int, value: int) -> bytes:
    return _pb_key(field, 0) + _pb_varint(value)


def _pb_bytes(field: int, data: bytes) -> bytes:
    return _pb_key(field, 2) + _pb_varint(len(data)) + data


def _pb_str(field: int, text: str) -> bytes:
    return _pb_bytes(field, text.encode("utf-8"))


def _pb_timestamp(field: int, seconds: int) -> bytes:
    inner = _pb_uint(1, seconds) + _pb_uint(2, 0)
    return _pb_bytes(field, inner)


def _ag_metadata(
    seconds: int, model_id: int | None = None, response_id: str | None = None
) -> bytes:
    data = _pb_timestamp(1, seconds) + _pb_uint(3, 4)
    if response_id:
        data += _pb_bytes(9, _pb_str(11, response_id))
    if model_id is not None:
        data += _pb_uint(11, model_id)
    return data


def _ag_step(idx: int, step_type: int, metadata: bytes, payload_field: int,
             payload_body: bytes) -> tuple:
    payload = (
        _pb_uint(1, step_type)
        + _pb_uint(4, 3)
        + _pb_bytes(5, metadata)
        + _pb_bytes(payload_field, payload_body)
    )
    return (idx, step_type, 3, metadata, payload)


@pytest.fixture
def antigravity_root(tmp_path: Path) -> Path:
    root = tmp_path / "antigravity"
    conversations = root / "conversations"
    conversations.mkdir(parents=True)

    db = conversations / "conv-1.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE trajectory_meta (
            trajectory_id TEXT, cascade_id TEXT, trajectory_type INTEGER,
            source INTEGER, PRIMARY KEY (trajectory_id)
        );
        CREATE TABLE steps (
            idx INTEGER PRIMARY KEY, step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0, metadata BLOB, step_payload BLOB
        );
        CREATE TABLE gen_metadata (
            idx INTEGER PRIMARY KEY, data BLOB, size INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE trajectory_metadata_blob (
            id TEXT DEFAULT "main", data BLOB, PRIMARY KEY (id)
        );
        """
    )
    conn.execute(
        "INSERT INTO trajectory_meta VALUES (?,?,?,?)",
        ("traj-1", "conv-1", 0, 28),
    )

    workspace = _pb_str(1, "file:///tmp/proj") + _pb_str(4, "main")
    main = (
        _pb_bytes(1, workspace)
        + _pb_timestamp(2, 1789814123)
        + _pb_str(6, "conv-1")
    )
    conn.execute(
        "INSERT INTO trajectory_metadata_blob VALUES (?,?)", ("main", main)
    )

    # gen_metadata: model-4 renders as "Gemini 3.5 Flash" via response id.
    usage = _pb_str(11, "resp-1")
    chat_model = (
        _pb_bytes(4, usage)
        + _pb_str(19, "gemini-3-flash-a")
        + _pb_str(21, "Gemini 3.5 Flash")
    )
    conn.execute(
        "INSERT INTO gen_metadata VALUES (?,?,?)",
        (0, _pb_bytes(1, chat_model), 0),
    )
    usage2 = _pb_str(11, "resp-2")
    chat_model2 = _pb_bytes(4, usage2) + _pb_str(21, "Gemini 3 Pro")
    conn.execute(
        "INSERT INTO gen_metadata VALUES (?,?,?)",
        (1, _pb_bytes(1, chat_model2), 0),
    )

    user_body = _pb_str(2, "please fix the parser") + _pb_bytes(
        3, _pb_str(1, "please fix the parser")
    )
    assistant_body = _pb_str(1, "Done.") + _pb_str(3, "thinking")
    steps = [
        _ag_step(0, 14, _ag_metadata(1789814123), 19, user_body),
        _ag_step(1, 15, _ag_metadata(1789814124, 4, "resp-1"), 20, assistant_body),
        _ag_step(
            2, 14, _ag_metadata(1789814133),
            19, _pb_str(2, "no still broken, why???"),
        ),
        _ag_step(
            3, 15, _ag_metadata(1789814134, 9), 20, _pb_str(1, "Sorry."),
        ),
    ]
    conn.executemany(
        "INSERT INTO steps (idx, step_type, status, metadata, step_payload) "
        "VALUES (?,?,?,?,?)",
        steps,
    )
    conn.commit()
    conn.close()

    # A foreign SQLite database must be ignored rather than crash the sync.
    other = conversations / "not-antigravity.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE unrelated (idx INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return root


@pytest.fixture
def vscode_root(tmp_path: Path) -> Path:
    root = tmp_path / "Code" / "User"
    ws = root / "workspaceStorage" / "hash1"
    chats = ws / "chatSessions"
    chats.mkdir(parents=True)
    (ws / "workspace.json").write_text(
        json.dumps({"folder": "file:///tmp/proj"}), encoding="utf-8"
    )

    request1 = {
        "requestId": "r1",
        "timestamp": 1789814123000,
        "modelId": "copilot/claude-sonnet-4.6",
        "message": {"text": "please fix the parser"},
        "response": [{"value": "Done."}],
    }
    request2 = {
        "requestId": "r2",
        "timestamp": 1789814133000,
        "modelId": "copilot/auto",
        "result": {"metadata": {"resolvedModel": "gpt-5.4-mini-2026-03-17"}},
        "message": {"text": "no still broken, why???"},
        "response": [{"kind": "markdownContent", "content": {"value": "Sorry."}}],
    }
    request3 = {
        "requestId": "r3",
        "timestamp": 1789815000000,
        "modelId": "github.copilot-chat/claude-sonnet-4",
        "message": {"text": "add a test"},
        "response": [{"value": "Sure."}],
    }
    request4 = {
        "requestId": "r4",
        "timestamp": 1789815010000,
        "modelId": "copilot/auto",
        "message": {"text": "try again"},
        "response": [{"value": "Working."}],
    }
    lines = [
        {"kind": 0, "v": {"version": 3, "sessionId": "sess-vscode-1",
                          "creationDate": 1789814123000, "requests": []}},
        {"kind": 1, "k": ["customTitle"], "v": "Fix parser"},
        {"kind": 2, "k": ["requests"], "v": [request1]},
        {"kind": 1, "k": ["requests", 0, "completionTokens"], "v": 42},
        {"kind": 2, "k": ["requests"], "v": [request2]},
    ]
    (chats / "sess-vscode-1.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    (chats / "sess-vscode-1.json").write_text(
        json.dumps({"sessionId": "sess-vscode-1",
                    "requests": [{"message": {"text": "legacy"}}]}),
        encoding="utf-8",
    )
    (chats / "sess-vscode-2.json").write_text(
        json.dumps({"sessionId": "sess-vscode-2", "creationDate": 1789815000000,
                    "requests": [request3, request4]}),
        encoding="utf-8",
    )
    return root
