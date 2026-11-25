import json
import sqlite3
from pathlib import Path

from ragebaitlm import analyze as analyze_mod
from ragebaitlm.adapters.codex import CodexAdapter
from ragebaitlm.adapters.opencode import OpenCodeAdapter
from ragebaitlm.pipeline import build_engine, sync
from ragebaitlm.report import build_report
from ragebaitlm.store import Store


def test_pipeline_codex(tmp_path: Path, fixtures_dir: Path):
    store = Store(tmp_path / "test.db")
    store.init_schema()
    engine = build_engine("lexicon")
    counts = sync(store, CodexAdapter(), engine, paths=[fixtures_dir / "codex"])
    assert counts["human"] == 2
    assert counts["sessions"] == 1

    # The opening prompt is stored but excluded from the report: nothing
    # precedes it, so there is no model to blame.
    assert len(store.scored_messages()) == 2
    messages = analyze_mod.load_messages(store)
    assert len(messages) == 1
    assert messages.iloc[0]["model"] == "gpt-6-astra"
    assert messages["mood_score"].min() < -0.2
    store.close()


def test_pipeline_opencode_and_report(tmp_path: Path, opencode_db: Path):
    store = Store(tmp_path / "test.db")
    store.init_schema()
    engine = build_engine("lexicon")
    counts = sync(store, OpenCodeAdapter(), engine, paths=[opencode_db])
    assert counts["human"] == 3

    result = analyze_mod.analyze(store)
    # The first prompt has no preceding assistant and is excluded.
    assert result["overall"]["n_messages"] == 2
    assert result["by_model"] == []  # groups below MIN_GROUP_MESSAGES are dropped
    assert result["by_harness"] == []
    assert "by_agent" not in result
    assert sum(result["mood_histogram"]["counts"]) == 2

    out = build_report(result, tmp_path / "report.html")
    assert out.exists()
    assert "ragebaitLM" in out.read_text(encoding="utf-8")
    store.close()


def _opencode_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
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
        """
    )
    for sid, created in (("s1", 1000), ("s2", 2000)):
        conn.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, None, "slug", "/tmp/proj", sid, "1.0", "build", None,
             created, created, None),
        )
        for seq, role in enumerate(("user", "assistant")):
            mid = f"{sid}-m{seq}"
            ts = created + seq
            conn.execute(
                "INSERT INTO message VALUES (?,?,?,?,?)",
                (mid, sid, ts, ts, json.dumps({"role": role, "time": {"created": ts}})),
            )
            text = "please fix the parser" if role == "user" else "Done."
            conn.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (f"{mid}-p", mid, sid, ts, ts, json.dumps({"type": "text", "text": text})),
            )
    conn.commit()
    conn.close()
    return path


def test_progressive_sync_skips_old_sessions(tmp_path: Path):
    db = _opencode_db(tmp_path / "opencode.db")
    store = Store(tmp_path / "test.db")
    store.init_schema()
    engine = build_engine("lexicon")
    adapter = OpenCodeAdapter()

    first = sync(store, adapter, engine, paths=[db])
    assert first["sessions"] == 2
    assert store.newest_session_at("opencode") == 2000

    # A brand new session appears after the last sync.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("s3", None, "slug", "/tmp/proj", "s3", "1.0", "build", None, 3000, 3000, None),
    )
    for seq, role in enumerate(("user", "assistant")):
        mid = f"s3-m{seq}"
        ts = 3000 + seq
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?,?)",
            (mid, "s3", ts, ts, json.dumps({"role": role, "time": {"created": ts}})),
        )
        text = "please fix the parser" if role == "user" else "Done."
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?,?)",
            (f"{mid}-p", mid, "s3", ts, ts, json.dumps({"type": "text", "text": text})),
        )
    conn.commit()
    conn.close()

    seen: list[str] = []
    counts = sync(
        store,
        adapter,
        engine,
        paths=[db],
        progressive=True,
        progress=lambda session: seen.append(session.id),
    )
    # s1 (older than the newest stored session) is skipped; s2 is rescanned so
    # further turns on the still-active session are picked up.
    assert set(seen) == {"s2", "s3"}
    assert counts["sessions"] == 2

    # --full rescans everything.
    seen.clear()
    counts = sync(store, adapter, engine, paths=[db], progress=lambda s: seen.append(s.id))
    assert set(seen) == {"s1", "s2", "s3"}

    # An explicit --since overrides the watermark, reaching back past it.
    seen.clear()
    counts = sync(
        store,
        adapter,
        engine,
        paths=[db],
        since=500,
        progressive=True,
        progress=lambda s: seen.append(s.id),
    )
    assert set(seen) == {"s1", "s2", "s3"}
    store.close()

