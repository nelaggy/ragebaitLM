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

    turns = analyze_mod.load_turns(store)
    assert len(turns) == 2
    assert turns.iloc[0]["model"] == "gpt-6-astra"
    assert turns["mood_score"].min() < -0.2
    assert turns["mood_score"].max() >= turns["mood_score"].min()
    store.close()


def test_pipeline_opencode_and_report(tmp_path: Path, opencode_db: Path):
    store = Store(tmp_path / "test.db")
    store.init_schema()
    engine = build_engine("lexicon")
    counts = sync(store, OpenCodeAdapter(), engine, paths=[opencode_db])
    assert counts["human"] == 3

    result = analyze_mod.analyze(store)
    assert result["overall"]["n_turns"] == 3
    assert result["by_model"] == []  # groups below MIN_GROUP_TURNS are dropped
    assert result["by_harness"] == []
    assert "by_agent" not in result
    assert sum(result["mood_histogram"]["counts"]) == 3

    out = build_report(result, tmp_path / "report.html")
    assert out.exists()
    assert "ragebaitLM" in out.read_text(encoding="utf-8")
    store.close()
