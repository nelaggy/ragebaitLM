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
