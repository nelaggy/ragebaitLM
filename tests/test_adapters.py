from pathlib import Path

from ragebaitlm.adapters.claude import ClaudeAdapter
from ragebaitlm.adapters.codex import CodexAdapter
from ragebaitlm.adapters.opencode import OpenCodeAdapter
from ragebaitlm.adapters.pi import PiAdapter
from ragebaitlm.model import Kind


def _by_id(sessions):
    return {s.id: s for s in sessions}


def test_codex(fixtures_dir: Path):
    adapter = CodexAdapter()
    sessions = list(adapter.iter_sessions(paths=[fixtures_dir / "codex"]))
    assert len(sessions) == 1
    session = sessions[0]
    assert session.id == "sess-codex-1"
    assert session.project_path == "/tmp/proj"

    humans = [m for m in session.messages if m.kind == Kind.HUMAN]
    assert [m.text for m in humans] == [
        "fix the build",
        "no this is STILL broken, I said fix the build!!! why",
    ]

    context = [m for m in session.messages if m.kind == Kind.CONTEXT]
    assert any(m.text.startswith("<environment_context>") for m in context)

    assistant = [m for m in session.messages if m.kind == Kind.ASSISTANT]
    assert assistant[0].model == "gpt-6-astra"
    # Second human turn follows a model switch.
    assert humans[1].model == "gpt-5.6-terra"

    assert any(m.text.startswith("# Response annotations") for m in context)


def test_claude(fixtures_dir: Path):
    adapter = ClaudeAdapter()
    sessions = _by_id(
        adapter.iter_sessions(paths=[fixtures_dir / "claude" / "projects"])
    )
    assert len(sessions) == 2
    main = sessions["sess-claude-1"]
    assert main.is_subagent is False
    humans = [m for m in main.messages if m.kind == Kind.HUMAN]
    assert [m.text for m in humans] == ["build a login page", "no that's wrong, do it again"]
    assistants = [m for m in main.messages if m.kind == Kind.ASSISTANT]
    assert all(m.model == "claude-opus-4-6" for m in assistants)

    sub = sessions["sess-claude-1:sub1"]
    assert sub.is_subagent is True
    assert all(m.kind != Kind.HUMAN for m in sub.messages)
    assert any(m.kind == Kind.SUBAGENT_PROMPT for m in sub.messages)


def test_pi(fixtures_dir: Path):
    adapter = PiAdapter()
    sessions = list(adapter.iter_sessions(paths=[fixtures_dir / "pi" / "agent" / "sessions"]))
    assert len(sessions) == 1
    session = sessions[0]
    humans = [m for m in session.messages if m.kind == Kind.HUMAN]
    assert [m.text for m in humans] == [
        "add a healthcheck endpoint",
        "let's try a different approach entirely",
    ]
    # The abandoned branch message is not scored.
    assert all("still 500s" not in m.text for m in humans)
    assert session.models[-1] == "gpt-6-astra"


def test_opencode(opencode_db: Path):
    adapter = OpenCodeAdapter()
    sessions = _by_id(adapter.iter_sessions(paths=[opencode_db]))
    main = sessions["ses_main"]
    humans = [m for m in main.messages if m.kind == Kind.HUMAN]
    assert [m.text for m in humans] == [
        "please fix the parser",
        "no still broken, why???",
        "ok that works now thanks",
    ]
    assistants = [m for m in main.messages if m.kind == Kind.ASSISTANT]
    assert assistants[-1].model == "claude-opus-4-6"

    sub = sessions["ses_sub"]
    assert sub.is_subagent is True
    assert any(m.kind == Kind.SUBAGENT_PROMPT for m in sub.messages)
