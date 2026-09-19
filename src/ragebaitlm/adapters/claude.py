"""Claude Code adapter.

Reads ``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`` plus subagent
transcripts under ``<session-uuid>/subagents/agent-*.jsonl``.

Human prompts are ``user`` records whose ``message.content`` is a string.
``user`` records whose content is a list are tool results (skipped), and
``isSidechain`` records are subagent traffic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt, text_from_content
from ..model import Kind, NormalizedMessage, NormalizedSession, RevisionSignal
from .base import collect_files, iter_jsonl, parse_iso

INTERRUPT_MARKERS = (
    "[request interrupted by user]",
    "[request interrupted by user for tool use]",
)


class ClaudeAdapter:
    name = "claude"

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else Path.home() / ".claude"
        self.default_path = self.home / "projects"

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        roots = [Path(p) for p in paths] if paths else [self.default_path]
        for root in roots:
            if not root.exists():
                continue
            if root.is_file() and root.suffix == ".jsonl":
                files = [root]
            else:
                files = collect_files(root, "*.jsonl")
            for file in files:
                is_subagent = "subagents" in file.parts
                session = self._parse(file, is_subagent, since)
                if session and session.messages:
                    yield session.finalize()

    def _parse(
        self, path: Path, is_subagent: bool, since: int | None
    ) -> NormalizedSession | None:
        records = list(iter_jsonl(path))
        if not records:
            return None

        first = records[0]
        session_id = str(first.get("sessionId") or path.stem)
        cwd = None
        agent_id = None
        for record in records:
            cwd = cwd or record.get("cwd")
            agent_id = agent_id or record.get("agentId")
        if is_subagent:
            session_id = f"{session_id}:{agent_id or path.stem}"

        session = NormalizedSession(
            id=session_id,
            harness=self.name,
            project_path=cwd,
            is_subagent=is_subagent,
            raw_path=str(path),
        )

        seq = 0
        for record in records:
            rtype = record.get("type")
            if rtype not in ("user", "assistant"):
                continue
            if record.get("isSidechain") and not is_subagent:
                # Embedded sidechain traffic is covered by the subagent files.
                continue
            message = record.get("message") or {}
            ts = parse_iso(record.get("timestamp"))
            role = message.get("role") or rtype

            if rtype == "user":
                content = message.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                ):
                    continue
                text = text_from_content(content)
                if not text.strip():
                    continue
                lowered = text.lower()
                if any(marker in lowered for marker in INTERRUPT_MARKERS):
                    session.revisions.append(
                        RevisionSignal(
                            session_id=session_id,
                            signal_type="interrupt",
                            ts=ts,
                            target_event_id=record.get("uuid"),
                            meta={"text": text[:200]},
                        )
                    )
                    continue
                if is_subagent:
                    kind = Kind.SUBAGENT_PROMPT
                elif is_human_prompt(text):
                    kind = Kind.HUMAN
                else:
                    kind = Kind.CONTEXT
                if session.title is None and kind == Kind.HUMAN:
                    session.title = text.strip().splitlines()[0][:120]
            else:
                text = text_from_content(message.get("content"))
                if not text.strip():
                    continue
                kind = Kind.ASSISTANT

            session.add(
                NormalizedMessage(
                    session_id=session_id,
                    seq=seq,
                    ts=ts,
                    role=role,
                    kind=kind,
                    text=text,
                    model=message.get("model"),
                    stop_reason=message.get("stop_reason"),
                    source_event_id=record.get("uuid"),
                    is_subagent=is_subagent,
                )
            )
            seq += 1

        return session
