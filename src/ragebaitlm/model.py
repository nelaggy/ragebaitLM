"""Normalized data model shared across all harness adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    """Normalized message kind."""

    HUMAN = "human_prompt"
    ASSISTANT = "assistant_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    CONTEXT = "context_injected"
    SUBAGENT_PROMPT = "subagent_prompt"
    MODEL_CHANGE = "model_change"
    REVISION = "revision"


@dataclass
class NormalizedMessage:
    session_id: str
    seq: int
    role: str
    kind: Kind
    ts: int | None = None
    text: str = ""
    model: str | None = None
    provider: str | None = None
    source_event_id: str | None = None
    stop_reason: str | None = None
    is_subagent: bool = False

    @property
    def is_human(self) -> bool:
        return self.kind == Kind.HUMAN

    @property
    def is_assistant(self) -> bool:
        return self.kind == Kind.ASSISTANT


@dataclass
class RevisionSignal:
    """A revert / undo / edit / branch signal observed in a session."""

    session_id: str
    signal_type: str  # revert | message_removed | branch | annotation | interrupt
    ts: int | None = None
    target_event_id: str | None = None
    model_at_time: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedSession:
    id: str
    harness: str
    project_path: str | None = None
    title: str | None = None
    started_at: int | None = None
    ended_at: int | None = None
    is_subagent: bool = False
    parent_session_id: str | None = None
    raw_path: str | None = None
    models: list[str] = field(default_factory=list)
    messages: list[NormalizedMessage] = field(default_factory=list)
    revisions: list[RevisionSignal] = field(default_factory=list)

    def add(self, message: NormalizedMessage) -> None:
        self.messages.append(message)

    def finalize(self) -> "NormalizedSession":
        """Sort messages by seq/timestamp and derive summary metadata."""
        self.messages.sort(key=lambda m: (m.seq, m.ts or 0))
        seen: list[str] = []
        for m in self.messages:
            if m.model and m.model not in seen:
                seen.append(m.model)
        self.models = seen
        stamps = [m.ts for m in self.messages if m.ts]
        if stamps:
            if self.started_at is None:
                self.started_at = min(stamps)
            if self.ended_at is None:
                self.ended_at = max(stamps)
        return self
