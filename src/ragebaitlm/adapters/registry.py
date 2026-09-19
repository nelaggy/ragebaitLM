"""Adapter registry and default harness paths."""

from __future__ import annotations

from .base import Adapter
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .opencode import OpenCodeAdapter
from .pi import PiAdapter

ADAPTERS: dict[str, type] = {
    "codex": CodexAdapter,
    "claude": ClaudeAdapter,
    "opencode": OpenCodeAdapter,
    "pi": PiAdapter,
}


def get_adapter(name: str, home=None) -> Adapter:
    try:
        factory = ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown harness {name!r}. Choose from {sorted(ADAPTERS)}"
        ) from None
    return factory(home) if home is not None else factory()


def all_adapters(home=None) -> list[Adapter]:
    return [get_adapter(name, home) for name in ADAPTERS]
