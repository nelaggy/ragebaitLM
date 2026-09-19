"""Heuristics for telling real human prompts apart from injected context.

Every agent CLI injects synthetic "user" messages (environment blocks, tool
instructions, reminders, subagent prompts). Those would poison sentiment
analysis, so adapters run candidate text through here.
"""

from __future__ import annotations

import re

# XML-ish tags used to wrap machine-generated context. If a message *starts*
# with one of these it is treated as injected.
INJECTED_TAGS = [
    "recommended_plugins",
    "environment_context",
    "app-context",
    "user_instructions",
    "user_instructions",
    "plugin",
    "plugins",
    "system-reminder",
    "command-name",
    "command-message",
    "command-args",
    "local-commands",
    "user-prompt-submit-hook",
    "task-notification",
    "turn_aborted",
    "model_switch",
    "permissions",
    "skill",
    "available_skills",
    "workspace",
    "ide_context",
    "project_context",
]

INJECTED_TAG_RE = re.compile(
    r"^\s*<(" + "|".join(re.escape(t) for t in INJECTED_TAGS) + r")\b",
    re.IGNORECASE,
)

# Plain-text prefixes that indicate a synthetic message.
INJECTED_PREFIXES = [
    "# response annotations",
    "caveat: the messages below were generated",
    "this session is being continued from a previous conversation",
    "[agent]",
    "[request interrupted by user]",
    "<local-command",
]

# Marker Codex uses for user annotations on a previous response.
ANNOTATION_RE = re.compile(r":codex-annotation\{", re.IGNORECASE)
ANNOTATION_HEADER_RE = re.compile(r"#\s*Response annotations", re.IGNORECASE)

# A message that is only whitespace / punctuation carries no sentiment.
_MEANINGFUL_RE = re.compile(r"[A-Za-z0-9]")


def looks_injected(text: str) -> bool:
    """True if the text looks like machine-injected context rather than human input."""
    if not text:
        return True
    stripped = text.lstrip()
    if INJECTED_TAG_RE.match(stripped):
        return True
    lowered = stripped.lower()
    if any(lowered.startswith(p) for p in INJECTED_PREFIXES):
        return True
    if ANNOTATION_RE.search(text):
        # Codex annotations embed real user comments but the wrapper itself is
        # synthetic; treat the whole block as a revision signal, not sentiment.
        return True
    return False


def has_annotation(text: str) -> bool:
    return bool(ANNOTATION_RE.search(text) or ANNOTATION_HEADER_RE.search(text))


def is_human_prompt(text: str, min_chars: int = 1) -> bool:
    """True if text is plausibly a real human message worth scoring."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    if not _MEANINGFUL_RE.search(stripped):
        return False
    return not looks_injected(stripped)


def text_from_content(content) -> str:
    """Extract plain text from a message content field.

    Handles: str, list of blocks with ``text``/``content``, and dicts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return text_from_content(content["content"])
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("image", "image_url"):
                    continue
                txt = block.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
                elif "content" in block:
                    parts.append(text_from_content(block["content"]))
        return "\n".join(p for p in parts if p)
    return ""
