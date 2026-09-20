"""VS Code Copilot Chat adapter.

Reads Copilot Chat session files from the VS Code ``User`` directory. Sessions
live at ``workspaceStorage/<hash>/chatSessions/<sessionId>.json`` (pre-v1.109) or
``.jsonl`` (v1.109+). The newer format is an append-only mutation log that is
replayed into the same session object; when both exist for one id the ``.jsonl``
file wins.

Each session holds ``requests[]``; a request's ``message.text`` is the human
prompt and its ``response[]`` parts of bare markdown make up the assistant
reply. VS Code publishes no formal schema, so parsing is best-effort.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from ..filters import is_human_prompt, text_from_content
from ..model import Kind, NormalizedMessage, NormalizedSession
from .base import app_data_dir, iter_jsonl, to_ms, uri_to_path

HOSTS = ("Code", "Code - Insiders", "VSCodium")
REMOTE_USER_DIRS = (
    "~/.vscode-server/data/User",
    "~/.vscode-server-insiders/data/User",
)
DEFAULT_PROVIDER = "copilot"
AUTO_LABEL = "copilot-auto"
# Trailing dated snapshot, e.g. gpt-5.4-mini-2026-03-17 or claude-haiku-4-5-20251001.
SNAPSHOT_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")


def _normalize_model(model: str) -> str:
    """Collapse dated model snapshots onto their family name."""
    return SNAPSHOT_RE.sub("", model)



def _default_user_dirs() -> list[Path]:
    dirs = [
        app_data_dir(host) / "User"
        for host in HOSTS
        if (app_data_dir(host) / "User").exists()
    ]
    dirs.extend(
        path
        for path in (Path(p).expanduser() for p in REMOTE_USER_DIRS)
        if path.exists()
    )
    return dirs or [app_data_dir("Code") / "User"]


def _set_path(state: object, path: object, value: object) -> None:
    if not isinstance(path, list) or not path:
        return
    current = state
    for key in path[:-1]:
        try:
            current = current[key]
        except (TypeError, KeyError, IndexError):
            return
    if not isinstance(current, (dict, list)):
        return
    try:
        current[path[-1]] = value
    except (TypeError, IndexError):
        return


def _push_path(state: object, path: object, values: object, start: object) -> None:
    if not isinstance(path, list) or not path or not isinstance(state, dict):
        return
    current: object = state
    for key in path[:-1]:
        try:
            current = current[key]
        except (TypeError, KeyError, IndexError):
            return
    if not isinstance(current, dict):
        return
    key = path[-1]
    array = current.get(key)
    if not isinstance(array, list):
        array = []
    if isinstance(start, int):
        del array[start:]
    if isinstance(values, list) and values:
        array.extend(values)
    current[key] = array


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, list):
        return ""
    parts: list[str] = []
    for part in response:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if "kind" not in part:
            # A bare MarkdownString: {value, supportHtml, ...}
            value = part.get("value")
        elif part.get("kind") == "markdownContent":
            content = part.get("content")
            value = content.get("value") if isinstance(content, dict) else content
        else:
            continue
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


class VSCodeAdapter:
    name = "vscode"

    def __init__(self, home: Path | None = None) -> None:
        self.user_dirs = [Path(home)] if home is not None else _default_user_dirs()
        self.default_path = self.user_dirs[0]

    def iter_sessions(
        self, paths: Iterable[Path] | None = None, since: int | None = None
    ) -> Iterator[NormalizedSession]:
        roots = [Path(p) for p in paths] if paths else self.user_dirs
        for root in roots:
            for file in self._session_files(root):
                session = self._parse(file, since)
                if session and session.messages:
                    yield session.finalize()

    @staticmethod
    def _session_files(root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix in (".json", ".jsonl") else []
        if not root.is_dir():
            return []
        if root.name == "chatSessions":
            directories = [root]
        else:
            directories = [d for d in root.rglob("chatSessions") if d.is_dir()]
        best: dict[Path, Path] = {}
        for directory in directories:
            for file in [*directory.glob("*.json"), *directory.glob("*.jsonl")]:
                key = file.with_suffix("")
                existing = best.get(key)
                if existing is None or (file.suffix == ".jsonl" and existing.suffix != ".jsonl"):
                    best[key] = file
        return sorted(best.values())

    def _parse(self, path: Path, since: int | None) -> NormalizedSession | None:
        data = self._load(path)
        if not isinstance(data, dict):
            return None

        started = to_ms(data.get("creationDate"))
        if since is not None and started is not None and started < since:
            return None

        session = NormalizedSession(
            id=str(data.get("sessionId") or path.stem),
            harness=self.name,
            project_path=self._workspace_folder(path),
            title=data.get("customTitle") or None,
            started_at=started,
            raw_path=str(path),
        )

        requests = data.get("requests")
        if not isinstance(requests, list):
            return None

        seq = 0
        ended = started
        for request in requests:
            if not isinstance(request, dict):
                continue
            ts = to_ms(request.get("timestamp"))
            if ts:
                ended = ts if ended is None else max(ended, ts)

            # The model selected for this turn is known at request time; keep it
            # on the human message too so attribution records the current model.
            model, provider = self._model(request)

            text = self._request_text(request)
            if text.strip():
                if request.get("isSystemInitiated"):
                    kind = Kind.CONTEXT
                else:
                    kind = Kind.HUMAN if is_human_prompt(text) else Kind.CONTEXT
                session.add(
                    NormalizedMessage(
                        session_id=session.id,
                        seq=seq,
                        ts=ts,
                        role="user",
                        kind=kind,
                        text=text,
                        model=model,
                        provider=provider,
                        source_event_id=request.get("requestId"),
                    )
                )
                seq += 1
                if kind == Kind.HUMAN and session.title is None:
                    session.title = text.strip().splitlines()[0][:120]

            reply = _response_text(request.get("response"))
            if not reply.strip() and not model:
                continue
            session.add(
                NormalizedMessage(
                    session_id=session.id,
                    seq=seq,
                    ts=to_ms(request.get("responseTimestamp")) or ts,
                    role="assistant",
                    kind=Kind.ASSISTANT,
                    text=reply,
                    model=model,
                    provider=provider,
                    source_event_id=request.get("responseId"),
                )
            )
            seq += 1

        session.ended_at = ended
        return session

    @staticmethod
    def _request_text(request: dict) -> str:
        message = request.get("message")
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            text = message.get("text")
            if isinstance(text, str) and text:
                return text
            return text_from_content(message.get("parts"))
        return ""

    @staticmethod
    def _model(request: dict) -> tuple[str | None, str | None]:
        resolved = None
        result = request.get("result")
        if isinstance(result, dict):
            metadata = result.get("metadata")
            if isinstance(metadata, dict):
                resolved = metadata.get("resolvedModel")

        provider = None
        raw = request.get("modelId")
        if isinstance(raw, str) and "/" in raw:
            provider = raw.split("/", 1)[0]
        if provider == "github.copilot-chat":
            provider = DEFAULT_PROVIDER

        model = resolved if isinstance(resolved, str) and resolved else None
        if model is None and isinstance(raw, str) and raw:
            model = raw.split("/", 1)[1] if "/" in raw else raw
            if model == "auto":
                # Auto-selection with no recorded resolved model: keep it as a
                # distinct label rather than presenting a fake concrete model.
                return AUTO_LABEL, provider or DEFAULT_PROVIDER
        if model is None:
            return None, None
        return _normalize_model(model), provider or DEFAULT_PROVIDER

    def _workspace_folder(self, path: Path) -> str | None:
        for parent in path.parents:
            manifest = parent / "workspace.json"
            if manifest.exists():
                return self._manifest_folder(manifest)
            if parent.name == "workspaceStorage":
                break
        return None

    @staticmethod
    def _manifest_folder(manifest: Path) -> str | None:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        for key in ("folder", "workspace"):
            resolved = uri_to_path(data.get(key))
            if resolved:
                return resolved
        return None

    @staticmethod
    def _load(path: Path) -> object | None:
        if path.suffix == ".jsonl":
            state: object | None = None
            for record in iter_jsonl(path):
                kind = record.get("kind")
                if kind == 0:
                    state = record.get("v")
                elif state is None:
                    continue
                elif kind == 1:
                    _set_path(state, record.get("k"), record.get("v"))
                elif kind == 2:
                    _push_path(state, record.get("k"), record.get("v"), record.get("i"))
                elif kind == 3:
                    _set_path(state, record.get("k"), None)
            return state
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
