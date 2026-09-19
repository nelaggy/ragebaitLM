"""Ingest pipeline: normalize -> score -> attribute -> persist."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .adapters.base import Adapter
from .model import Kind, NormalizedMessage, NormalizedSession
from .sentiment import (
    LexiconScorer,
    TransformerScorer,
    get_mood_engine,
    transformer_available,
)
from .sentiment.mood import LexiconMood, MoodEngine
from .store import Attribution, EventRow, SessionRows, Store


@dataclass
class EngineConfig:
    name: str
    scorer: object
    mood: MoodEngine
    note: str = ""


def build_engine(name: str, allow_fallback: bool = True) -> EngineConfig:
    """Create the sentiment scorer + mood engine for an engine preset."""
    mood = get_mood_engine(name)
    if name == "lexicon":
        return EngineConfig(name, LexiconScorer(), mood)

    if transformer_available():
        return EngineConfig(name, TransformerScorer(), mood)

    if name == "hybrid" and allow_fallback:
        return EngineConfig(
            "hybrid",
            LexiconScorer(),
            LexiconMood(),
            note="transformers/torch not installed; fell back to lexicon",
        )
    raise RuntimeError(
        f"Engine {name!r} needs the optional ML dependencies. "
        "Install with: pip install 'ragebaitLM[ml]'"
    )


def _nearest_preceding_assistant(
    messages: list[NormalizedMessage], index: int
) -> NormalizedMessage | None:
    for j in range(index - 1, -1, -1):
        if messages[j].kind == Kind.ASSISTANT:
            return messages[j]
    return None


def score_session(
    session: NormalizedSession, engine: EngineConfig
) -> SessionRows:
    """Score every human message and attach model attribution."""
    messages = session.messages
    human_indices = [i for i, m in enumerate(messages) if m.kind == Kind.HUMAN]

    sentiments = {}
    moods = {}
    if human_indices:
        texts = [messages[i].text for i in human_indices]
        results = engine.scorer.score_batch(texts)  # type: ignore[attr-defined]
        for i, sentiment in zip(human_indices, results):
            sentiments[i] = sentiment
            moods[i] = engine.mood.quantify(messages[i].text, sentiment)

    rows: list[EventRow] = []
    for i, message in enumerate(messages):
        row = EventRow(message=message)
        if i in sentiments:
            row.sentiment = sentiments[i]
            row.mood = moods[i]
            prev = _nearest_preceding_assistant(messages, i)
            row.attribution = Attribution(
                user_seq=message.seq,
                prev_assistant_seq=prev.seq if prev else None,
                prev_model=prev.model if prev else None,
                prev_provider=prev.provider if prev else None,
                prev_stop_reason=prev.stop_reason if prev else None,
                prev_latency_ms=(message.ts - prev.ts)
                if prev and prev.ts and message.ts
                else None,
                current_model_at_turn=message.model,
            )
        rows.append(row)
    return SessionRows(session=session, rows=rows)


def sync(
    store: Store,
    adapter: Adapter,
    engine: EngineConfig,
    paths: Iterable[Path] | None = None,
    since: int | None = None,
    progress=None,
) -> dict:
    """Ingest all sessions for one adapter, returning aggregate counts."""
    run_id = store.start_run(adapter.name, str(adapter.default_path), engine.name)
    counts = {"sessions": 0, "messages": 0, "human": 0, "skipped": 0}
    try:
        for session in adapter.iter_sessions(paths=paths, since=since):
            data = score_session(session, engine)
            stats = store.persist_session(data, run_id)
            counts["sessions"] += 1
            counts["messages"] += stats["messages"]
            counts["human"] += stats["human"]
            if progress:
                progress(session)
        store.finish_run(
            run_id,
            counts["sessions"],
            counts["messages"],
            counts["human"],
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        store.finish_run(
            run_id,
            counts["sessions"],
            counts["messages"],
            counts["human"],
            status="error",
            error=str(exc),
        )
        raise
    return counts


def sync_many(
    store: Store,
    adapters: list[Adapter],
    engine: EngineConfig,
    since: int | None = None,
    progress=None,
) -> dict:
    total = {"sessions": 0, "messages": 0, "human": 0}
    started = time.time()
    for adapter in adapters:
        result = sync(store, adapter, engine, since=since, progress=progress)
        for key in total:
            total[key] += result[key]
    total["elapsed_s"] = round(time.time() - started, 2)
    return total
