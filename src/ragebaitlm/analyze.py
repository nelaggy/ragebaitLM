"""Aggregate scored messages into mood / blame statistics.

Mood is signed: ``+1`` is joy, ``-1`` is rage. Aggregates therefore use a
double-sided axis and report the share of *rage* (strongly negative) and *joy*
(strongly positive) messages rather than a single one-sided "high" rate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .store import Store

RAGE_THRESHOLD = -1 / 3
JOY_THRESHOLD = 1 / 3
HIST_BINS_PER_SECTION = 7
MOOD_RANGE = (-1.0, 1.0)
MIN_GROUP_MESSAGES = 6


def load_messages(store: Store) -> pd.DataFrame:
    rows = [dict(r) for r in store.scored_messages()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("mood_score", "sentiment_compound"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mood_score"])
    if df.empty:
        return df

    df["model"] = df["prev_model"].fillna(df["current_model_at_turn"]).fillna("unknown")
    df["provider"] = df["prev_provider"].fillna("unknown")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.sort_values(["session_id", "ts", "seq"]).reset_index(drop=True)

    df["is_rage"] = df["mood_score"] <= RAGE_THRESHOLD
    df["is_joy"] = df["mood_score"] >= JOY_THRESHOLD
    return df


def _bootstrap_ci(values, iters: int = 1000, seed: int = 0) -> tuple[float | None, float | None]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return None, None
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(arr, size=(iters, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


def _group_stats(df: pd.DataFrame, column: str) -> list[dict]:
    out: list[dict] = []
    for key, group in df.groupby(column, dropna=False):
        low, high = _bootstrap_ci(group["mood_score"])
        mood = group["mood_score"]
        out.append(
            {
                "key": str(key),
                "n": int(len(group)),
                "mean": float(mood.mean()),
                "median": float(mood.median()),
                "q1": float(mood.quantile(0.25)),
                "q3": float(mood.quantile(0.75)),
                "low": float(mood.min()),
                "high": float(mood.max()),
                "values": [round(float(v), 4) for v in mood],
                "pct_rage": float(group["is_rage"].mean()),
                "pct_joy": float(group["is_joy"].mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    out.sort(key=lambda r: r["mean"])
    return out


def _histogram(df: pd.DataFrame) -> dict:
    edges = np.concatenate(
        [
            np.linspace(MOOD_RANGE[0], RAGE_THRESHOLD, HIST_BINS_PER_SECTION + 1),
            np.linspace(RAGE_THRESHOLD, JOY_THRESHOLD, HIST_BINS_PER_SECTION + 1)[1:],
            np.linspace(JOY_THRESHOLD, MOOD_RANGE[1], HIST_BINS_PER_SECTION + 1)[1:],
        ]
    )
    counts, edges = np.histogram(
        df["mood_score"].to_numpy(dtype=float), bins=edges
    )
    return {
        "bin_edges": [round(float(e), 4) for e in edges],
        "bin_centers": [round(float((edges[i] + edges[i + 1]) / 2), 4) for i in range(len(counts))],
        "counts": [int(c) for c in counts],
    }


def _session_series(df: pd.DataFrame, markers: dict[str, list[dict]]) -> list[dict]:
    sessions = []
    for session_id, group in df.groupby("session_id"):
        head = group.iloc[0]
        points = [
            {
                "ts": int(row["ts"]) if pd.notna(row["ts"]) else None,
                "mood": float(row["mood_score"]),
                "model": str(row["model"]),
                "char_len": int(row["char_len"]) if pd.notna(row["char_len"]) else 0,
            }
            for _, row in group.iterrows()
        ]
        sessions.append(
            {
                "session_id": session_id,
                "title": head["title"] or session_id,
                "harness": head["harness"],
                "project_path": head["project_path"],
                "n_messages": len(group),
                "mean_mood": float(group["mood_score"].mean()),
                "points": points,
                "markers": markers.get(session_id, []),
            }
        )
    sessions.sort(key=lambda s: s["mean_mood"])
    return sessions


def analyze(store: Store, max_sessions: int = 40) -> dict:
    df = load_messages(store)
    markers = store.model_markers()
    sessions_total = store.session_counts()

    base_overall = {
        "n_messages": 0,
        "n_sessions": 0,
        "n_subagent_sessions": sessions_total["subagents"],
        "n_stored_sessions": sessions_total["sessions"],
        "rage_threshold": RAGE_THRESHOLD,
        "joy_threshold": JOY_THRESHOLD,
    }
    if df.empty:
        return {
            "overall": base_overall,
            "by_model": [],
            "by_harness": [],
            "by_provider": [],
            "mood_histogram": {"bin_edges": [], "bin_centers": [], "counts": []},
            "top_sessions": [],
            "sessions": [],
        }

    by_model = [r for r in _group_stats(df, "model") if r["n"] >= MIN_GROUP_MESSAGES]
    by_harness = [r for r in _group_stats(df, "harness") if r["n"] >= MIN_GROUP_MESSAGES]
    by_provider = _group_stats(df, "provider")

    top_sessions = [
        {
            "session_id": sid,
            "title": group.iloc[0]["title"] or sid,
            "harness": group.iloc[0]["harness"],
            "n_messages": int(len(group)),
            "mean_mood": float(group["mood_score"].mean()),
            "min_mood": float(group["mood_score"].min()),
        }
        for sid, group in df.groupby("session_id")
    ]
    top_sessions.sort(key=lambda r: r["mean_mood"])

    overall = {
        "n_messages": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_subagent_sessions": sessions_total["subagents"],
        "n_stored_sessions": sessions_total["sessions"],
        "mean_mood": float(df["mood_score"].mean()),
        "median_mood": float(df["mood_score"].median()),
        "pct_rage": float(df["is_rage"].mean()),
        "pct_joy": float(df["is_joy"].mean()),
        "rage_threshold": RAGE_THRESHOLD,
        "joy_threshold": JOY_THRESHOLD,
    }

    sessions = _session_series(df, markers)
    return {
        "overall": overall,
        "by_model": by_model,
        "by_harness": by_harness,
        "by_provider": by_provider,
        "mood_histogram": _histogram(df),
        "top_sessions": top_sessions[:max_sessions],
        "sessions": sessions[:max_sessions],
    }


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2, default=str)
