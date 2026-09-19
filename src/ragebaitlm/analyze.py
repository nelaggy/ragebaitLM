"""Aggregate scored turns into mood / blame statistics.

Mood is signed: ``+1`` is joy, ``-1`` is rage. Aggregates therefore use a
double-sided axis and report the share of *rage* (strongly negative) and *joy*
(strongly positive) turns rather than a single one-sided "high" rate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .store import Store

RAGE_THRESHOLD = -0.5
JOY_THRESHOLD = 0.5
HIST_BINS = 20
MOOD_RANGE = (-1.0, 1.0)


def load_turns(store: Store) -> pd.DataFrame:
    rows = [dict(r) for r in store.scored_turns()]
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

    df["prev_turn_mood"] = df.groupby("session_id")["mood_score"].shift(1)
    df["delta"] = df["mood_score"] - df["prev_turn_mood"]
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
        dlow, dhigh = _bootstrap_ci(group["delta"])
        out.append(
            {
                "key": str(key),
                "n": int(len(group)),
                "mean": float(group["mood_score"].mean()),
                "median": float(group["mood_score"].median()),
                "pct_rage": float(group["is_rage"].mean()),
                "pct_joy": float(group["is_joy"].mean()),
                "mean_delta": None
                if group["delta"].dropna().empty
                else float(group["delta"].mean()),
                "delta_ci_low": dlow,
                "delta_ci_high": dhigh,
                "ci_low": low,
                "ci_high": high,
            }
        )
    out.sort(key=lambda r: r["mean"])
    return out


def _histogram(df: pd.DataFrame) -> dict:
    counts, edges = np.histogram(
        df["mood_score"].to_numpy(dtype=float), bins=HIST_BINS, range=MOOD_RANGE
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
                "n_turns": len(group),
                "mean_mood": float(group["mood_score"].mean()),
                "points": points,
                "markers": markers.get(session_id, []),
            }
        )
    sessions.sort(key=lambda s: s["mean_mood"])
    return sessions


def analyze(store: Store, max_sessions: int = 40) -> dict:
    df = load_turns(store)
    markers = store.model_markers()
    revisions = store.revision_counts()
    sessions_total = store.session_counts()

    base_overall = {
        "n_turns": 0,
        "n_sessions": 0,
        "n_subagent_sessions": sessions_total["subagents"],
        "n_stored_sessions": sessions_total["sessions"],
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
            "revisions": revisions,
        }

    by_model = _group_stats(df, "model")
    by_harness = _group_stats(df, "harness")
    by_provider = _group_stats(df, "provider")

    top_sessions = [
        {
            "session_id": sid,
            "title": group.iloc[0]["title"] or sid,
            "harness": group.iloc[0]["harness"],
            "n_turns": int(len(group)),
            "mean_mood": float(group["mood_score"].mean()),
            "min_mood": float(group["mood_score"].min()),
        }
        for sid, group in df.groupby("session_id")
    ]
    top_sessions.sort(key=lambda r: r["mean_mood"])

    overall = {
        "n_turns": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_subagent_sessions": sessions_total["subagents"],
        "n_stored_sessions": sessions_total["sessions"],
        "mean_mood": float(df["mood_score"].mean()),
        "median_mood": float(df["mood_score"].median()),
        "pct_rage": float(df["is_rage"].mean()),
        "pct_joy": float(df["is_joy"].mean()),
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
        "revisions": revisions,
    }


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2, default=str)
