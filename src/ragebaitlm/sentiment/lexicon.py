"""Lexicon-based sentiment + mood signals.

Used as the sentiment backend for ``--engine lexicon`` and as the auxiliary
signal source for the default ``hybrid`` mood engine.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .types import SentimentResult

ENGINE_VERSION = "vader+custom-1"

# Words that reliably signal rage in a coding-agent conversation.
PROFANITY = {
    "fuck",
    "fucking",
    "fucked",
    "shit",
    "shitty",
    "damn",
    "crap",
    "bloody",
    "wtf",
    "ffs",
    "smh",
    "ugh",
    "arse",
    "ass",
    "bastard",
    "dumb",
    "stupid",
    "idiotic",
    "garbage",
    "useless",
    "nonsense",
}

# Correction / repetition / urgency markers — "you did the wrong thing again".
CORRECTION_MARKERS = [
    "no",
    "nope",
    "stop",
    "again",
    "still",
    "still not",
    "not working",
    "doesn't work",
    "does not work",
    "didn't work",
    "did not work",
    "i said",
    "i told you",
    "i already said",
    "how many times",
    "why",
    "why is",
    "why does",
    "broken",
    "revert",
    "undo",
    "wrong",
    "incorrect",
    "that's not",
    "thats not",
    "not what i",
    "you missed",
    "you broke",
    "regression",
    "repeatedly",
    "for the last time",
]

# Coding-specific negative terms (mild on their own, meaningful in aggregate).
NEGATIVE_TERMS = [
    "bug",
    "buggy",
    "fail",
    "failed",
    "failing",
    "error",
    "crash",
    "crashed",
    "timeout",
    "timed out",
    "stuck",
    "hang",
    "hung",
    "mess",
    "worse",
    "useless",
    "broken",
]

# Words that signal satisfaction / delight in a coding-agent conversation.
JOY_MARKERS = [
    "thanks",
    "thank you",
    "perfect",
    "great",
    "nice",
    "awesome",
    "excellent",
    "love",
    "brilliant",
    "amazing",
    "appreciate",
    "helpful",
    "good job",
    "well done",
    "beautiful",
    "exactly",
    "works",
    "working",
    "fixed",
    "clean",
]


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _word_boundary(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-zA-Z])" + re.escape(term) + r"(?![a-zA-Z])", re.IGNORECASE)


class LexiconScorer:
    """VADER-backed sentiment scorer (no torch dependency)."""

    name = "lexicon"

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        out: list[SentimentResult] = []
        for text in texts:
            out.append(self.score(text))
        return out

    def score(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(0.0, 1.0, 0.0, "neutral", 0.0, self.name, ENGINE_VERSION)
        scores = _vader().polarity_scores(text)
        pos = float(scores["pos"])
        neu = float(scores["neu"])
        neg = float(scores["neg"])
        compound = float(scores["compound"])
        label = max(
            (("negative", neg), ("neutral", neu), ("positive", pos)),
            key=lambda kv: kv[1],
        )[0]
        return SentimentResult(neg, neu, pos, label, compound, self.name, ENGINE_VERSION)


def extract_mood_signals(text: str) -> tuple[float, dict]:
    """Return a lexicon mood score in [-1, 1] plus the fired signals.

    The score is ``positive_strength - negative_strength``: ``+1`` is joy and
    ``-1`` is rage.
    """
    if not text or not text.strip():
        return 0.0, {"empty": True}

    lowered = text.lower()
    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    letters = [w for w in words if len(w) >= 3]

    profanity_hits = [w for w in words if w.lower() in PROFANITY]
    caps_words = [w for w in letters if w.isupper()]
    caps_ratio = (len(caps_words) / len(letters)) if letters else 0.0

    exclam_runs = re.findall(r"!{2,}", text)
    single_exclaim = len(re.findall(r"(?<!\!)\!(?!\!)", text))
    question_runs = re.findall(r"\?{2,}", text)

    marker_hits: list[str] = []
    for marker in CORRECTION_MARKERS:
        if _word_boundary(marker).search(lowered):
            marker_hits.append(marker)
    negative_hits = [t for t in NEGATIVE_TERMS if _word_boundary(t).search(lowered)]
    joy_hits = [m for m in JOY_MARKERS if _word_boundary(m).search(lowered)]

    vader = _vader().polarity_scores(text)
    pos_component = float(vader["pos"])
    neg_component = max(float(vader["neg"]), max(0.0, -float(vader["compound"])))

    profanity_s = min(1.0, len(profanity_hits) / 2)
    caps_s = min(1.0, caps_ratio / 0.5)
    exclaim_s = min(1.0, (len(exclam_runs) + 0.3 * single_exclaim) / 2)
    marker_s = min(1.0, len(marker_hits) / 3)
    negative_term_s = min(1.0, len(negative_hits) / 3)
    joy_s = min(1.0, len(joy_hits) / 2)

    negative_strength = (
        0.40 * neg_component
        + 0.18 * profanity_s
        + 0.14 * caps_s
        + 0.10 * exclaim_s
        + 0.10 * marker_s
        + 0.08 * negative_term_s
    )
    positive_strength = 0.60 * pos_component + 0.40 * joy_s

    negative_strength = max(0.0, min(1.0, negative_strength))
    positive_strength = max(0.0, min(1.0, positive_strength))
    mood = max(-1.0, min(1.0, positive_strength - negative_strength))

    signals = {
        "positive_strength": round(positive_strength, 4),
        "negative_strength": round(negative_strength, 4),
        "joy_markers": joy_hits,
        "profanity": profanity_hits,
        "caps_ratio": round(caps_ratio, 4),
        "exclam_runs": len(exclam_runs),
        "question_runs": len(question_runs),
        "correction_markers": marker_hits,
        "negative_terms": negative_hits,
        "sub_scores": {
            "positive": round(positive_strength, 4),
            "negative": round(negative_strength, 4),
            "pos_component": round(pos_component, 4),
            "neg_component": round(neg_component, 4),
            "profanity": round(profanity_s, 4),
            "caps": round(caps_s, 4),
            "exclaim": round(exclaim_s, 4),
            "markers": round(marker_s, 4),
            "negative_terms": round(negative_term_s, 4),
            "joy": round(joy_s, 4),
        },
    }
    return mood, signals
