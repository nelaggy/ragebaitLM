"""Mood quantification strategies.

Three engines share the same input (a :class:`SentimentResult`, plus raw text
where the lexicon is used) and produce a single signed ``[-1, 1]`` mood score:
``+1`` is joy, ``-1`` is rage, ``0`` is neutral. The score is a difference
between positive and negative evidence rather than a one-sided intensity.

``transformer`` uses *only* the class probabilities from the sentiment model::

    mood = clamp(positive - negative, -1, 1)

``lexicon`` contrasts positive evidence (VADER positivity + praise markers)
with negative evidence (VADER negativity + surface rage cues such as profanity,
ALL CAPS, ``!`` runs and correction/urgency markers).

``hybrid`` is a weighted blend of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .lexicon import extract_mood_signals
from .types import SentimentResult


@dataclass
class MoodResult:
    """A signed mood score in ``[-1, 1]`` (joy positive, rage negative)."""

    score: float
    engine: str
    signals: dict[str, Any] = field(default_factory=dict)


class MoodEngine(Protocol):
    name: str

    def quantify(self, text: str, sentiment: SentimentResult) -> MoodResult: ...


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class TransformerMood:
    """Transformer-only mood: the difference of the probability masses."""

    name = "transformer"

    def quantify(self, text: str, sentiment: SentimentResult) -> MoodResult:
        pos = sentiment.positive
        neg = sentiment.negative
        mood = _clamp(pos - neg)
        return MoodResult(
            score=mood,
            engine=self.name,
            signals={
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral": round(sentiment.neutral, 4),
                "difference": round(mood, 4),
            },
        )


class LexiconMood:
    """Lexicon-only mood from surface cues and VADER polarity."""

    name = "lexicon"

    def quantify(self, text: str, sentiment: SentimentResult) -> MoodResult:
        mood, signals = extract_mood_signals(text)
        return MoodResult(score=mood, engine=self.name, signals=signals)


class HybridMood:
    """Weighted blend of the transformer and lexicon mood scores."""

    name = "hybrid"

    def __init__(self, transformer_weight: float = 0.65, lexicon_weight: float = 0.35):
        total = transformer_weight + lexicon_weight
        self.transformer_weight = transformer_weight / total
        self.lexicon_weight = lexicon_weight / total
        self._transformer = TransformerMood()
        self._lexicon = LexiconMood()

    def quantify(self, text: str, sentiment: SentimentResult) -> MoodResult:
        tf = self._transformer.quantify(text, sentiment)
        lx = self._lexicon.quantify(text, sentiment)
        mood = _clamp(self.transformer_weight * tf.score + self.lexicon_weight * lx.score)
        return MoodResult(
            score=mood,
            engine=self.name,
            signals={
                "transformer_score": round(tf.score, 4),
                "lexicon_score": round(lx.score, 4),
                "transformer": tf.signals,
                "lexicon": lx.signals,
            },
        )


_ENGINES = {
    "transformer": TransformerMood,
    "lexicon": LexiconMood,
    "hybrid": HybridMood,
}


def get_mood_engine(name: str) -> MoodEngine:
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown mood engine {name!r}. Choose from {sorted(_ENGINES)}"
        ) from None
    return factory()
