"""Sentiment scoring and mood quantification.

The sentiment layer is split from the mood layer:

* A :class:`SentimentScorer` turns text into class probabilities.
* A :class:`MoodEngine` turns those probabilities (optionally plus the raw
  text) into a single signed ``[-1, 1]`` mood score: ``+1`` joy, ``-1`` rage.

Keeping them separate is what allows a transformer-only mood measure to be
selected independently of the sentiment backend.
"""

from .lexicon import LexiconScorer
from .mood import (
    HybridMood,
    LexiconMood,
    MoodEngine,
    MoodResult,
    TransformerMood,
    get_mood_engine,
)
from .transformer import DEFAULT_MODEL, TransformerScorer, transformer_available
from .types import SentimentResult

__all__ = [
    "DEFAULT_MODEL",
    "HybridMood",
    "LexiconMood",
    "LexiconScorer",
    "MoodEngine",
    "MoodResult",
    "SentimentResult",
    "TransformerMood",
    "TransformerScorer",
    "get_mood_engine",
    "transformer_available",
]
