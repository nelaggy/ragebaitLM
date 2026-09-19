"""Shared sentiment result type."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SentimentResult:
    """Class probabilities plus a polarity convenience field.

    ``compound`` is ``positive - negative`` in ``[-1, 1]``: ``+1`` is maximally
    positive, ``-1`` maximally negative. It carries the same sign convention as
    the mood score.
    """

    negative: float
    neutral: float
    positive: float
    label: str
    compound: float
    source: str
    engine_version: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def negative_magnitude(self) -> float:
        """Strength of the negative signal, independent of positive mass."""
        return max(self.negative, max(0.0, -self.compound))
