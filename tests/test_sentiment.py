import pytest

from ragebaitlm.filters import is_human_prompt, looks_injected, text_from_content
from ragebaitlm.sentiment.lexicon import LexiconScorer
from ragebaitlm.sentiment.mood import HybridMood, LexiconMood, TransformerMood
from ragebaitlm.sentiment.types import SentimentResult


def test_transformer_mood_negative_dominant_is_rage():
    sent = SentimentResult(0.90, 0.05, 0.05, "negative", -0.85, "transformer")
    result = TransformerMood().quantify("still broken", sent)
    assert result.score == pytest.approx(-0.85, abs=0.02)
    assert result.signals["difference"] < -0.8


def test_transformer_mood_positive_is_joy():
    sent = SentimentResult(0.02, 0.08, 0.90, "positive", 0.88, "transformer")
    assert TransformerMood().quantify("thanks!", sent).score > 0.8


def test_transformer_mood_neutral_is_zero():
    sent = SentimentResult(0.05, 0.90, 0.05, "neutral", 0.0, "transformer")
    assert abs(TransformerMood().quantify("the file is at /tmp", sent).score) < 0.1


def test_lexicon_mood_profanity_is_rage():
    text = "this is fucking broken"
    sent = LexiconScorer().score(text)
    result = LexiconMood().quantify(text, sent)
    assert result.score < 0
    assert "fuck" in result.signals["profanity"][0].lower()


def test_lexicon_mood_praise_is_joy():
    text = "thanks, that works perfectly!"
    sent = LexiconScorer().score(text)
    result = LexiconMood().quantify(text, sent)
    assert result.score > 0


def test_lexicon_mood_calm_is_near_zero():
    text = "could you add a test for the parser"
    sent = LexiconScorer().score(text)
    assert abs(LexiconMood().quantify(text, sent).score) < 0.2


def test_hybrid_blends():
    text = "no no no this is STILL broken!!"
    sent = SentimentResult(0.88, 0.07, 0.05, "negative", -0.83, "transformer")
    hybrid = HybridMood(transformer_weight=0.65, lexicon_weight=0.35)
    result = hybrid.quantify(text, sent)
    assert -1.0 <= result.score <= 1.0
    assert result.score < 0
    assert "transformer_score" in result.signals
    assert "lexicon_score" in result.signals


def test_filters_injected():
    assert is_human_prompt("fix the build")
    assert not is_human_prompt("<system-reminder>be nice</system-reminder>")
    assert not is_human_prompt("   ")
    assert not is_human_prompt("<recommended_plugins> x") 
    assert not looks_injected("just a normal message")
    assert looks_injected("# Response annotations: stuff :codex-annotation{index=\"1\"}")


def test_text_from_content():
    assert text_from_content("hello") == "hello"
    assert text_from_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert text_from_content([{"type": "image", "data": "..."}]) == ""
