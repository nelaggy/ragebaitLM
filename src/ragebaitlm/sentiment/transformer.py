"""Transformer sentiment scorer (cardiffnlp twitter-roberta sentiment)."""

from __future__ import annotations

from .types import SentimentResult

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def transformer_available() -> bool:
    """True if torch + transformers can be imported."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


class TransformerScorer:
    """3-class sentiment probabilities from a HuggingFace sequence classifier.

    The model is loaded lazily so that lexicon-only runs never import torch.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = "transformer"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

        raw = self.model.config.id2label
        self.labels = [str(raw[i]).lower() for i in sorted(raw)]

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            results.extend(self._score_chunk(chunk))
        return results

    def score(self, text: str) -> SentimentResult:
        return self.score_batch([text])[0]

    def _score_chunk(self, texts: list[str]) -> list[SentimentResult]:
        torch = self._torch
        clean = [t if t and t.strip() else "neutral" for t in texts]
        encoded = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        out: list[SentimentResult] = []
        for row, original in zip(probs, texts):
            by_label = {self.labels[i]: float(row[i]) for i in range(len(self.labels))}
            neg = by_label.get("negative", 0.0)
            neu = by_label.get("neutral", 0.0)
            pos = by_label.get("positive", 0.0)
            label = max(by_label.items(), key=lambda kv: kv[1])[0]
            if not original or not original.strip():
                neg, neu, pos, label = 0.0, 1.0, 0.0, "neutral"
            out.append(
                SentimentResult(
                    negative=neg,
                    neutral=neu,
                    positive=pos,
                    label=label,
                    compound=pos - neg,
                    source="transformer",
                    engine_version=self.model_name,
                )
            )
        return out
