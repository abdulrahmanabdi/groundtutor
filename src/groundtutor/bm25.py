"""Okapi BM25 over the in-memory chunk catalog."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from groundtutor.tokenize import tokenize


class BM25Index:
    """Append-only BM25 index. Scores follow Robertson-Sparck Jones IDF."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")
        for chunk_id, text in zip(ids, texts, strict=True):
            tokens = tokenize(text, drop_stopwords=True)
            counts = Counter(tokens)
            self.ids.append(chunk_id)
            self._tf.append(counts)
            self._doc_len.append(len(tokens))
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, score)`` pairs with a positive score, best first."""
        if k <= 0 or not self.ids:
            return []
        terms = tokenize(text, drop_stopwords=True)
        if not terms:
            return []
        total = len(self.ids)
        average = sum(self._doc_len) / total if total else 0.0
        if average == 0.0:
            return []
        scored: list[tuple[str, float]] = []
        for index, counts in enumerate(self._tf):
            score = 0.0
            length = self._doc_len[index]
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._df.get(term, 0)
                idf = math.log(
                    1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / average)
                score += idf * (frequency * (self.k1 + 1.0)) / denominator
            if score > 0.0:
                scored.append((self.ids[index], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]
