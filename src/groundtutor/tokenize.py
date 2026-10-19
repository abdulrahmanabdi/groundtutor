"""Shared tokenization for BM25, hashing embeddings, and lexical checks."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "what",
        "how",
        "does",
        "do",
        "did",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "be",
        "as",
        "at",
        "from",
        "into",
        "about",
        "your",
        "you",
        "can",
        "could",
        "should",
        "would",
        "will",
        "just",
        "than",
        "then",
        "not",
        "but",
        "if",
        "so",
        "we",
        "they",
        "their",
        "our",
        "i",
        "me",
        "my",
    }
)


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercase alphanumeric tokens, optionally dropping stopwords and short tokens."""
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return tokens


def term_coverage(query: str, text: str) -> float:
    """Fraction of content words in ``query`` that also appear in ``text``."""
    terms = tokenize(query, drop_stopwords=True)
    if not terms:
        return 0.0
    haystack = set(tokenize(text))
    return sum(1 for term in terms if term in haystack) / len(terms)
