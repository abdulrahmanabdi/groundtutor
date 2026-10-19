"""Hybrid retrieval: dense search, BM25, reciprocal rank fusion, optional rerank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from groundtutor.bm25 import BM25Index
from groundtutor.embeddings import Embedder
from groundtutor.stores import VectorStore
from groundtutor.tokenize import term_coverage
from groundtutor.types import Chunk


class Reranker(Protocol):
    """Reorders a candidate list. Higher scores should come first."""

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> Sequence[tuple[Chunk, float]]:
        """Return ``(chunk, score)`` pairs, best first."""


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Chunks selected for a query, with the scores used by the refusal gate."""

    chunks: tuple[Chunk, ...]
    dense_scores: tuple[float, ...]
    term_coverages: tuple[float, ...]

    @property
    def dense_score(self) -> float:
        return max(self.dense_scores, default=0.0)

    @property
    def term_coverage(self) -> float:
        return max(self.term_coverages, default=0.0)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. ``k`` is the RRF constant, not the result size.

    Score(d) = sum over lists of ``1 / (k + rank)`` with ranks starting at 1.
    Ties break by id so the order is stable.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        return ordered[:top_n]
    return ordered


class CrossEncoderReranker:
    """Cross-encoder reranker. Requires the ``local`` extra unless ``model`` is passed."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                from groundtutor.errors import ConfigurationError

                raise ConfigurationError(
                    'The cross-encoder reranker requires: pip install "groundtutor[local]"'
                ) from exc
            model = CrossEncoder(model_name)
        self.model_name = model_name
        self._model = model

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        raw = self._model.predict(pairs)
        scores = [float(score) for score in raw]
        order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index].id))
        return [(chunks[index], scores[index]) for index in order]


class HybridRetriever:
    """Owns the chunk catalog, the BM25 index, and the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self.chunks: dict[str, Chunk] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        ids = [chunk.id for chunk in chunks]
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(chunks):
            raise ValueError("embedder returned an unexpected shape")
        self.store.add(ids, array)
        self.bm25.add(ids, [chunk.text for chunk in chunks])
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def restore(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Load chunks and a precomputed matrix into an empty memory index."""
        if self.chunks:
            raise ValueError("retriever is not empty")
        ids = [chunk.id for chunk in chunks]
        self.store.add(ids, np.asarray(vectors, dtype=np.float32))
        self.bm25.add(ids, [chunk.text for chunk in chunks])
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def retrieve(self, query: str, *, k: int, candidate_k: int = 20) -> Retrieval:
        """Return the fused top ``k`` chunks and the scores used for refusal."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.chunks:
            return Retrieval((), (), ())
        fetch = max(k, candidate_k)
        dense_hits = list(self.store.query(self.embedder.embed_query(query), fetch))
        lexical_hits = self.bm25.query(query, fetch)
        fused = reciprocal_rank_fusion(
            [[chunk_id for chunk_id, _ in dense_hits], [chunk_id for chunk_id, _ in lexical_hits]],
            k=self.rrf_k,
            top_n=fetch,
        )
        dense_map = dict(dense_hits)
        ordered = [self.chunks[chunk_id] for chunk_id, _ in fused if chunk_id in self.chunks]
        if self.reranker is not None and ordered:
            ordered = [chunk for chunk, _ in self.reranker.rerank(query, ordered)]
        chosen = ordered[:k]
        return Retrieval(
            chunks=tuple(chosen),
            dense_scores=tuple(float(dense_map.get(chunk.id, 0.0)) for chunk in chosen),
            term_coverages=tuple(term_coverage(query, chunk.text) for chunk in chosen),
        )
