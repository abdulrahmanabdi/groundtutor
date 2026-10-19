import numpy as np
import pytest

from groundtutor.bm25 import BM25Index
from groundtutor.embeddings import HashEmbedder, l2_normalize
from groundtutor.errors import ConfigurationError
from groundtutor.retrieve import HybridRetriever, reciprocal_rank_fusion
from groundtutor.stores import MemoryStore, PgVectorStore, resolve_store
from groundtutor.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="doc",
        text=text,
        source="notes.md",
        source_type="markdown",
        char_start=0,
        char_end=len(text),
    )


def test_reciprocal_rank_fusion_sums_inverse_ranks():
    fused = reciprocal_rank_fusion(
        [["b", "a", "c"], ["c", "b"]],
        k=60,
    )
    scores = dict(fused)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0][0] == "b"
    assert [item[0] for item in fused] == sorted(scores, key=lambda key: (-scores[key], key))


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index()
    index.add(
        ["a", "b"],
        [
            "Backpropagation computes gradients by applying the chain rule.",
            "Photosynthesis converts light into chemical energy.",
        ],
    )
    hits = index.query("How does backpropagation compute gradients?", k=2)
    assert [hit[0] for hit in hits] == ["a"]
    assert hits[0][1] > 0


def test_memory_store_cosine_order():
    store = MemoryStore()
    store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    hits = store.query(np.array([0.9, 0.1], dtype=np.float32), k=2)
    assert [hit[0] for hit in hits] == ["a", "b"]
    assert hits[0][1] > hits[1][1]
    ids, matrix = store.dump()
    restored = MemoryStore()
    restored.load(ids, matrix)
    again = restored.query(np.array([0.9, 0.1], dtype=np.float32), k=1)
    assert again[0][0] == "a"


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=64)
    first = embedder.embed_documents(["chain rule gradients", "tomato soup"])
    second = embedder.embed_documents(["chain rule gradients", "tomato soup"])
    assert first.shape == (2, 64)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    query = embedder.embed_query("gradients and the chain rule")
    assert float(first[0] @ query) > float(first[1] @ query)
    assert l2_normalize(np.zeros(4, dtype=np.float32)).shape == (4,)


def test_hybrid_retrieval_surfaces_the_lexical_and_dense_match():
    retriever = HybridRetriever(HashEmbedder(), MemoryStore())
    retriever.add(
        [
            _chunk("a", "Backpropagation computes gradients by applying the chain rule."),
            _chunk("b", "Photosynthesis converts light into chemical energy in leaves."),
        ]
    )
    found = retriever.retrieve("How does backpropagation compute gradients?", k=2)
    assert found.chunks[0].id == "a"
    assert found.term_coverage > 0.45
    assert found.dense_score > 0.2


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        self.connection.calls.append((statement, params))
        if statement.strip().upper().startswith("SELECT"):
            self._rows = [("chunk-1", 0.82)]
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.calls.append(("COMMIT", None))


def test_pgvector_writes_parameterized_sql_and_rejects_bad_table_names():
    connection = _FakeConnection()
    store = PgVectorStore("postgresql://localhost/tutor", dim=2, connection=connection)
    store.add(["chunk-1"], np.array([[1.0, 0.0]], dtype=np.float32))
    hits = store.query(np.array([1.0, 0.0], dtype=np.float32), k=1)
    statements = "\n".join(call[0] for call in connection.calls)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert "vector(2)" in statements
    assert "<=>" in statements
    assert hits == [("chunk-1", 0.82)]
    with pytest.raises(ValueError):
        PgVectorStore("postgresql://localhost/tutor", dim=2, table="chunks;drop")


def test_store_spec_parser():
    assert isinstance(resolve_store("memory"), MemoryStore)
    with pytest.raises(ConfigurationError):
        resolve_store("redis")
    with pytest.raises(ConfigurationError):
        resolve_store("pgvector:")


def test_reranker_reorders_before_the_cutoff():
    class Reverse:
        def rerank(self, query, chunks):
            return [(chunk, float(index)) for index, chunk in enumerate(reversed(list(chunks)))]

    retriever = HybridRetriever(HashEmbedder(), MemoryStore(), reranker=Reverse())
    retriever.add(
        [
            _chunk("a", "Backpropagation computes gradients by applying the chain rule."),
            _chunk("b", "Backpropagation also computes gradients with the chain rule."),
        ]
    )
    found = retriever.retrieve("backpropagation gradients chain rule", k=1, candidate_k=2)
    assert len(found.chunks) == 1
