import numpy as np
import pytest

from groundtutor.errors import ConfigurationError
from groundtutor.retrieve import CrossEncoderReranker
from groundtutor.stores import ChromaStore
from groundtutor.types import Chunk


class _Collection:
    def __init__(self):
        self.ids: list[str] = []
        self.rows: list[list[float]] = []

    def upsert(self, ids, embeddings):
        for chunk_id, row in zip(ids, embeddings, strict=True):
            if chunk_id in self.ids:
                self.rows[self.ids.index(chunk_id)] = row
            else:
                self.ids.append(chunk_id)
                self.rows.append(row)

    def query(self, query_embeddings, n_results):
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        matrix = np.asarray(self.rows, dtype=np.float32)
        scores = matrix @ query
        order = sorted(range(len(self.ids)), key=lambda index: -float(scores[index]))[:n_results]
        return {
            "ids": [[self.ids[index] for index in order]],
            "distances": [[1.0 - float(scores[index]) for index in order]],
        }


def test_chroma_adapter_converts_cosine_distance_to_similarity():
    store = ChromaStore(collection_obj=_Collection())
    store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    hits = store.query(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0] == ("a", pytest.approx(1.0))
    assert hits[1][0] == "b"
    assert hits[1][1] == pytest.approx(0.0)


def test_cross_encoder_uses_the_supplied_model():
    class Model:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    reranker = CrossEncoderReranker(model=Model())
    chunks = [
        Chunk("a", "d", "short", "s", "text", 0, 5),
        Chunk("b", "d", "a much longer piece of text", "s", "text", 0, 28),
    ]
    ranked = reranker.rerank("query", chunks)
    assert [chunk.id for chunk, _ in ranked] == ["b", "a"]


def test_cross_encoder_reports_a_missing_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ConfigurationError, match="local"):
        CrossEncoderReranker()
