"""Vector stores. The memory store is the default; Chroma and pgvector are optional."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from groundtutor.embeddings import l2_normalize
from groundtutor.errors import ConfigurationError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VectorStore(Protocol):
    """Stores embedding rows and returns cosine similarity, highest first."""

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert or replace rows. ``vectors`` has shape ``(len(ids), dim)``."""

    def query(self, vector: np.ndarray, k: int) -> Sequence[tuple[str, float]]:
        """Return up to ``k`` ``(id, cosine)`` pairs."""


class MemoryStore:
    """Brute-force cosine search over an in-memory float32 matrix."""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        if not ids:
            return
        rows = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if rows.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if self._matrix is None:
            self._ids = list(ids)
            self._matrix = rows
            return
        if rows.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: store has {self._matrix.shape[1]}, got {rows.shape[1]}"
            )
        position = {chunk_id: index for index, chunk_id in enumerate(self._ids)}
        fresh_ids: list[str] = []
        fresh_rows: list[np.ndarray] = []
        for chunk_id, row in zip(ids, rows, strict=True):
            if chunk_id in position:
                self._matrix[position[chunk_id]] = row
            else:
                fresh_ids.append(chunk_id)
                fresh_rows.append(row)
        if fresh_rows:
            self._ids.extend(fresh_ids)
            self._matrix = np.vstack([self._matrix, np.stack(fresh_rows)])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None or k <= 0:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: store has {self._matrix.shape[1]}, got {query.shape[0]}"
            )
        scores = self._matrix @ query
        limit = min(k, len(self._ids))
        if limit < len(self._ids):
            chosen = np.argpartition(-scores, limit - 1)[:limit]
        else:
            chosen = np.arange(len(self._ids))
        order = sorted(chosen, key=lambda index: (-float(scores[index]), self._ids[index]))
        return [(self._ids[int(index)], float(scores[index])) for index in order]

    def dump(self) -> tuple[list[str], np.ndarray]:
        """Copy the current ids and unit-length matrix."""
        if self._matrix is None:
            return [], np.zeros((0, 0), dtype=np.float32)
        return list(self._ids), self._matrix.copy()

    def load(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Replace the store with a previously dumped matrix."""
        if self._matrix is not None:
            raise ValueError("store is not empty")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        if not ids:
            return
        self._ids = list(ids)
        self._matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))


class ChromaStore:
    """Chroma collection using cosine distance.

    Pass ``collection_obj`` to reuse a collection, or ``path`` to persist one.
    With neither, the collection is in-process only.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        name: str = "groundtutor",
        client=None,
        collection_obj=None,
    ):
        if collection_obj is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ConfigurationError(
                    'Chroma requires the optional dependency: pip install "groundtutor[chroma]"'
                ) from exc
            if client is None:
                client = (
                    chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
                )
            collection_obj = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        self._collection = collection_obj

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        rows = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._collection.upsert(ids=list(ids), embeddings=rows.tolist())

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        found = self._collection.query(query_embeddings=[query.tolist()], n_results=k)
        ids = (found.get("ids") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        scored = []
        for chunk_id, distance in zip(ids, distances, strict=True):
            similarity = 1.0 - float(distance)
            scored.append((chunk_id, similarity))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored


class PgVectorStore:
    """Postgres store using the pgvector cosine distance operator.

    The table name is restricted to a plain identifier. ``dim`` is the embedding width
    and is fixed when the table is created.
    """

    def __init__(
        self,
        dsn: str,
        *,
        dim: int,
        table: str = "groundtutor_chunks",
        connection=None,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not _IDENT.fullmatch(table):
            raise ValueError(f"Invalid table name: {table}")
        self.dsn = dsn
        self.dim = dim
        self.table = table
        self._conn = connection
        self._ready = False

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        rows = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if rows.shape[1] != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: store has {self.dim}, got {rows.shape[1]}"
            )
        self._ensure()
        statement = (
            f"INSERT INTO {self.table} (id, embedding) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
        )
        connection = self._connect()
        with connection.cursor() as cursor:
            for chunk_id, row in zip(ids, rows, strict=True):
                cursor.execute(statement, (chunk_id, _vector_literal(row)))
        connection.commit()

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if query.shape[0] != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: store has {self.dim}, got {query.shape[0]}"
            )
        self._ensure()
        literal = _vector_literal(query)
        statement = (
            f"SELECT id, 1 - (embedding <=> %s::vector) AS score FROM {self.table} "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        connection = self._connect()
        with connection.cursor() as cursor:
            cursor.execute(statement, (literal, literal, int(k)))
            found = cursor.fetchall()
        return [(str(row[0]), float(row[1])) for row in found]

    def _ensure(self) -> None:
        if self._ready:
            return
        connection = self._connect()
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                f"id TEXT PRIMARY KEY, embedding vector({self.dim}))"
            )
        connection.commit()
        self._ready = True

    def _connect(self):
        if self._conn is None:
            try:
                import psycopg
            except ImportError as exc:
                raise ConfigurationError(
                    'pgvector requires the optional dependency: pip install "groundtutor[pgvector]"'
                ) from exc
            self._conn = psycopg.connect(self.dsn)
        return self._conn


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector.tolist()) + "]"


def resolve_store(spec: str, *, dim: int | None = None) -> VectorStore:
    """Build ``memory``, ``chroma``, ``chroma:<path>``, or ``pgvector:<dsn>``."""
    if spec == "memory":
        return MemoryStore()
    if spec == "chroma" or spec.startswith("chroma:"):
        path = spec.partition(":")[2] or None
        return ChromaStore(path=path)
    if spec.startswith("pgvector:"):
        dsn = spec[len("pgvector:") :]
        if not dsn:
            raise ConfigurationError("pgvector spec needs a connection string after 'pgvector:'.")
        if dim is None:
            raise ConfigurationError(
                "pgvector needs an embedding dimension. Use an embedder with a known dim, "
                "or pass a PgVectorStore(dsn, dim=...)."
            )
        return PgVectorStore(dsn, dim=dim)
    raise ConfigurationError(
        f"Unknown store '{spec}'. Expected 'memory', 'chroma', 'chroma:<path>', or 'pgvector:<dsn>'."
    )
