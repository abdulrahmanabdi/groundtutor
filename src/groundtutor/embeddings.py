"""Pluggable embedders.

``hash`` is a deterministic bag-of-words embedder for tests. ``local`` loads a
sentence-transformers model. ``openai`` calls the embeddings API.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from groundtutor.errors import ConfigurationError, ProviderError
from groundtutor.tokenize import tokenize

_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_DEFAULT_LOCAL = "all-MiniLM-L6-v2"
_DEFAULT_OPENAI = "text-embedding-3-small"


class Embedder(Protocol):
    """Turns text into L2-friendly vectors. Implementations should L2-normalize."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of document chunks. Shape is ``(len(texts), dim)``."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Shape is ``(dim,)``."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy with each row scaled to unit length."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            return array
        return array / norm
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


class HashEmbedder:
    """Hashing-trick bag of words.

    Shared words rank near each other. Paraphrases with no shared words do not.
    Use this for tests and offline smoke checks, not for a real tutor.
    """

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in tokenize(text, drop_stopwords=True):
                bucket = _bucket(token, self.dim)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            for bucket, count in counts.items():
                vectors[row, bucket] = np.log1p(count)
        return l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        embedded = self.embed_documents([text])
        if embedded.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return embedded[0]


class SentenceTransformerEmbedder:
    """Local embedder backed by sentence-transformers. Requires the ``local`` extra."""

    def __init__(self, model_name: str = _DEFAULT_LOCAL, model=None):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ConfigurationError(
                    'Local embeddings require the optional dependency: pip install "groundtutor[local]"'
                ) from exc
            model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._model = model
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """OpenAI embeddings over HTTP. Reads ``OPENAI_API_KEY`` when ``api_key`` is omitted."""

    def __init__(
        self,
        model: str = _DEFAULT_OPENAI,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client=None,
        batch_size: int = 64,
        timeout: float = 60.0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.dim = _OPENAI_DIMS.get(model)
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            width = self.dim or 0
            return np.zeros((0, width), dtype=np.float32)
        rows: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            data = self._post(batch)
            ordered = sorted(data, key=lambda item: item["index"])
            rows.extend(item["embedding"] for item in ordered)
        array = np.asarray(rows, dtype=np.float32)
        if self.dim is None and array.size:
            self.dim = int(array.shape[1])
        return l2_normalize(array)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _post(self, texts: list[str]) -> list[dict]:
        import httpx

        try:
            response = self._http().post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._key()}",
                    "User-Agent": "groundtutor/0.1.0",
                },
                json={"model": self.model, "input": texts},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI embeddings request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI embeddings returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("OpenAI embeddings returned an unexpected payload") from exc

    def _key(self) -> str:
        key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError("Set OPENAI_API_KEY or pass api_key.")
        return key

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def resolve_embedder(spec: str) -> Embedder:
    """Build an embedder from ``hash``, ``local``, ``local:<model>``, ``openai``, or ``openai:<model>``."""
    name, _, model = spec.partition(":")
    key = name.strip().lower()
    model = model.strip()
    if key == "hash":
        return HashEmbedder()
    if key == "local":
        return SentenceTransformerEmbedder(model or _DEFAULT_LOCAL)
    if key == "openai":
        return OpenAIEmbedder(model or _DEFAULT_OPENAI)
    raise ConfigurationError(
        f"Unknown embedder '{spec}'. Expected 'hash', 'local', 'local:<model>', 'openai', or 'openai:<model>'."
    )
