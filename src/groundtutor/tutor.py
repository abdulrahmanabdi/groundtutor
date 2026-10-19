"""The tutor facade: add documents, ask, and build study items."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from groundtutor.embeddings import Embedder, resolve_embedder
from groundtutor.errors import ConfigurationError
from groundtutor.grounding import (
    MODES,
    LexicalVerifier,
    LLMJudgeVerifier,
    Verifier,
    answer_question,
)
from groundtutor.ingest import (
    LoadedDocument,
    TextUnit,
    chunks_from_document,
    load_path,
    load_transcript,
    make_doc_id,
)
from groundtutor.llm import LLM, resolve_llm
from groundtutor.retrieve import HybridRetriever, Reranker
from groundtutor.stores import MemoryStore, VectorStore, resolve_store
from groundtutor.study import generate_flashcards, generate_quiz
from groundtutor.types import Answer, Chunk, FlashcardSet, Quiz

_INDEX_VERSION = 1


class Tutor:
    """Citation-grounded tutor over a private document index.

    ``llm`` is ``openai:<model>``, ``anthropic:<model>``, ``ollama:<model>``, or
    any object with ``complete(messages, *, temperature, json_mode)``.
    ``embedder`` is ``local``, ``local:<model>``, ``openai``, ``openai:<model>``,
    ``hash``, or an object with ``embed_documents`` and ``embed_query``.
    ``store`` is ``memory``, ``chroma``, ``chroma:<path>``, ``pgvector:<dsn>``,
    or an object with ``add`` and ``query``.

    The refusal gate accepts a question when either the best dense cosine is at
    least ``min_dense_score`` or the best query-term coverage is at least
    ``min_term_coverage``. Otherwise ``ask`` returns "I couldn't find this in
    your materials" and does not call the model.

    ``mode`` defaults to ``flag``. ``strict`` refuses the whole answer when any
    claim is unsupported. ``permissive`` checks that quotes are real and skips
    the entailment judge.
    """

    def __init__(
        self,
        llm: str | LLM = "openai:gpt-4o-mini",
        embedder: str | Embedder = "local",
        store: str | VectorStore = "memory",
        *,
        reranker: Reranker | None = None,
        verifier: str | Verifier | None = "llm",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        retrieval_k: int = 6,
        candidate_k: int = 20,
        min_dense_score: float = 0.28,
        min_term_coverage: float = 0.45,
        mode: str = "flag",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if retrieval_k <= 0 or candidate_k <= 0:
            raise ValueError("retrieval_k and candidate_k must be positive")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(sorted(MODES))}")
        self.llm = resolve_llm(llm) if isinstance(llm, str) else llm
        self.embedder = resolve_embedder(embedder) if isinstance(embedder, str) else embedder
        self.store = (
            resolve_store(store, dim=getattr(self.embedder, "dim", None))
            if isinstance(store, str)
            else store
        )
        self.retriever = HybridRetriever(self.embedder, self.store, reranker=reranker)
        self.verifier = _resolve_verifier(verifier, self.llm)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.candidate_k = candidate_k
        self.min_dense_score = min_dense_score
        self.min_term_coverage = min_term_coverage
        self.mode = mode
        self._docs: dict[str, tuple[str, ...]] = {}

    def add(self, path: str | Path, *, doc_id: str | None = None) -> str:
        """Index a PDF, PowerPoint, text, Markdown, WebVTT, or SRT file."""
        return self._index(load_path(Path(path)), doc_id)

    def add_text(
        self,
        text: str,
        *,
        source: str = "inline.txt",
        source_type: str | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Index a string. Markdown sources keep their heading path."""
        kind = source_type
        if kind is None:
            kind = "markdown" if str(source).endswith((".md", ".markdown")) else "text"
        if kind not in {"text", "markdown"}:
            raise ValueError("source_type must be 'text' or 'markdown'")
        loaded = LoadedDocument(source_type=kind, source=source, units=[TextUnit(text=text)])
        return self._index(loaded, doc_id)

    def add_transcript(
        self,
        source: str | Path,
        *,
        doc_id: str | None = None,
        youtube_fetcher=None,
    ) -> str:
        """Index a transcript file, a YouTube URL, or an 11-character video id."""
        loaded = load_transcript(source, youtube_fetcher=youtube_fetcher)
        return self._index(loaded, doc_id)

    def ask(self, question: str, *, mode: str | None = None, k: int | None = None) -> Answer:
        """Answer ``question`` from the index, or refuse when it is not supported."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        selected = mode or self.mode
        if selected not in MODES:
            raise ValueError(f"mode must be one of {', '.join(sorted(MODES))}")
        limit = self.retrieval_k if k is None else k
        if limit <= 0:
            raise ValueError("k must be positive")
        retrieval = self.retriever.retrieve(question, k=limit, candidate_k=self.candidate_k)
        return answer_question(
            question,
            retrieval,
            llm=self.llm,
            verifier=self.verifier,
            mode=selected,
            min_dense_score=self.min_dense_score,
            min_term_coverage=self.min_term_coverage,
        )

    def quiz(self, n: int = 5, *, topic: str | None = None) -> Quiz:
        """Build up to ``n`` multiple-choice questions, each citing a source."""
        return generate_quiz(self.llm, self.verifier, self._study_chunks(topic, n), n)

    def flashcards(self, n: int = 10, *, topic: str | None = None) -> FlashcardSet:
        """Build up to ``n`` flashcards, each citing a source."""
        return generate_flashcards(self.llm, self.verifier, self._study_chunks(topic, n), n)

    def save(self, directory: str | Path) -> None:
        """Write the memory index to ``directory``. Other stores raise ``ConfigurationError``."""
        if not isinstance(self.store, MemoryStore):
            raise ConfigurationError("save() supports the in-memory store")
        ids, matrix = self.store.dump()
        missing = [chunk_id for chunk_id in ids if chunk_id not in self.retriever.chunks]
        if missing:
            raise ConfigurationError("The vector store contains ids that are not in the catalog")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = [asdict(self.retriever.chunks[chunk_id]) for chunk_id in ids]
        (directory / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
        np.save(directory / "vectors.npy", matrix)
        meta = {"version": _INDEX_VERSION, "dim": int(matrix.shape[1]) if matrix.size else 0}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        llm: str | LLM = "openai:gpt-4o-mini",
        embedder: str | Embedder = "local",
        **kwargs,
    ) -> Tutor:
        """Load an index written by ``save``. ``embedder`` must match the saved dimension."""
        directory = Path(directory)
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
            matrix = np.load(directory / "vectors.npy")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not read index at {directory}: {exc}") from exc
        if meta.get("version") != _INDEX_VERSION:
            raise ConfigurationError(f"Unsupported index version: {meta.get('version')}")
        if "store" in kwargs:
            raise ConfigurationError("load() always restores the in-memory store")
        tutor = cls(llm=llm, embedder=embedder, store="memory", **kwargs)
        saved_dim = int(meta.get("dim") or 0)
        embedder_dim = getattr(tutor.embedder, "dim", None)
        if saved_dim and embedder_dim and embedder_dim != saved_dim:
            raise ConfigurationError(
                f"Embedder dimension is {embedder_dim}, but the index was built with {saved_dim}"
            )
        chunks = tuple(_chunk_from_dict(item) for item in payload)
        if len(chunks) != len(matrix):
            raise ConfigurationError("chunks.json and vectors.npy have different lengths")
        tutor.retriever.restore(chunks, matrix)
        for chunk in chunks:
            tutor._docs.setdefault(chunk.doc_id, ())
            tutor._docs[chunk.doc_id] = (*tutor._docs[chunk.doc_id], chunk.id)
        return tutor

    @property
    def documents(self) -> tuple[str, ...]:
        """Document ids in the order they were added."""
        return tuple(self._docs)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Every chunk in the catalog, in index order."""
        return tuple(self.retriever.chunks.values())

    def _index(self, loaded: LoadedDocument, doc_id: str | None) -> str:
        if loaded.cues:
            content = "\n".join(f"{cue.start:.3f}\t{cue.text}" for cue in loaded.cues)
        else:
            content = "\n".join(unit.text for unit in loaded.units)
        assigned = make_doc_id(loaded.source, doc_id, content=content)
        if assigned in self._docs:
            raise ValueError(f"Document {assigned} is already indexed")
        chunks = chunks_from_document(
            loaded,
            doc_id=assigned,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        self.retriever.add(chunks)
        self._docs[assigned] = tuple(chunk.id for chunk in chunks)
        return assigned

    def _study_chunks(self, topic: str | None, n: int) -> list[Chunk]:
        catalog = list(self.retriever.chunks.values())
        if not catalog:
            return []
        if topic:
            retrieval = self.retriever.retrieve(
                topic,
                k=min(12, max(n, 6)),
                candidate_k=self.candidate_k,
            )
            if (
                retrieval.dense_score < self.min_dense_score
                and retrieval.term_coverage < self.min_term_coverage
            ):
                return []
            return list(retrieval.chunks)
        return _spread(catalog, min(12, max(n, 4)))


def _resolve_verifier(verifier: str | Verifier | None, llm: LLM) -> Verifier:
    if verifier is None or verifier == "llm":
        return LLMJudgeVerifier(llm)
    if verifier == "lexical":
        return LexicalVerifier()
    if hasattr(verifier, "verify"):
        return verifier
    raise ConfigurationError("verifier must be 'llm', 'lexical', or an object with verify()")


def _spread(chunks: list[Chunk], n: int) -> list[Chunk]:
    if len(chunks) <= n:
        return list(chunks)
    if n == 1:
        return [chunks[0]]
    last = len(chunks) - 1
    picked: list[Chunk] = []
    seen: set[int] = set()
    for step in range(n):
        index = round(step * last / (n - 1))
        if index not in seen:
            seen.add(index)
            picked.append(chunks[index])
    return picked


def _chunk_from_dict(data: dict) -> Chunk:
    payload = dict(data)
    payload["heading_path"] = tuple(payload.get("heading_path") or ())
    return Chunk(**payload)
