"""Public result types.

Locators are 1-based page and slide numbers, or timestamps in seconds.
Character offsets are into the normalized text of that page, slide, transcript,
or notes file — the same string stored on the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


def format_timestamp(seconds: float) -> str:
    """Format a timestamp as ``MM:SS`` or ``H:MM:SS``."""
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable piece of a document."""

    id: str
    doc_id: str
    text: str
    source: str
    source_type: str
    char_start: int
    char_end: int
    page: int | None = None
    slide: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    heading_path: tuple[str, ...] = ()

    def locator(self) -> str:
        """Human-readable location, such as ``p. 3 · Backprop`` or ``01:04–01:20``."""
        if self.page is not None:
            label = f"p. {self.page}"
        elif self.slide is not None:
            label = f"slide {self.slide}"
        elif self.start_time is not None and self.end_time is not None:
            label = f"{format_timestamp(self.start_time)}–{format_timestamp(self.end_time)}"
        else:
            label = "source"
        if self.heading_path:
            label = f"{label} · {' › '.join(self.heading_path)}"
        return label


@dataclass(frozen=True, slots=True)
class Citation:
    """A numbered source a claim or study item points at."""

    id: int
    chunk_id: str
    doc_id: str
    source: str
    source_type: str
    locator: str
    char_start: int
    char_end: int
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    slide: int | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True, slots=True)
class QuoteSpan:
    """An exact span of a cited chunk that supports a claim."""

    citation_id: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic statement extracted from an answer."""

    text: str
    citation_ids: tuple[int, ...]
    quotes: tuple[QuoteSpan, ...]
    grounding_score: float
    supported: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SectionHint:
    """A nearby section returned when an answer is refused."""

    source: str
    source_type: str
    locator: str
    snippet: str
    score: float
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    slide: int | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer, or a refusal."""

    answer: str
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    ungrounded: tuple[Claim, ...]
    refused: bool = False
    suggestions: tuple[SectionHint, ...] = ()
    dense_score: float = 0.0
    term_coverage: float = 0.0

    @property
    def confidence(self) -> float:
        """The stronger of the dense score and the query-term coverage."""
        return max(self.dense_score, self.term_coverage)

    @property
    def fully_grounded(self) -> bool:
        """True when the answer was not refused and every claim is supported."""
        return not self.refused and bool(self.claims) and not self.ungrounded


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    """A multiple-choice question whose answer is quoted from a source."""

    question: str
    choices: tuple[str, ...]
    answer: str
    explanation: str
    quote: str
    citations: tuple[Citation, ...]
    grounding_score: float


@dataclass(frozen=True, slots=True)
class Quiz:
    """Questions that survived the grounding check."""

    questions: tuple[QuizQuestion, ...]

    def __iter__(self):
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> QuizQuestion:
        return self.questions[index]


@dataclass(frozen=True, slots=True)
class Flashcard:
    """A card whose back is quoted from a source."""

    front: str
    back: str
    quote: str
    citations: tuple[Citation, ...]
    grounding_score: float


@dataclass(frozen=True, slots=True)
class FlashcardSet:
    """Flashcards that survived the grounding check."""

    cards: tuple[Flashcard, ...]

    def __iter__(self):
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, index: int) -> Flashcard:
        return self.cards[index]


def citation_for(number: int, chunk: Chunk) -> Citation:
    """Build the public citation for a chunk numbered in a prompt."""
    return Citation(
        id=number,
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        source=chunk.source,
        source_type=chunk.source_type,
        locator=chunk.locator(),
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        heading_path=chunk.heading_path,
        page=chunk.page,
        slide=chunk.slide,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
    )
