"""Citation-grounded tutoring over your own documents."""

from groundtutor.errors import (
    ConfigurationError,
    GroundingError,
    GroundtutorError,
    IngestError,
    ProviderError,
)
from groundtutor.tutor import Tutor
from groundtutor.types import (
    Answer,
    Chunk,
    Citation,
    Claim,
    Flashcard,
    FlashcardSet,
    Quiz,
    QuizQuestion,
    QuoteSpan,
    SectionHint,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Claim",
    "ConfigurationError",
    "Flashcard",
    "FlashcardSet",
    "GroundingError",
    "GroundtutorError",
    "IngestError",
    "ProviderError",
    "Quiz",
    "QuizQuestion",
    "QuoteSpan",
    "SectionHint",
    "Tutor",
    "__version__",
]
