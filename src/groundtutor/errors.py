"""Errors raised by groundtutor."""


class GroundtutorError(Exception):
    """Base class for errors raised by this library."""


class IngestError(GroundtutorError):
    """A document could not be read, or it contained no usable text."""


class ConfigurationError(GroundtutorError):
    """A provider, embedder, store, or optional dependency is not usable."""


class ProviderError(GroundtutorError):
    """A language-model or embedding request failed."""


class GroundingError(GroundtutorError):
    """The model response could not be parsed into grounded claims."""
