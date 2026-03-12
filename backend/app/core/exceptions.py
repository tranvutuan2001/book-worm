"""
Domain-level exceptions.

Services raise these exceptions; the API layer translates them into appropriate
HTTP error responses.  This keeps the service layer free from any web-framework
dependency.
"""


class BookWormError(Exception):
    """Base exception for all application errors."""


class DocumentNotFoundError(BookWormError):
    """Raised when a requested document does not exist on disk."""


class DocumentProcessingError(BookWormError):
    """Raised when document pre-analysis fails."""


class InvalidDocumentError(BookWormError):
    """Raised for invalid file type or missing filename."""


class LLMError(BookWormError):
    """Raised when the LLM call fails."""


class ModelNotFoundError(BookWormError):
    """Raised when a requested model does not exist on disk."""


class ModelLoadError(BookWormError):
    """Raised when loading a model into memory fails."""


class SessionError(BookWormError):
    """Raised for session-related failures."""
