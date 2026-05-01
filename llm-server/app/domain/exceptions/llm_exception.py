from typing import Optional

class LLMGenerationException(Exception):
    """Custom exception for LLM generation errors across all providers."""
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.original_error = original_error
