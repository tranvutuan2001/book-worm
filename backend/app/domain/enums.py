"""Domain-level enumerations."""

from enum import Enum


class Role(str, Enum):
    """Speaker role in a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentType(str, Enum):
    """Functional type of an LLM agent.

    Each value maps to a distinct role the agent plays in the pipeline.
    New agent types should be added here as the application grows.
    """

    SUMMARY = "summary"
    """Condenses text into a shorter, structured representation."""

    VERIFY = "verify"
    """Fact-checks a task/result pair; output must be 'yes' or 'no'."""

    DOCUMENT_ASSISTANT = "document_assistant"
    """Answers user questions about a document using retrieval tools."""
