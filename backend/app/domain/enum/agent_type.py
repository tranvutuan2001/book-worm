from enum import Enum

class AgentType(str, Enum):
    """Functional type of an LLM agent."""
    SUMMARY = "summary"
    VERIFY = "verify"
    DOCUMENT_ASSISTANT = "document_assistant"
