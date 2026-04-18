"""Logging helpers for LLM interactions.

The LangChain ``BaseCallbackHandler`` has been removed.  This module is kept
as a placeholder for any future logging hooks (e.g. Pydantic AI
instrumentation).
"""

import logging

logger = logging.getLogger("app.llm_connector")


def log_tool_call(tool_name: str, tool_input: str) -> None:
    """Log a tool invocation."""
    logger.info("-" * 60)
    logger.info("Tool called: %s", tool_name)
    logger.info("Tool input: %.500s", tool_input)


def log_tool_result(tool_output: str) -> None:
    """Log a tool result."""
    logger.info("Tool output: %.500s", tool_output)
    logger.info("-" * 60)


def log_tool_error(error: Exception) -> None:
    """Log a tool error."""
    logger.error("Tool error: %s", error)

