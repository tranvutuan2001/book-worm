"""LangChain callback handler for logging LLM and tool interactions."""

import logging

from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger("app.llm_connector")


class LLMLoggingHandler(BaseCallbackHandler):
    """Logs all significant LangChain agent events (tool calls, chain steps)."""

    def on_tool_start(
        self, serialized: dict[str, object], input_str: str, **kwargs: object
    ) -> None:
        logger.info("-" * 60)
        logger.info("Tool called: %s", serialized.get("name", "unknown"))
        logger.info("Tool input: %.500s", input_str)

    def on_tool_end(self, output: str, **kwargs: object) -> None:
        logger.info("Tool output: %.500s", str(output))
        logger.info("-" * 60)

    def on_tool_error(self, error: Exception, **kwargs: object) -> None:
        logger.error("Tool error: %s", error)

    def on_agent_action(self, action: object, **kwargs: object) -> None:
        logger.info("Agent action: %s | input: %s", action.tool, action.tool_input)  # type: ignore[attr-defined]

    def on_agent_finish(self, finish: object, **kwargs: object) -> None:
        logger.info("Agent finished. Output: %.300s", str(finish.return_values))  # type: ignore[attr-defined]
        logger.info("=" * 80)

    def on_chain_start(
        self,
        serialized: dict[str, object],
        inputs: dict[str, object],
        **kwargs: object,
    ) -> None:
        chain_name = (serialized or {}).get("name", "unknown")
        logger.info("Chain start: %s | inputs: %.200s", chain_name, str(inputs))

    def on_chain_end(
        self, outputs: dict[str, object], **kwargs: object
    ) -> None:
        logger.info("Chain end. Outputs: %.200s", str(outputs))

    def on_chain_error(self, error: Exception, **kwargs: object) -> None:
        logger.error("Chain error: %s", error)
