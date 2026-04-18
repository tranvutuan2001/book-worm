"""
LLM inference service — powered by Pydantic AI agents.

This module is the single entry-point that the rest of the application uses
for chat completions (with or without tools) and embeddings.  All model
lifecycle (loading, caching, unloading) is delegated to :class:`LLMManager`.
"""

import logging
from typing import Any, Callable, List

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_manager import LLMManager

logger = logging.getLogger("app.llm_connector")


class LLMService:
    """
    Handles LLM inference.

    Responsibilities
    ----------------
    * :meth:`complete_chat`       — single-turn or multi-turn chat completion.
    * :meth:`agent_complete_chat` — chat completion with tool-calling (Pydantic AI Agent).
    * :meth:`embed_text`          — text embedding.

    All model lifecycle (loading, caching, unloading) is delegated to
    :class:`LLMManager`.  This class holds no model state of its own.
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self._manager = llm_manager

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        json_schema: str = None,
        temperature: float = None,
        max_tokens: int = None,
        frequency_penalty: float = None,
    ) -> str:
        """
        Run a pure chat completion without any agent or tool-calling.

        Args:
            model_path:        Local path (or HF name) of the chat model.
            message_list:      Conversation history as ``Message`` objects.
            system_prompt:     System instruction to prepend to the conversation.
            json_schema:       Optional JSON Schema string for constrained decoding.
            temperature:       Sampling temperature for this request.
            max_tokens:        Maximum number of tokens to generate.
            frequency_penalty: Frequency penalty for this request.

        Returns:
            The assistant reply as a plain string.
        """
        backend = self._manager.get_chat_model(model_path)

        agent: Agent[None, str] = Agent(
            model=backend,
            instructions=system_prompt,
            model_settings={
                "max_tokens": max_tokens or 4000,
                "temperature": temperature or 0.1,
                "frequency_penalty": frequency_penalty or 0,
                "json_schema": json_schema,
            },
        )

        history: list[ModelMessage] = []
        for msg in message_list[:-1]:
            if msg.role == Role.USER:
                history.append(
                    ModelRequest(parts=[UserPromptPart(content=msg.content)])
                )
            elif msg.role == Role.ASSISTANT:
                history.append(
                    ModelResponse(parts=[TextPart(content=msg.content)])
                )

        user_query = message_list[-1].content if message_list else ""

        result = agent.run_sync(
            user_query,
            message_history=history if history else None,
        )
        return result.output

    def agent_complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        tools: List[Callable[..., Any]],
        max_retries: int = 3,
        json_schema: str = None,
        max_tokens: int = None,
        temperature: float = None,
        frequency_penalty: float = None,
    ) -> str:
        """
        Run a full chat turn with tool-calling support via Pydantic AI Agent.

        Args:
            model_path:        Local path (or HF name) of the chat model.
            message_list:      Conversation history as ``Message`` objects.
            system_prompt:     System instruction to prepend to the conversation.
            tools:             Plain Python functions decorated with
                               ``pydantic_ai.tool`` (or bare callables).
            max_retries:       Maximum retries on validation errors (default 3).
            json_schema:       Optional JSON Schema string for constrained decoding.
            max_tokens:        Maximum number of tokens to generate.
            temperature:       Sampling temperature for this request.
            frequency_penalty: Frequency penalty for this request.

        Returns:
            The final assistant text response.
        """
        backend = self._manager.get_chat_model(model_path)

        agent: Agent[None, str] = Agent(
            model=backend,
            instructions=system_prompt,
            retries=max_retries,
            model_settings={
                "max_tokens": max_tokens or 4000,
                "temperature": temperature or 0.1,
                "frequency_penalty": frequency_penalty or 0,
                "json_schema": json_schema,
            },
        )

        for tool in tools:
            agent.tool(tool)

        history: list[ModelMessage] = []
        for msg in message_list[:-1]:
            if msg.role == Role.USER:
                history.append(
                    ModelRequest(parts=[UserPromptPart(content=msg.content)])
                )
            elif msg.role == Role.ASSISTANT:
                history.append(
                    ModelResponse(parts=[TextPart(content=msg.content)])
                )

        user_query = message_list[-1].content if message_list else ""

        result = agent.run_sync(
            user_query,
            message_history=history if history else None,
        )

        logger.info(
            "Agent completed: %d messages, output length=%d",
            len(result.all_messages()),
            len(result.output),
        )
        return result.output

    def embed_text(self, model_path: str, text: str) -> List[float]:
        """
        Create a text embedding using the embedding model.

        Args:
            model_path: Local path to the embedding model directory.
            text:       The text to embed.

        Returns:
            A unit-normalised float vector.
        """
        model = self._manager.get_embedding_model(model_path)
        return model.embed(text)

