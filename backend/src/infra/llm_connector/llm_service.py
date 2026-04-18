"""
LLM inference service — powered by Pydantic AI agents.

This module is the single entry-point that the rest of the application uses
for chat completions (with or without tools) and embeddings.  All model
lifecycle (loading, caching, unloading) is delegated to :class:`LLMManager`.
"""

import logging
import os
from typing import List

from langfuse import get_client
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from src.config.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from src.domain.entity.agent import Agent as DomainAgent
from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_manager import LLMManager

logger = logging.getLogger("app.infra.llm_service")

# ---------------------------------------------------------------------------
# Langfuse monitoring setup
# ---------------------------------------------------------------------------
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

langfuse = get_client()
Agent.instrument_all()


class LLMService:
    """
    Handles LLM inference.

    Responsibilities
    ----------------
    * :meth:`agent_complete_chat` — chat completion with or without tool-calling (Pydantic AI Agent).
    * :meth:`embed_text`          — text embedding.

    All model lifecycle (loading, caching, unloading) is delegated to
    :class:`LLMManager`.  This class holds no model state of its own.
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self._manager = llm_manager

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def agent_complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        agent: DomainAgent,
    ) -> str:
        """
        Run a full chat turn with tool-calling support via Pydantic AI Agent.

        The domain :class:`~src.domain.entity.agent.Agent` is translated into
        a Pydantic AI ``Agent`` internally; callers never need to interact with
        the Pydantic AI API directly.

        Args:
            model_path:   Local path (or HF name) of the chat model.
            message_list: Conversation history as ``Message`` objects.
            agent:        Domain agent carrying the system prompt, tools,
                          model settings, and retry count.

        Returns:
            The final assistant text response.
        """
        backend = self._manager.get_chat_model(model_path)

        pydantic_agent: Agent[None, str] = Agent(
            model=backend,
            instructions=agent.system_prompt,
            retries=agent.max_retries,
            instrument=True,
            model_settings={
                "max_tokens": agent.model_settings.max_tokens,
                "temperature": agent.model_settings.temperature,
                "frequency_penalty": agent.model_settings.frequency_penalty,
                "json_schema": agent.model_settings.json_schema,
            },
        )

        for tool in agent.tools:
            pydantic_agent.tool(tool)

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

        result = pydantic_agent.run_sync(
            user_query,
            message_history=history if history else None,
        )

        langfuse.flush()
        logger.info(
            "Agent [%s] completed: %d messages, output length=%d",
            agent.agent_type.value,
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

