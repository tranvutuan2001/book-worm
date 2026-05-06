"""
LLM inference service — powered by a remote Multi-Provider LLM Server.

This module provides a unified interface for chat completions (with tool-calling)
and text embeddings by communicating with an external LLM server.
"""

import logging
from langfuse.openai import AsyncOpenAI
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from app.config.app_setting import app_setting
from app.domain.entity.agent import Agent as DomainAgent
from app.domain.entity.message import Message
from app.domain.enum.role import Role

logger = logging.getLogger("app.infra.llm_service")


class LLMService:
    """
    Handles LLM inference via a remote server.

    Responsibilities
    ----------------
    * :meth:`agent_complete_chat` — chat completion with tool-calling support.
    * :meth:`embed_text`          — text embedding.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        # We use Langfuse wrapper to get observability out of the box
        # API key is required by the client but may not be used by the remote server
        self._client = AsyncOpenAI(
            base_url=f"{self._base_url}",
            api_key="none",
        )

    async def agent_complete_chat(
        self,
        message_list: list[Message],
        domain_agent: DomainAgent,
    ) -> str:
        """
        Run a full chat turn with tool-calling support via Pydantic AI.

        Args:
            message_list: Conversation history as ``Message`` objects.
            agent:        Domain agent carrying the system prompt, tools,
                          and model settings.

        Returns:
            The final assistant text response.
        """
        # 1. Initialize Pydantic AI Model with the Langfuse-instrumented client
        model = OpenAIModel(
            model_name="",  # The remote server handles the actual model
            provider=OpenAIProvider(openai_client=self._client)
        )

        # 2. Create the Pydantic AI Agent
        pydantic_ai_agent = PydanticAgent(
            model=model,
            output_type=str,
            system_prompt=domain_agent.system_prompt,
            retries=domain_agent.max_retries
        )

        # 3. Register tools
        for tool_func in domain_agent.tools:
            pydantic_ai_agent.tool(tool_func)

        # 4. Map message history
        # Pydantic-AI expects ModelMessage (ModelRequest or ModelResponse)
        history: list[ModelRequest | ModelResponse] = []
        for msg in message_list[:-1]:
            if msg.role == Role.USER:
                history.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
            elif msg.role == Role.ASSISTANT:
                history.append(ModelResponse(parts=[TextPart(content=msg.content)]))

        # 5. Run the agent
        # The last message in message_list is the current user query
        user_query = message_list[-1].content if message_list else ""
        
        try:
            result = await pydantic_ai_agent.run(
                user_query,
                message_history=history,
                model_settings={
                    "max_tokens": app_setting.chat_max_tokens,
                    "temperature": domain_agent.model_settings.temperature,
                }
            )
            return result.output
        except Exception as e:
            logger.error(f"Pydantic AI agent run failed: {e}", exc_info=True)
            raise



    async def embed_text(self, text: str, model: str | None = None, *args, **kwargs) -> list[float]:
        """
        Create a text embedding using the remote embedding server.

        Args:
            text: The text to embed.
            model: Optional model name (handled natively by the OpenAI SDK)

        Returns:
            A list of floats representing the embedding vector.
        """
        response = await self._client.embeddings.create(
            input=text,
            model=model or "",
            name="",
            metadata={},
            **kwargs
        )
        return response.data[0].embedding
