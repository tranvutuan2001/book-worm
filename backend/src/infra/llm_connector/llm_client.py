import logging
from typing import List

from langchain.agents import create_agent
from langchain.tools import BaseTool

from src.domain.entity.message import Message
from src.infra.llm_connector.llm_logging_handler import LLMLoggingHandler
from src.infra.llm_connector.mlx_chat import MLXChatModel
from src.infra.llm_connector.mlx_embedding import MLXEmbeddingModel
from src.infra.llm_connector.parsing_service import ParsingService, _parsing_service

logger = logging.getLogger('app.llm_connector')


class LLMService:
    """
    Service layer wrapping MLX-based LLM inference.

    Receives a ``ParsingService`` instance which is forwarded to every
    ``MLXChatModel`` created by this client so that raw model output is
    parsed by the correct parser for the given chat template.

    Obtain an instance via the ``get_llm_service`` FastAPI dependency.
    """

    def __init__(self, parsing_service: ParsingService) -> None:
        self._parsing_service = parsing_service

    def complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        tools: List[BaseTool],
        template_name: str = "qwen",
        max_iterations: int = 25,
    ) -> str:
        """
        Run a full chat turn with optional tool-calling support.

        Args:
            model_name:     Local path (or HF name) of the MLX chat model.
            message_list:   Conversation history as ``Message`` objects.
            system_prompt:  System instruction to prepend to the conversation.
            tools:          LangChain tools made available to the agent.
            template_name:  Chat-template family name used by ``model_name``
                            (e.g. ``"qwen"``).  Forwarded to
                            ``ParsingService`` to select the correct output
                            parser.
            max_iterations: Maximum number of agent reasoning/tool-call cycles
                            before the agent is forced to stop.  Maps to
                            LangGraph's ``recursion_limit`` (default ``25``).
        """
        llm = MLXChatModel(
            model_path=model_path,
            max_tokens=2048,
            temperature=0.1,
            parsing_service=self._parsing_service,
            template_name=template_name,
        )
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
        messages = [{'role': m.role.value, 'content': m.content} for m in message_list]

        response = agent.invoke(
            input={"messages": messages},
            config={
                "callbacks": [LLMLoggingHandler()],
                "recursion_limit": max_iterations,
            },
        )
        return response["messages"][-1].content

    def embed_text(self, model_path: str, text: str) -> List[float]:
        """
        Create a text embedding using the local MLX embedding model.
        ``model_name`` is the local path to the MLX embedding model directory.
        """
        return MLXEmbeddingModel(model_path).embed(text)


# Singleton instance
_llm_service: LLMService = LLMService(parsing_service=_parsing_service)


def get_llm_service() -> LLMService:
    """
    FastAPI dependency factory for ``LLMService``.

    Usage::

        from fastapi import Depends
        from app.infra.llm_connector.llm_client import LLMService, get_llm_service

        async def my_route(llm_service: LLMService = Depends(get_llm_service)):
            ...
    """
    return _llm_service
