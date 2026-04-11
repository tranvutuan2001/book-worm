import logging
from typing import List

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_logging_handler import LLMLoggingHandler
from src.infra.llm_connector.llm_manager import LLMManager

logger = logging.getLogger("app.llm_connector")


class LLMService:
    """
    Handles LLM inference.

    Responsibilities
    ----------------
    * :meth:`complete_chat`       — single-turn or multi-turn chat completion.
    * :meth:`agent_complete_chat` — chat completion with tool-calling support.
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
            model_path:    Local path (or HF name) of the MLX chat model.
            message_list:  Conversation history as ``Message`` objects.
            system_prompt: System instruction to prepend to the conversation.
            json_schema:   Optional JSON Schema string.  When provided,
                           xgrammar constrained decoding is applied so the
                           model can only emit tokens that form a valid
                           sequence under the schema.
            temperature:   Sampling temperature for this request.
            max_tokens:    Maximum number of tokens to generate.
            frequency_penalty: Frequency penalty for this request.

        Returns:
            The assistant reply as a plain string.
        """
        model = self._manager.get_chat_model(model_path)

        lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for message in message_list:
            if message.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=message.content))
            elif message.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=message.content))
            else:
                lc_messages.append(HumanMessage(content=message.content))

        response = (
            model
            .with_config(
                configurable={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "json_schema": json_schema,
                    "frequency_penalty": frequency_penalty,
                }
            )
            .invoke(
                lc_messages,
                config={
                    "callbacks": [LLMLoggingHandler()],
                },
            )
        )
        return response.content

    def agent_complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        tools: List[BaseTool],
        max_iterations: int = None,
        json_schema: str = None,
        max_tokens: int = None,
        temperature: float = None,
        frequency_penalty: float = None,
    ) -> str:
        """
        Run a full chat turn with optional tool-calling support.

        Args:
            model_path:     Local path (or HF name) of the MLX chat model.
            message_list:   Conversation history as ``Message`` objects.
            system_prompt:  System instruction to prepend to the conversation.
            tools:          LangChain tools made available to the agent.
            max_iterations: Maximum number of agent reasoning/tool-call cycles
                            before the agent is forced to stop.  Maps to
                            LangGraph's ``recursion_limit`` (default ``25``).
            json_schema:    Optional JSON Schema string for constrained decoding.
            max_tokens:     Maximum number of tokens to generate.
            temperature:    Sampling temperature for this request.
            frequency_penalty: Frequency penalty for this request.
        """
        model = self._manager.get_chat_model(model_path)

        agent = create_agent(
            model=model.with_config(
                configurable={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "json_schema": json_schema,
                    "frequency_penalty": frequency_penalty,
                }
            ),
            tools=tools,
            system_prompt=system_prompt,
        )
        messages = [{"role": m.role.value, "content": m.content} for m in message_list]

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
        Create a text embedding using the embedding model.

        Args:
            model_path: Local path to the embedding model directory.
            text:       The text to embed.

        Returns:
            A unit-normalised float vector.
        """
        model = self._manager.get_embedding_model(model_path)
        return model.embed(text)

