"""
LM Studio chat model — OpenAI-compatible HTTP API adapter.

``LMStudioChatModel`` is a LangChain ``BaseChatModel`` backed by LM Studio's
local OpenAI-compatible endpoint.  It exposes the same ``configurable_fields``
interface as ``MLXChatModel`` so that ``LLMService`` can treat both backends
uniformly.

JSON-schema constrained output is supported via the OpenAI ``response_format``
parameter (requires LM Studio ≥ 0.3.x with structured-output support).
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

logger = logging.getLogger("app.llm_connector")


class LMStudioChatModel(BaseChatModel):
    """
    LangChain ``BaseChatModel`` backed by LM Studio's OpenAI-compatible API.

    LM Studio exposes the same ``/v1/chat/completions`` interface as the
    OpenAI API.  This class tunnels inference requests to that endpoint via
    ``langchain_openai.ChatOpenAI`` while keeping the same field contract as
    ``MLXChatModel`` so that ``LLMService`` needs no special-casing logic.

    Tool-calling is fully delegated to ``ChatOpenAI`` and therefore works for
    any LM Studio model that exposes native function-calling support.

    Example::

        model = LMStudioChatModel(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="qwen2.5-7b-instruct",
        )
        model.invoke([HumanMessage(content="Hello")])
    """

    base_url: str = Field(description="LM Studio API base URL, e.g. http://localhost:1234/v1")
    api_key: str = Field(description="LM Studio API key (any non-empty string is accepted)")
    model: str = Field(description="Model identifier as listed in LM Studio")
    max_tokens: int = Field(default=4000, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    json_schema: Optional[str] = Field(
        default=None,
        description="Optional JSON Schema string. When set, LM Studio structured-output is enabled.",
    )

    # Internal ChatOpenAI client — rebuilt whenever key fields change.
    _client: ChatOpenAI = PrivateAttr()
    _bound_tools: list[dict[str, object]] = []

    def model_post_init(self, __context: Any) -> None:
        """Initialise the underlying ``ChatOpenAI`` client."""
        self._bound_tools = []
        self._rebuild_client()

    def _rebuild_client(self) -> None:
        """Recreate the ``ChatOpenAI`` instance from the current field values."""
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": 120,
        }
        if self.json_schema is not None:
            try:
                schema_dict = json.loads(self.json_schema)
                kwargs["model_kwargs"] = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_dict.get("title", "output"),
                            "schema": schema_dict,
                            "strict": True,
                        },
                    }
                }
                logger.info("[LMStudioChatModel] Structured-output (json_schema) enabled")
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning(
                    f"[LMStudioChatModel] Could not parse json_schema; ignoring: {exc}"
                )

        self._client = ChatOpenAI(**kwargs)
        if self._bound_tools:
            self._client = self._client.bind_tools(self._bound_tools)  # type: ignore[assignment]

    @property
    def _llm_type(self) -> str:
        return "lm-studio-chat"

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, object]],
        **kwargs: object,
    ) -> "LMStudioChatModel":
        """
        Return a copy with the given tools bound.

        Delegates to the underlying ``ChatOpenAI`` client for proper
        OpenAI-style function-call formatting.
        """
        tool_schemas: list[dict[str, object]] = []
        for t in tools:
            if isinstance(t, BaseTool):
                tool_schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.args_schema.schema() if t.args_schema else {},
                        },
                    }
                )
            else:
                tool_schemas.append(t)  # type: ignore[arg-type]

        new_instance = self.model_copy()
        new_instance._bound_tools = tool_schemas
        new_instance._rebuild_client()
        return new_instance

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        logger.info(
            f"[LMStudioChatModel] Sending request to {self.base_url} "
            f"model={self.model} max_tokens={self.max_tokens} temperature={self.temperature}"
        )
        result: ChatResult = self._client._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        logger.info("[LMStudioChatModel] Response received")
        return result
