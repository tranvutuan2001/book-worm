"""
LM Studio chat model — OpenAI-compatible HTTP API adapter.

``LMStudioChatModel`` wraps the ``openai`` Python client to talk to LM Studio's
local ``/v1/chat/completions`` endpoint.  It implements the
``pydantic_ai.models.Model`` interface (for use with Pydantic AI ``Agent``) and
also exposes a lower-level ``complete()`` method for non-agent inference used by
``LLMService.complete_chat``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelSettings

from src.domain.entity.chat_response import ChatResponse, ToolCall

logger = logging.getLogger("app.llm_connector")


def _tool_def_to_schema(tool_def: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_def.name,
            "description": tool_def.description or "",
            "parameters": tool_def.parameters_json_schema,
        },
    }


def _pydantic_messages_to_chat_dicts(
    messages: list[ModelMessage],
    model_request_parameters: ModelRequestParameters,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert pydantic_ai ModelMessages to plain chat dicts."""
    instructions = models.Model._get_instructions(messages, model_request_parameters)
    chat_dicts: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                kind = getattr(part, "part_kind", None)
                if kind == "user-prompt":
                    content = part.content
                    if not isinstance(content, str):
                        content = " ".join(
                            c if isinstance(c, str) else (c.get("text", "") if isinstance(c, dict) else "")
                            for c in content
                        )
                    chat_dicts.append({"role": "user", "content": content})
                elif kind == "tool-return":
                    chat_dicts.append({
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": str(part.content),
                    })
                elif kind == "retry-prompt":
                    chat_dicts.append({"role": "user", "content": str(part.content)})
        elif isinstance(msg, ModelResponse):
            text_parts = [p for p in msg.parts if getattr(p, "part_kind", None) == "text"]
            tool_parts = [p for p in msg.parts if getattr(p, "part_kind", None) == "tool-call"]
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg_dict["content"] = text_parts[0].content
            if tool_parts:
                msg_dict["tool_calls"] = [
                    {
                        "id": p.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": p.tool_name,
                            "arguments": (
                                json.dumps(p.args) if isinstance(p.args, dict) else (p.args or "{}")
                            ),
                        },
                    }
                    for p in tool_parts
                ]
            chat_dicts.append(msg_dict)

    return instructions, chat_dicts


class LMStudioChatModel(models.Model):
    """
    Chat model backed by LM Studio's OpenAI-compatible API.

    Implements ``pydantic_ai.models.Model`` for use with Pydantic AI ``Agent``
    and also exposes ``complete()`` for non-agent inference.

    Example::

        model = LMStudioChatModel(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="qwen2.5-7b-instruct",
        )
        resp = model.complete([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        super().__init__()
        self.model = model
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        logger.info(
            f"[LMStudioChatModel] Configured for model='{model}' at {base_url}"
        )

    # ------------------------------------------------------------------
    # pydantic_ai.models.Model interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return "lm_studio"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Run a single inference turn (called by Pydantic AI Agent)."""
        settings = model_settings or {}
        max_tokens: int = settings.get("max_tokens", 4000)
        temperature: float = settings.get("temperature", 0.1)
        frequency_penalty: float = settings.get("frequency_penalty", 0.0)
        json_schema: str | None = settings.get("json_schema", None)

        tool_schemas = [
            _tool_def_to_schema(t)
            for t in model_request_parameters.function_tools
        ] + [
            _tool_def_to_schema(t)
            for t in model_request_parameters.output_tools
        ]

        instructions, chat_dicts = _pydantic_messages_to_chat_dicts(
            messages, model_request_parameters
        )

        all_messages: list[dict[str, Any]] = []
        if instructions:
            all_messages.append({"role": "system", "content": instructions})
        all_messages.extend(chat_dicts)

        loop = asyncio.get_event_loop()
        chat_response: ChatResponse = await loop.run_in_executor(
            None,
            lambda: self.complete(
                all_messages,
                tool_schemas=tool_schemas or None,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                json_schema=json_schema,
            ),
        )

        parts: list[Any] = []
        for tc in chat_response.tool_calls:
            parts.append(ToolCallPart(tool_name=tc.name, args=tc.args, tool_call_id=tc.id))
        if chat_response.content:
            parts.append(TextPart(content=chat_response.content))
        if not parts:
            parts.append(TextPart(content=""))

        return ModelResponse(parts=parts)

    # ------------------------------------------------------------------
    # Lower-level inference (used by LLMService.complete_chat)
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, object]],
        *,
        tool_schemas: list[dict[str, object]] | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        frequency_penalty: float = 0,
        json_schema: str | None = None,
    ) -> ChatResponse:
        """Run a single chat completion via LM Studio.

        Args:
            messages:          List of ``{role, content, ...}`` dicts.
            tool_schemas:      OpenAI-style function tool schemas to expose.
            max_tokens:        Maximum number of tokens to generate.
            temperature:       Sampling temperature.
            frequency_penalty: Additive penalty for repeated tokens.
            json_schema:       Optional JSON Schema string for structured output.

        Returns:
            A ``ChatResponse`` with content and tool_calls.
        """
        logger.info(
            f"[LMStudioChatModel] Sending request model={self.model} "
            f"max_tokens={max_tokens} temperature={temperature}"
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": 120,
        }
        if frequency_penalty:
            kwargs["frequency_penalty"] = frequency_penalty

        if tool_schemas:
            kwargs["tools"] = tool_schemas

        if json_schema is not None:
            try:
                schema_dict = json.loads(json_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_dict.get("title", "output"),
                        "schema": schema_dict,
                        "strict": True,
                    },
                }
                logger.info("[LMStudioChatModel] Structured-output (json_schema) enabled")
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning(
                    f"[LMStudioChatModel] Could not parse json_schema; ignoring: {exc}"
                )

        response = self._client.chat.completions.create(**kwargs)
        logger.info("[LMStudioChatModel] Response received")

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(id=tc.id or "", name=tc.function.name, args=args)
                )

        return ChatResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
        )
