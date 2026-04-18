"""
MLX chat model — Pydantic AI compatible local model.

``MLXChatModel`` loads an MLX model from disk and implements the
``pydantic_ai.models.Model`` interface so it can be passed directly to a
Pydantic AI ``Agent``.  It also exposes a lower-level ``complete()`` method
for non-agent (single-turn/multi-turn) inference used by ``LLMService``.

Tool schemas are injected into the tokenizer chat-template so that models
with native function-calling markup (Qwen, Gemma, etc.) work transparently.
JSON-schema constrained decoding is handled via xgrammar when available.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import mlx.nn as nn
from mlx_lm import generate, load as mlx_load
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelSettings

from src.domain.entity.chat_response import ChatResponse
from src.infra.llm_connector.local_llm.parsing_service import ParsingService
from src.infra.llm_connector.local_llm.xgrammar_processor import make_json_schema_logits_processor
from src.config.config import DEFAULT_CHAT_TEMPLATE

logger = logging.getLogger("app.llm_connector")

def _tool_def_to_schema(tool_def: Any) -> dict[str, Any]:
    """Convert a pydantic_ai ToolDefinition to an OpenAI-style tool schema dict."""
    return {
        "type": "function",
        "function": {
            "name": tool_def.name,
            "description": tool_def.description or "",
            "parameters": tool_def.parameters_json_schema,
        },
    }


def _messages_to_chat_dicts(
    messages: list[ModelMessage],
    model_request_parameters: ModelRequestParameters,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Convert Pydantic AI ``ModelMessage`` objects to plain chat dicts.

    Returns
    -------
    instructions : str | None
        System/instruction text for the current request.
    chat_dicts : list[dict]
        Conversation history (without system message).
    """
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
                                p.args
                                if isinstance(p.args, dict)
                                else (json.loads(p.args) if p.args else {})
                            ),
                        },
                    }
                    for p in tool_parts
                ]
            chat_dicts.append(msg_dict)

    return instructions, chat_dicts


class MLXChatModel(models.Model):
    """
    Local MLX chat model implementing the Pydantic AI ``Model`` interface.

    Loads weights from *model_path* on construction and holds the
    ``mlx_lm`` model/tokenizer pair in memory for the lifetime of the
    instance.

    Parameters
    ----------
    model_path:
        Absolute path to the model directory.
    parsing_service:
        Application-wide :class:`ParsingService` injected directly — used to
        extract tool calls and thinking blocks from raw model output.
    """

    def __init__(
        self,
        model_path: str,
        parsing_service: ParsingService,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._parsing_service = parsing_service
        self._template_name = DEFAULT_CHAT_TEMPLATE

        logger.info("[MLXChatModel] Loading model from '%s' …", model_path)
        self._mlx_model: nn.Module
        self._tokenizer: Any
        self._mlx_model, self._tokenizer = mlx_load(model_path)
        logger.info("[MLXChatModel] Model loaded: '%s'", model_path)

    # ------------------------------------------------------------------
    # pydantic_ai.models.Model interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Short identifier — the last directory component of the model path."""
        return Path(self._model_path).name

    @property
    def system(self) -> str:
        return "mlx"

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

        instructions, chat_dicts = _messages_to_chat_dicts(
            messages, model_request_parameters
        )

        loop = asyncio.get_event_loop()
        raw_output: str = await loop.run_in_executor(
            None,
            lambda: self._generate(
                instructions=instructions,
                chat_dicts=chat_dicts,
                tool_schemas=tool_schemas,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                json_schema=json_schema,
            ),
        )

        chat_response = self._parsing_service.parse(raw_output, self._template_name)
        return self._to_model_response(chat_response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        instructions: str | None,
        chat_dicts: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        """Apply the tokenizer chat template and return the prompt string."""
        full_messages: list[dict[str, Any]] = []
        if instructions:
            full_messages.append({"role": "system", "content": instructions})
        full_messages.extend(chat_dicts)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        try:
            prompt: str = self._tokenizer.apply_chat_template(full_messages, **kwargs)
        except Exception as exc:
            logger.warning(
                "[MLXChatModel] apply_chat_template failed (%s); "
                "falling back to plain concatenation",
                exc,
            )
            prompt = "\n".join(
                f"{m['role'].upper()}: {m.get('content', '')}"
                for m in full_messages
            )
        return prompt

    def _generate(
        self,
        *,
        instructions: str | None,
        chat_dicts: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        frequency_penalty: float,
        json_schema: str | None,
    ) -> str:
        """Run synchronous MLX generation and return the raw output string."""
        prompt = self._build_prompt(instructions, chat_dicts, tool_schemas)
        sampler = make_sampler(temp=temperature)

        logits_processors = make_logits_processors(
            logit_bias=None,
        )
        if json_schema:
            xg_processor = make_json_schema_logits_processor(
                self._tokenizer, json_schema
            )
            if xg_processor is not None:
                logits_processors = [xg_processor] + (logits_processors or [])

        logger.debug(
            "[MLXChatModel] Generating: max_tokens=%d temperature=%.2f tools=%d json_schema=%s",
            max_tokens,
            temperature,
            len(tool_schemas),
            bool(json_schema),
        )

        output: str = generate(
            self._mlx_model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
            verbose=False,
        )
        return output

    @staticmethod
    def _to_model_response(chat_response: ChatResponse) -> ModelResponse:
        """Convert a ``ChatResponse`` to a Pydantic AI ``ModelResponse``."""
        parts: list[Any] = []

        for tc in chat_response.tool_calls:
            parts.append(
                ToolCallPart(
                    tool_name=tc.name,
                    args=tc.args,
                    tool_call_id=tc.id,
                )
            )

        if chat_response.content:
            parts.append(TextPart(content=chat_response.content))

        if not parts:
            parts.append(TextPart(content=""))

        return ModelResponse(parts=parts)


class MLXChatModelFactory:
    """
    Thin factory that creates :class:`MLXChatModel` instances with
    *parsing_service* pre-injected.

    Intended to be registered as a singleton in the DI container so that
    :class:`~src.infra.llm_connector.llm_manager.LLMManager` never needs to
    depend on :class:`ParsingService` directly.
    """

    def __init__(self, parsing_service: ParsingService) -> None:
        self._parsing_service = parsing_service

    def __call__(self, model_path: str) -> MLXChatModel:
        return MLXChatModel(
            model_path=model_path,
            parsing_service=self._parsing_service,
        )
