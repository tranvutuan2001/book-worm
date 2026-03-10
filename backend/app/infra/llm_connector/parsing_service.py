from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

logger = logging.getLogger("app.llm_connector")


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------

class BaseResponseParser(ABC):
    """Convert a raw model output string into a LangChain ``AIMessage``."""

    @abstractmethod
    def parse(self, raw: str) -> AIMessage: ...


# ---------------------------------------------------------------------------
# Concrete parsers — add new model families here
# ---------------------------------------------------------------------------

class Qwen3ResponseParser(BaseResponseParser):
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    # XML-param format: <function=NAME> ... <parameter=KEY>VALUE</parameter> ... </function>
    _FUNCTION_RE = re.compile(r"<function=([^>]+)>")
    _PARAM_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    def parse(self, raw: str) -> AIMessage:
        thinking_blocks = self._THINK_RE.findall(raw)
        thinking_text = "\n".join(
            re.sub(r"^<think>|</think>$", "", b, flags=re.DOTALL).strip()
            for b in thinking_blocks
        )
        # Remove matched <think>…</think> blocks, then strip any orphaned
        # </think> closing tag the model may emit when it omits the opening tag.
        text = self._THINK_RE.sub("", raw)
        # text = re.sub(r"</think>", "", text).strip()

        tool_calls: List[ToolCall] = []
        for match in self._TOOL_CALL_RE.finditer(text):
            block = match.group(1).strip()
            tc = self._parse_tool_call_block(block)
            if tc:
                tool_calls.append(tc)

        content = self._TOOL_CALL_RE.sub("", text).strip()
        additional_kwargs: Dict[str, Any] = {}
        if thinking_text:
            additional_kwargs["thinking"] = thinking_text

        return AIMessage(content=content, tool_calls=tool_calls, additional_kwargs=additional_kwargs)

    def _parse_tool_call_block(self, block: str) -> ToolCall | None:
        # Format 1 — standard JSON: {"name": "...", "arguments": {...}}
        try:
            data = json.loads(block)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=data["name"],
                args=data.get("arguments", {}),
            )
        except (json.JSONDecodeError, KeyError):
            pass

        # Format 2 — XML-param: <function=NAME>\n<parameter=K>V</parameter>\n</function>
        fn_match = self._FUNCTION_RE.search(block)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            args: Dict[str, Any] = {}
            for p in self._PARAM_RE.finditer(block):
                key = p.group(1).strip()
                value_raw = p.group(2).strip()
                try:
                    args[key] = json.loads(value_raw)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value_raw
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=fn_name,
                args=args,
            )

        logger.warning(f"Qwen3ResponseParser: unrecognised tool_call block: {block!r}")
        return None


# ---------------------------------------------------------------------------
# Registry  —  template_name (lowercase) → parser class
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseResponseParser]] = {
    "qwen": Qwen3ResponseParser,
    # "openai":  OpenAIResponseParser,
    # "llama":   LlamaResponseParser,
}

_DEFAULT_TEMPLATE = "qwen"


# ---------------------------------------------------------------------------
# ParsingService
# ---------------------------------------------------------------------------

class ParsingService:
    """
    Central parsing service.

    Dispatches ``parse(raw, template_name)`` to the correct parser based on
    the ``template_name`` string (e.g. ``"qwen"``, ``"openai"``).

    Usage::

        svc = ParsingService()
        ai_message = svc.parse(raw_text, template_name="qwen")
    """

    def __init__(self) -> None:
        self._parsers: Dict[str, BaseResponseParser] = {
            name: cls() for name, cls in _REGISTRY.items()
        }

    def parse(self, raw: str, template_name: str) -> AIMessage:
        parser = self._parsers.get(template_name.lower())
        if parser is None:
            logger.warning(
                f"ParsingService: unknown template '{template_name}', "
                f"falling back to '{_DEFAULT_TEMPLATE}'"
            )
            parser = self._parsers[_DEFAULT_TEMPLATE]
        return parser.parse(raw)


# Module-level singleton — stateless, safe to share across requests
_parsing_service = ParsingService()


def get_parsing_service() -> ParsingService:
    """FastAPI dependency that provides the shared ``ParsingService`` instance."""
    return _parsing_service
