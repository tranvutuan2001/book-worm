from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod

from src.config.config import DEFAULT_CHAT_TEMPLATE
from src.domain.entity.chat_response import ChatResponse, ToolCall


logger = logging.getLogger("app.llm_connector")


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------

class BaseResponseParser(ABC):
    """Convert a raw model output string into a ``ChatResponse``."""

    @abstractmethod
    def parse(self, raw: str) -> ChatResponse: ...


# ---------------------------------------------------------------------------
# Concrete parsers — add new model families here
# ---------------------------------------------------------------------------

class Qwen3ResponseParser(BaseResponseParser):
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    # XML-param format: <function=NAME> ... <parameter=KEY>VALUE</parameter> ... </function>
    _FUNCTION_RE = re.compile(r"<function=([^>]+)>")
    _PARAM_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    def parse(self, raw: str) -> ChatResponse:
        thinking_blocks = self._THINK_RE.findall(raw)
        thinking_text = "\n".join(
            re.sub(r"^<think>|</think>$", "", b, flags=re.DOTALL).strip()
            for b in thinking_blocks
        )
        # Remove matched <think>…</think> blocks, then strip any orphaned
        # </think> closing tag the model may emit when it omits the opening tag,
        # along with all content that precedes it (the untagged thinking block).
        text = self._THINK_RE.sub("", raw)
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()

        tool_calls: list[ToolCall] = []
        for match in self._TOOL_CALL_RE.finditer(text):
            block = match.group(1).strip()
            tc = self._parse_tool_call_block(block)
            if tc:
                tool_calls.append(tc)

        content = self._TOOL_CALL_RE.sub("", text).strip()

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking_text or None,
        )

    @staticmethod
    def _parse_tool_call_block(block: str) -> ToolCall | None:
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
        fn_match = Qwen3ResponseParser._FUNCTION_RE.search(block)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            args: dict[str, object] = {}
            for p in Qwen3ResponseParser._PARAM_RE.finditer(block):
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


class Gemma4ResponseParser(BaseResponseParser):
    """
    Response parser for Google Gemma 4 models (e.g. ``gemma-4-27b-it``).

    Token formats (from the Gemma 4 Jinja chat template):

    * **Thinking block** — ``<|channel>thought\\n…\\n<channel|>``
      The model prepends a thinking channel when reasoning is enabled.
      Content inside is stripped and exposed via ``additional_kwargs["thinking"]``.

    * **Tool call** — ``<|tool_call>call:NAME{key:value,…}<tool_call|>``
      Keys are bare identifiers; string values are delimited by ``<|"|>``.
      Nested objects (``{...}``) and arrays (``[...]``) are supported.

    * **Plain text** — returned as ``content`` with no tool calls.
    """

    # Thinking: <|channel>thought\n…\n<channel|>
    _THINK_RE = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)
    # Opening of a tool-call token: <|tool_call>call:NAME{
    _TOOL_CALL_OPEN_RE = re.compile(r"<\|tool_call>call:([^{<]+)\{")
    # Closing sentinel
    _TOOL_CALL_CLOSE = "<tool_call|>"
    # Gemma string delimiter
    _GEMMA_QUOTE = '<|"|>'

    def parse(self, raw: str) -> ChatResponse:
        # --- 1. Extract and strip thinking blocks ---
        thinking_parts = self._THINK_RE.findall(raw)
        thinking_text = "\n".join(p.strip() for p in thinking_parts)
        text = self._THINK_RE.sub("", raw).strip()

        # --- 2. Extract tool calls (brace-aware so nested objects work) ---
        tool_calls: list[ToolCall] = []
        # track spans to remove from content
        remove_spans: list[tuple[int, int]] = []
        for m in self._TOOL_CALL_OPEN_RE.finditer(text):
            name = m.group(1).strip()
            args_str, end_pos = self._extract_braced_args(text, m.end() - 1)
            # find the closing sentinel after the closing brace
            close_idx = text.find(self._TOOL_CALL_CLOSE, end_pos)
            if close_idx == -1:
                logger.warning(
                    f"Gemma4ResponseParser: missing <tool_call|> sentinel after tool call '{name}'"
                )
                span_end = end_pos
            else:
                span_end = close_idx + len(self._TOOL_CALL_CLOSE)
            remove_spans.append((m.start(), span_end))
            args = self._parse_args(args_str)
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=name,
                args=args,
            ))

        # Remove tool-call tokens from content (in reverse order to keep indices valid)
        content = text
        for start, end in sorted(remove_spans, reverse=True):
            content = content[:start] + content[end:]
        content = content.strip()

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking_text or None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_braced_args(text: str, open_brace_pos: int) -> tuple[str, int]:
        """
        Starting at *open_brace_pos* (which must be ``'{'``), walk forward
        counting braces to find the matching ``'}'``.

        Returns ``(inner_str, pos_after_closing_brace)``.
        """
        assert text[open_brace_pos] == "{"
        depth = 0
        in_string = False
        i = open_brace_pos
        while i < len(text):
            ch = text[i]
            # Detect <|"|> string boundaries (3-char token)
            if text[i:i + 5] == '<|"|>':
                in_string = not in_string
                i += 5
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[open_brace_pos + 1 : i], i + 1
            i += 1
        # Unmatched brace — return everything after the opening brace
        return text[open_brace_pos + 1:], len(text)

    @classmethod
    def _parse_args(cls, args_str: str) -> dict[str, object]:
        """
        Convert Gemma 4's custom argument serialisation into a Python dict.

        The format uses ``<|"|>`` as string delimiters and bare identifier keys::

            param1:<|"|>hello<|"|>,param2:42,flag:true,nested:{key:<|"|>v<|"|>}

        Strategy:
        1. Replace ``<|"|>`` with standard ``"``.
        2. Quote bare identifier keys (unquoted word followed by ``:``).
        3. Wrap in ``{}`` and parse as JSON.
        """
        if not args_str.strip():
            return {}
        # Step 1 — normalise string delimiters
        s = args_str.replace(cls._GEMMA_QUOTE, '"')
        # Step 2 — quote unquoted keys: an identifier not already preceded by "
        # Matches: word boundary identifier followed by ':'
        # Negative lookbehind prevents double-quoting already-quoted keys.
        s = re.sub(r'(?<!["\w])([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*:)', r'"\1"', s)
        # Step 3 — parse as JSON
        try:
            return json.loads("{" + s + "}")
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Gemma4ResponseParser: could not parse args: {args_str!r} — {exc}"
            )
            return {}


# ---------------------------------------------------------------------------
# Registry  —  template_name (lowercase) → parser class
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseResponseParser]] = {
    "qwen": Qwen3ResponseParser,
    "gemma": Gemma4ResponseParser,
    # "openai":  OpenAIResponseParser,
    # "llama":   LlamaResponseParser,
}

# ---------------------------------------------------------------------------
# ParsingService
# ---------------------------------------------------------------------------

class ParsingService:
    """
    Central parsing service.

    Dispatches ``parse(raw, template_name)`` to the correct parser based on
    the ``template_name`` string (e.g. ``"qwen"``, ``"gemma"``).

    Usage::

        svc = ParsingService()
        ai_message = svc.parse(raw_text, template_name="qwen")
        ai_message = svc.parse(raw_text, template_name="gemma")
    """

    def __init__(self) -> None:
        self._parsers: dict[str, BaseResponseParser] = {
            name: cls() for name, cls in _REGISTRY.items()
        }

    def parse(self, raw: str, template_name: str) -> ChatResponse:
        parser = self._parsers.get(template_name.lower())
        if parser is None:
            logger.warning(
                f"ParsingService: unknown template '{template_name}', "
                f"falling back to '{DEFAULT_CHAT_TEMPLATE}'"
            )
            parser = self._parsers[DEFAULT_CHAT_TEMPLATE]
        return parser.parse(raw)


