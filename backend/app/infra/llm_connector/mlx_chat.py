import json
import logging
import traceback
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from pydantic import Field

from mlx_lm import load as mlx_load

from app.infra.llm_connector.mlx_base import MLXModelBase
from app.infra.llm_connector.parsing_service import ParsingService

logger = logging.getLogger("app.llm_connector")


class MLXChatModel(MLXModelBase, BaseChatModel):
    """
    LangChain BaseChatModel implementation backed by a locally-stored MLX model.

    Uses mlx-lm for inference. The model files must already exist on disk at
    `model_path` in a format compatible with mlx-lm (e.g. converted Llama /
    Mistral / Phi weights).

    Tool-calling is supported for models whose tokenizer chat-template includes
    native function-calling markup (e.g. Llama-3.1 / Mistral function-calling
    variants). For models without native tool markup the bound tools are
    serialised as a JSON schema block inside the system prompt.
    """

    model_path: str = Field(description="Absolute path to the local MLX model directory")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    parsing_service: ParsingService = Field(description="Shared ParsingService used to parse raw model output")
    template_name: str = Field(default="qwen", description="Chat-template family name forwarded to ParsingService (e.g. 'qwen', 'openai')")

    # Per-class model cache, keyed by resolved absolute path
    _model_cache: ClassVar[Dict[str, Any]] = {}

    # Tools bound via bind_tools(); kept as serialisable dicts
    _bound_tools: List[Dict[str, Any]] = []

    @classmethod
    def _load_model(cls, model_path: str) -> Any:
        """
        Load and cache the MLX chat model + tokenizer at *model_path*.

        The resolved absolute path is used as the cache key.

        Returns:
            A ``(model, tokenizer)`` tuple as returned by ``mlx_lm.load``.
        """
        resolved = str(cls._resolve_model_path(model_path))
        if resolved not in cls._model_cache:
            logger.info(f"Loading MLX chat model from: {resolved}")
            cls._model_cache[resolved] = mlx_load(resolved)
            logger.info(f"MLX chat model loaded successfully: {resolved}")
        return cls._model_cache[resolved]

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> "MLXChatModel":
        """
        Return a copy of this model with the given tools attached.

        Each tool is converted to an OpenAI-style function-schema dict so that
        models whose chat-template understands that format will receive proper
        tool definitions.  For models without native tool-calling, the schemas
        are injected into the system prompt automatically inside `_generate`.
        """
        tool_schemas = []
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
            elif isinstance(t, dict):
                tool_schemas.append(t)
            else:
                # Assume it has a .schema() method (e.g. pydantic model)
                tool_schemas.append(t)

        new_instance = self.model_copy(deep=True)
        new_instance._bound_tools = tool_schemas
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
        try:
            model, tokenizer = self._load_model(self.model_path)
        except Exception as e:
            error_msg = f"Failed to load MLX chat model from '{self.model_path}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        chat_messages = self._to_chat_dicts(messages)

        # Build the prompt string via the tokenizer chat-template
        try:
            template_kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self._bound_tools:
                # Pass tools to the template if it accepts them; fall back to
                # injecting a JSON block into the system prompt otherwise.
                try:
                    prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tools=self._bound_tools,
                        **template_kwargs,
                    )
                except TypeError:
                    # Template doesn't accept 'tools' kwarg → inject manually
                    chat_messages = self._inject_tools_into_system(chat_messages)
                    prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
            else:
                prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
        except Exception as e:
            error_msg = f"Failed to apply chat template: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}\n{traceback.format_exc()}")
            raise

        # Run inference
        try:
            logger.info(
                f"Running MLX generation: model={self.model_path}, "
                f"max_tokens={self.max_tokens}, temperature={self.temperature}"
            )
            response_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=make_sampler(temp=self.temperature),
                verbose=False,
            )
            logger.info("MLX generation complete")
        except Exception as e:
            error_msg = f"MLX generate() failed: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}\n{traceback.format_exc()}")
            raise

        ai_message = self.parsing_service.parse(response_text, self.template_name)
        logger.info(
            f"Parsed response: tool_calls={len(ai_message.tool_calls)}, "
            f"has_thinking={'thinking' in ai_message.additional_kwargs}"
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chat_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain message objects to plain dicts for the tokenizer."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                d: Dict[str, Any] = {
                    "role": "assistant",
                    "content": str(msg.content),
                }
                if msg.tool_calls:
                    # Convert to the format the Qwen3 chat template expects:
                    # {"function": {"name": ..., "arguments": {...}}}
                    d["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["args"]}}
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif isinstance(msg, ToolMessage):
                # Qwen3 template expects tool responses as role="tool" with a
                # tool_call_id so it can pair them up.  Name is optional.
                d = {
                    "role": "tool",
                    "content": str(msg.content),
                    "tool_call_id": msg.tool_call_id or "",
                }
                if hasattr(msg, "name") and msg.name:
                    d["name"] = msg.name
                result.append(d)
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    @staticmethod
    def _inject_tools_into_system(
        chat_messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Prepend (or append to) the system message with JSON tool schemas."""
        if not tool_schemas:
            return chat_messages
        tool_block = (
            "\n\n# Available tools\n"
            "You may call one or more of the following functions to complete the request.\n"
            f"```json\n{json.dumps(tool_schemas, indent=2)}\n```"
        )
        messages = list(chat_messages)
        if messages and messages[0]["role"] == "system":
            messages[0] = {**messages[0], "content": messages[0]["content"] + tool_block}
        else:
            messages.insert(0, {"role": "system", "content": tool_block.strip()})
        return messages
