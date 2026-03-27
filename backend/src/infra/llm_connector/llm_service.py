import logging
import threading
from pathlib import Path
from typing import Annotated, ClassVar, List, Literal, TypedDict

from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableField

from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_logging_handler import LLMLoggingHandler
from src.infra.llm_connector.local_llm.mlx_chat import MLXChatModel
from src.infra.llm_connector.local_llm.mlx_embedding import MLXEmbeddingModel
from src.infra.llm_connector.local_llm.parsing_service import (
    ParsingService,
    get_parsing_service,
)

logger = logging.getLogger("app.llm_connector")

# Project root: four levels up from this file (backend/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

ModelType = Literal["chat", "embedding"]


class LoadedModelRecord(TypedDict):
    """Lightweight record describing a model that is in memory or being loaded."""

    model_name: str
    model_path: str
    model_type: str  # "chat" | "embedding"
    loading: bool


class LLMService:
    """
    Single entry point for all LLM and model-lifecycle operations.

    Responsibilities
    ----------------
    * Manages the in-memory cache for both chat and embedding models.
    * Tracks which models are currently being loaded or unloaded so that
      callers can avoid duplicate loads or race conditions.
    * Exposes inference helpers: :meth:`complete_chat` and :meth:`embed_text`.
    * Exposes lifecycle helpers: :meth:`load_model`, :meth:`unload_model`,
      :meth:`is_model_loaded`, :meth:`is_model_loading`,
      :meth:`list_loaded_models`.

    All modules **outside** ``llm_connector`` must interact with models
    exclusively through this class — direct imports of ``MLXChatModel``,
    ``MLXEmbeddingModel``, or their caches are not permitted.

    ``ParsingService`` is injected via FastAPI's ``Depends`` mechanism.

    Obtain an instance via the :func:`get_llm_service` FastAPI dependency.
    """

    # Class-level state shared across all instances.
    _loading: ClassVar[set[str]] = set()
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _instance: ClassVar["LLMService | None"] = None
    _parsing_service: ParsingService

    def __init__(self, parsing_service: ParsingService) -> None:
        self._parsing_service = parsing_service
        # Instance-level model caches — keyed by resolved absolute path.
        self._chat_models: dict[str, MLXChatModel] = {}
        self._embedding_models: dict[str, MLXEmbeddingModel] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loading_key(model_type: str, resolved_path: str) -> str:
        return f"{model_type}:{resolved_path}"

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """
        Resolve *model_path* to an existing local directory.

        Resolution order:
        1. Absolute path — used as-is if it exists.
        2. Docker-style ``/models/...`` path — remapped to
           ``<project_root>/models/...`` when running outside Docker.
        3. Path relative to ``<project_root>`` (e.g. ``models/chat/...``).
        4. Path relative to ``<project_root>/models/chat/``.
        5. Path relative to ``<project_root>/models/embedding/``.

        Falls back to the original string unchanged so that ``mlx_lm`` can
        attempt a Hugging Face Hub download if the path does not exist locally.
        """
        p = Path(model_path)
        if p.exists():
            return str(p)

        stripped = model_path.lstrip("/")
        for candidate in (
            _PROJECT_ROOT / stripped,
            _PROJECT_ROOT / "models" / "chat" / stripped,
            _PROJECT_ROOT / "models" / "embedding" / stripped,
        ):
            if candidate.exists():
                logger.debug(f"Resolved model path '{model_path}' → '{candidate}'")
                return str(candidate)

        return model_path

    def _cache_for(self, model_type: str) -> dict:
        if model_type == "embedding":
            return self._embedding_models
        return self._chat_models

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, model_type: ModelType) -> None:
        """
        Load a model into the in-memory cache.

        No-op if the model is already loaded.  Marks the model as *loading*
        for the duration of the operation so that :meth:`is_model_loading`
        reflects the in-progress state.

        Args:
            model_path: Absolute path (or project-relative path) to the model
                        directory.
            model_type: ``"chat"`` or ``"embedding"``.

        Raises:
            Exception: Propagates any error thrown by the underlying MLX load.
        """
        resolved = self._resolve_model_path(model_path)
        key = self._loading_key(model_type, resolved)
        cache = self._cache_for(model_type)

        with self._lock:
            if resolved in cache:
                logger.info(f"[LLMService] Model already loaded: {resolved}")
                return
            if key in self._loading:
                logger.info(f"[LLMService] Model already being loaded: {resolved}")
                return
            self._loading.add(key)

        try:
            logger.info(f"[LLMService] Loading {model_type} model: {resolved}")
            if model_type == "embedding":
                self._embedding_models[resolved] = MLXEmbeddingModel(resolved)
            else:
                self._chat_models[resolved] = MLXChatModel(
                    model_path=resolved,
                    parsing_service=self._parsing_service,
                ).configurable_fields(
                    max_tokens=ConfigurableField(
                        id="max_tokens",
                        name="Max Tokens",
                        description="Maximum number of tokens to generate for this run",
                    ),
                    temperature=ConfigurableField(
                        id="temperature",
                        name="Temperature",
                        description="Sampling temperature for this run",
                    ),
                    json_schema=ConfigurableField(
                        id="json_schema",
                        name="JSON Schema",
                        description="Optional JSON schema for constrained decoding",
                    ),
                )
            logger.info(f"[LLMService] Model loaded successfully: {resolved}")
        except Exception:
            logger.exception(f"[LLMService] Failed to load model: {resolved}")
            raise
        finally:
            with self._lock:
                self._loading.discard(key)

    def unload_model(self, model_path: str, model_type: ModelType) -> bool:
        """
        Remove a model from the in-memory cache.

        Args:
            model_path: Absolute or project-relative path to the model.
            model_type: ``"chat"`` or ``"embedding"``.

        Returns:
            ``True`` if the model was found and removed, ``False`` if it was
            not loaded.
        """
        resolved = self._resolve_model_path(model_path)
        cache = self._cache_for(model_type)

        with self._lock:
            if resolved not in cache:
                return False
            del cache[resolved]

        logger.info(f"[LLMService] Unloaded {model_type} model: {resolved}")
        return True

    def is_model_loaded(self, model_path: str, model_type: ModelType) -> bool:
        """Return ``True`` if the model is currently held in memory."""
        resolved = self._resolve_model_path(model_path)
        return resolved in self._cache_for(model_type)

    def is_model_loading(self, model_path: str, model_type: ModelType) -> bool:
        """Return ``True`` if a load operation is currently in progress."""
        resolved = self._resolve_model_path(model_path)
        key = self._loading_key(model_type, resolved)
        return key in self._loading

    def unload_all_models(self) -> int:
        """
        Remove every cached model from memory.

        Returns:
            The number of models that were unloaded.
        """
        count = 0
        with self._lock:
            count += len(self._chat_models) + len(self._embedding_models)
            self._chat_models.clear()
            self._embedding_models.clear()
        logger.info(f"[LLMService] Unloaded all models ({count} total)")
        return count

    def list_loaded_models(self) -> List[LoadedModelRecord]:
        """
        Return metadata for every model that is loaded or currently loading.

        Returns:
            A list of :class:`LoadedModelRecord` dicts.  Callers in the
            service layer should convert these to their own response types.
        """
        results: List[LoadedModelRecord] = []

        for path_str in list(self._chat_models):
            results.append(
                LoadedModelRecord(
                    model_name=Path(path_str).name,
                    model_path=path_str,
                    model_type="chat",
                    loading=False,
                )
            )

        for path_str in list(self._embedding_models):
            results.append(
                LoadedModelRecord(
                    model_name=Path(path_str).name,
                    model_path=path_str,
                    model_type="embedding",
                    loading=False,
                )
            )

        # Include models whose load is still in flight
        for key in list(self._loading):
            model_type, _, resolved = key.partition(":")
            results.append(
                LoadedModelRecord(
                    model_name=Path(resolved).name,
                    model_path=resolved,
                    model_type=model_type,
                    loading=True,
                )
            )

        return results

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
            max_tokens:    Maximum number of tokens to generate.  Default
                           is 4000.

        Returns:
            The assistant reply as a plain string.
        """
        resolved = self._resolve_model_path(model_path)
        if resolved not in self._chat_models:
            logger.info(
                f"[LLMService] Auto-loading chat model for inference: {resolved}"
            )
            self.load_model(model_path, "chat")

        lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for message in message_list:
            if message.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=message.content))
            elif message.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=message.content))
            else:
                lc_messages.append(HumanMessage(content=message.content))

        response = (
            self._chat_models[resolved]
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
            json_schema:    Optional JSON Schema string.  When provided,
                            xgrammar constrained decoding is applied so the
                            model can only emit tokens that form a valid
                            sequence under the schema.
            max_tokens:     Maximum number of tokens to generate.  Default
                            is 4000; increase for large structured outputs.
        """
        resolved = self._resolve_model_path(model_path)
        if resolved not in self._chat_models:
            logger.info(
                f"[LLMService] Auto-loading chat model for agent inference: {resolved}"
            )
            self.load_model(model_path, "chat")

        agent = create_agent(
            model=self._chat_models[resolved].with_config(
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
        Create a text embedding using the local MLX embedding model.

        Args:
            model_path: Local path to the MLX embedding model directory.
            text:       The text to embed.

        Returns:
            A unit-normalised float vector.
        """
        resolved = self._resolve_model_path(model_path)
        if resolved not in self._embedding_models:
            logger.info(
                f"[LLMService] Auto-loading embedding model for inference: {resolved}"
            )
            self._embedding_models[resolved] = MLXEmbeddingModel(resolved)
        return self._embedding_models[resolved].embed(text)


def get_llm_service(
    parsing_service: Annotated[
        ParsingService | None, Depends(get_parsing_service)
    ] = None,
) -> LLMService:
    if LLMService._instance is None:
        if parsing_service is None:
            parsing_service = get_parsing_service()
        LLMService._instance = LLMService(parsing_service=parsing_service)
    return LLMService._instance
