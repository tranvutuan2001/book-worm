import logging
import threading
from pathlib import Path
from typing import Annotated, ClassVar, List, Literal, TypedDict

from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_logging_handler import LLMLoggingHandler
from src.infra.llm_connector.local_llm.mlx_base import MLXModelBase
from src.infra.llm_connector.local_llm.mlx_chat import MLXChatModel
from src.infra.llm_connector.local_llm.mlx_embedding import MLXEmbeddingModel
from src.infra.llm_connector.local_llm.parsing_service import ParsingService, get_parsing_service

logger = logging.getLogger("app.llm_connector")

ModelType = Literal["chat", "embedding"]


class LoadedModelRecord(TypedDict):
    """Lightweight record describing a model that is in memory or being loaded."""

    model_name: str
    model_path: str
    model_type: str   # "chat" | "embedding"
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loading_key(model_type: str, resolved_path: str) -> str:
        return f"{model_type}:{resolved_path}"

    @staticmethod
    def _cache_for(model_type: str) -> dict:
        if model_type == "embedding":
            return MLXEmbeddingModel._model_cache
        return MLXChatModel._model_cache

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
        resolved = str(MLXModelBase._resolve_model_path(model_path))
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
                MLXEmbeddingModel._load_model(resolved)
            else:
                MLXChatModel._load_model(resolved)
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
        resolved = str(MLXModelBase._resolve_model_path(model_path))
        cache = self._cache_for(model_type)

        with self._lock:
            if resolved not in cache:
                return False
            try:
                del cache[resolved]
            except KeyError:
                return False

        logger.info(f"[LLMService] Unloaded {model_type} model: {resolved}")
        return True

    def is_model_loaded(self, model_path: str, model_type: ModelType) -> bool:
        """Return ``True`` if the model is currently held in memory."""
        resolved = str(MLXModelBase._resolve_model_path(model_path))
        return resolved in self._cache_for(model_type)

    def is_model_loading(self, model_path: str, model_type: ModelType) -> bool:
        """Return ``True`` if a load operation is currently in progress."""
        resolved = str(MLXModelBase._resolve_model_path(model_path))
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
            for cache in (MLXChatModel._model_cache, MLXEmbeddingModel._model_cache):
                count += len(cache)
                cache.clear()
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

        for path_str in list(MLXChatModel._model_cache):
            results.append(LoadedModelRecord(
                model_name=Path(path_str).name,
                model_path=path_str,
                model_type="chat",
                loading=False,
            ))

        for path_str in list(MLXEmbeddingModel._model_cache):
            results.append(LoadedModelRecord(
                model_name=Path(path_str).name,
                model_path=path_str,
                model_type="embedding",
                loading=False,
            ))

        # Include models whose load is still in flight
        for key in list(self._loading):
            model_type, _, resolved = key.partition(":")
            results.append(LoadedModelRecord(
                model_name=Path(resolved).name,
                model_path=resolved,
                model_type=model_type,
                loading=True,
            ))

        return results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        template_name: str = "qwen",
        json_schema: str = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Run a pure chat completion without any agent or tool-calling.

        Args:
            model_path:    Local path (or HF name) of the MLX chat model.
            message_list:  Conversation history as ``Message`` objects.
            system_prompt: System instruction to prepend to the conversation.
            template_name: Chat-template family name used by ``model_path``
                           (e.g. ``"qwen"``).  Forwarded to
                           ``ParsingService`` to select the correct output
                           parser.
            json_schema:   Optional JSON Schema string.  When provided,
                           xgrammar constrained decoding is applied so the
                           model can only emit tokens that form a valid
                           sequence under the schema.
            max_tokens:    Maximum number of tokens to generate.  Default
                           is 4000.

        Returns:
            The assistant reply as a plain string.
        """
        llm = MLXChatModel(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=0.1,
            parsing_service=self._parsing_service,
            template_name=template_name,
            json_schema=json_schema,
        )

        lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for m in message_list:
            if m.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=m.content))
            elif m.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=m.content))
            else:
                lc_messages.append(HumanMessage(content=m.content))

        response = llm.invoke(
            lc_messages,
            config={"callbacks": [LLMLoggingHandler()]},
        )
        return response.content

    def agent_complete_chat(
        self,
        model_path: str,
        message_list: List[Message],
        system_prompt: str,
        tools: List[BaseTool],
        template_name: str = "qwen",
        max_iterations: int = None,
        json_schema: str = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Run a full chat turn with optional tool-calling support.

        Args:
            model_path:     Local path (or HF name) of the MLX chat model.
            message_list:   Conversation history as ``Message`` objects.
            system_prompt:  System instruction to prepend to the conversation.
            tools:          LangChain tools made available to the agent.
            template_name:  Chat-template family name used by ``model_path``
                            (e.g. ``"qwen"``).  Forwarded to
                            ``ParsingService`` to select the correct output
                            parser.
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
        llm = MLXChatModel(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=0.1,
            parsing_service=self._parsing_service,
            template_name=template_name,
            json_schema=json_schema,
        )
        agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
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
        return MLXEmbeddingModel(model_path).embed(text)


def get_llm_service(
    parsing_service: Annotated[ParsingService | None, Depends(get_parsing_service)] = None,
) -> LLMService:
    if LLMService._instance is None:
        if parsing_service is None:
            parsing_service = get_parsing_service()
        LLMService._instance = LLMService(parsing_service=parsing_service)
    return LLMService._instance