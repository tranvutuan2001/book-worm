import logging
import threading
from pathlib import Path
from typing import Annotated, Any, ClassVar, List, Literal, TypedDict

from fastapi import Depends
from langchain_core.runnables import ConfigurableField

from src.core import config
from src.infra.llm_connector.external_llm import LMStudioChatModel, LMStudioEmbeddingModel
from src.infra.llm_connector.local_llm.mlx_chat import MLXChatModel
from src.infra.llm_connector.local_llm.mlx_embedding import MLXEmbeddingModel
from src.infra.llm_connector.local_llm.parsing_service import (
    ParsingService,
    get_parsing_service,
)
from langchain.chat_models.base import BaseChatModel

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


class LLMManager:
    """
    Single point of responsibility for model lifecycle: loading, caching,
    unloading, and providing access to model instances.

    Responsibilities
    ----------------
    * Manages the in-memory cache for both chat and embedding models.
    * Tracks which models are currently being loaded or unloaded so that
      callers can avoid duplicate loads or race conditions.
    * Exposes lifecycle helpers: :meth:`load_model`, :meth:`unload_model`,
      :meth:`unload_all_models`, :meth:`is_model_loaded`,
      :meth:`is_model_loading`, :meth:`list_loaded_models`.
    * Exposes model-access helpers: :meth:`get_chat_model`,
      :meth:`get_embedding_model` — auto-loading a model if it is not yet
      in cache and returning the ready instance to callers.

    Obtain an instance via the :func:`get_llm_manager` FastAPI dependency.
    """

    # Class-level state shared across all instances.
    _loading: ClassVar[set[str]] = set()
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _instance: ClassVar["LLMManager | None"] = None

    def __init__(self, parsing_service: ParsingService) -> None:
        self._parsing_service = parsing_service
        # Instance-level model caches.
        # Keys are resolved local paths (local backend) or
        # "lm_studio:<model_type>:<model_id>" strings (lm_studio backend).
        self._chat_models: dict[str, Any] = {}
        self._embedding_models: dict[str, Any] = {}

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

    def _create_lm_studio_chat(self, model_path: str) -> BaseChatModel:
        """Create and return a new ``LMStudioChatModel`` instance (no caching)."""
        model_id = model_path or config.LM_STUDIO_DEFAULT_CHAT_MODEL
        logger.info(
            f"[LLMManager] Creating LMStudioChatModel: model='{model_id}' "
            f"base_url='{config.LM_STUDIO_BASE_URL}'"
        )
        return LMStudioChatModel(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
            model=model_id,
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

    def _create_lm_studio_embedding(self, model_path: str) -> Any:
        """Create and return a new ``LMStudioEmbeddingModel`` instance (no caching)."""
        model_id = model_path or config.LM_STUDIO_DEFAULT_EMBEDDING_MODEL
        logger.info(
            f"[LLMManager] Creating LMStudioEmbeddingModel: model='{model_id}' "
            f"base_url='{config.LM_STUDIO_BASE_URL}'"
        )
        return LMStudioEmbeddingModel(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
            model=model_id,
        )

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
        if config.LLM_BACKEND == "lm_studio":
            logger.info(
                "[LLMManager] load_model() is a no-op when LLM_BACKEND='lm_studio'. "
                "Models are managed by LM Studio."
            )
            return

        resolved = self._resolve_model_path(model_path)
        key = self._loading_key(model_type, resolved)
        cache = self._cache_for(model_type)

        with self._lock:
            if resolved in cache:
                logger.info(f"[LLMManager] Model already loaded: {resolved}")
                return
            if key in self._loading:
                logger.info(f"[LLMManager] Model already being loaded: {resolved}")
                return
            self._loading.add(key)

        try:
            logger.info(f"[LLMManager] Loading {model_type} model: {resolved}")
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
            logger.info(f"[LLMManager] Model loaded successfully: {resolved}")
        except Exception:
            logger.exception(f"[LLMManager] Failed to load model: {resolved}")
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

        logger.info(f"[LLMManager] Unloaded {model_type} model: {resolved}")
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
        logger.info(f"[LLMManager] Unloaded all models ({count} total)")
        return count

    def list_loaded_models(self) -> List[LoadedModelRecord]:
        """
        Return metadata for every model that is loaded or currently loading.

        Returns:
            A list of :class:`LoadedModelRecord` dicts.
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
    # Model access (used by LLMService)
    # ------------------------------------------------------------------

    def get_chat_model(self, model_path: str) -> BaseChatModel:
        """
        Return the chat model instance for *model_path*, auto-loading if needed.

        For LM Studio backends the model is lazily created on first call.
        For local MLX backends the model is loaded from disk if not cached.
        """
        if config.LLM_BACKEND == "lm_studio":
            return self._create_lm_studio_chat(model_path)

        resolved = self._resolve_model_path(model_path)
        if resolved not in self._chat_models:
            logger.info(f"[LLMManager] Auto-loading chat model: {resolved}")
            self.load_model(model_path, "chat")
        return self._chat_models[resolved]

    def get_embedding_model(self, model_path: str) -> Any:
        """
        Return the embedding model instance for *model_path*, auto-loading if needed.
        """
        if config.LLM_BACKEND == "lm_studio":
            return self._create_lm_studio_embedding(model_path)

        resolved = self._resolve_model_path(model_path)
        if resolved not in self._embedding_models:
            logger.info(f"[LLMManager] Auto-loading embedding model: {resolved}")
            self._embedding_models[resolved] = MLXEmbeddingModel(resolved)
        return self._embedding_models[resolved]


def get_llm_manager(
    parsing_service: Annotated[
        ParsingService | None, Depends(get_parsing_service)
    ] = None,
) -> LLMManager:
    if LLMManager._instance is None:
        if parsing_service is None:
            parsing_service = get_parsing_service()
        LLMManager._instance = LLMManager(parsing_service=parsing_service)
    return LLMManager._instance
