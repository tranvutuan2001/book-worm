import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from mlx_lm import load

logger = logging.getLogger("app.llm_connector")

# Project root: three levels up from this file (backend/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MLXModelBase:
    """
    Base class providing shared model-loading utilities for MLX-backed models.

    Subclasses (e.g. ``MLXChatModel``, ``MLXEmbeddingModel``) inherit
    :meth:`_resolve_model_path` and :meth:`_load_model` so that path
    resolution logic and the in-process model cache are not duplicated.

    The cache is held at the class level and is shared across all subclasses,
    so a model loaded for chat re-use will not be loaded again if the same
    path is later requested for embedding (or vice-versa).
    """

    # Shared across all subclasses
    _model_cache: Dict[str, Any] = {}

    @classmethod
    def _resolve_model_path(cls, model_path: str) -> Path:
        """
        Resolve *model_path* to an existing local directory.

        Handles:
        - Absolute paths (e.g. real local paths or Docker volume paths)
        - Docker-style ``/models/...`` paths that are remapped to
          ``<project_root>/models/...`` when running outside Docker.
        - Relative paths (resolved relative to the project root).

        Falls back to the original string unchanged so that ``mlx_lm`` can
        attempt a Hugging Face Hub download if the path does not exist locally.
        """
        p = Path(model_path)
        if p.exists():
            return p

        stripped = model_path.lstrip("/")
        candidate = _PROJECT_ROOT / stripped
        if candidate.exists():
            logger.debug(f"Resolved model path '{model_path}' → '{candidate}'")
            return candidate

        return p

    @classmethod
    def _load_model(cls, model_path: str) -> Tuple[Any, Any]:
        """
        Load and cache the MLX model + tokenizer at *model_path*.

        Subsequent calls with the same path return the cached pair without
        hitting disk again.

        Returns:
            A ``(model, tokenizer)`` tuple as returned by ``mlx_lm.load``.
        """
        if model_path not in cls._model_cache:
            resolved = cls._resolve_model_path(model_path)
            logger.info(f"Loading MLX model from: {resolved}")
            cls._model_cache[model_path] = load(str(resolved))
            logger.info(f"MLX model loaded successfully: {resolved}")
        return cls._model_cache[model_path]
