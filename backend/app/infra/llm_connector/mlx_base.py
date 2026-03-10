import logging
from pathlib import Path

logger = logging.getLogger("app.llm_connector")

# Project root: three levels up from this file (backend/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MLXModelBase:
    """
    Base class providing shared path-resolution utilities for MLX-backed models.

    Each subclass owns its own ``_model_cache`` and ``_load_model`` method so
    that chat and embedding models are fully independent.  Callers must always
    supply a path (absolute, project-relative, or Docker-style ``/models/...``)
    — plain model names are not supported.
    """

    # ---------------------------------------------------------------------------
    # Path resolution
    # ---------------------------------------------------------------------------

    @classmethod
    def _resolve_model_path(cls, model_path: str) -> Path:
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
            return p

        stripped = model_path.lstrip("/")

        for candidate in (
            _PROJECT_ROOT / stripped,
            _PROJECT_ROOT / "models" / "chat" / stripped,
            _PROJECT_ROOT / "models" / "embedding" / stripped,
        ):
            if candidate.exists():
                logger.debug(f"Resolved model path '{model_path}' → '{candidate}'")
                return candidate

        return p
