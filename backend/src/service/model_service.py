import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, ClassVar, List

from fastapi import Depends

from src.api.schemas.model import (
    DownloadableModelInfo,
    LoadedModelInfo,
    ModelDownloadResponse,
    ModelInfo,
    ModelLoadResponse,
    ModelUnloadResponse,
)
from src.infra.llm_connector.llm_service import LLMService, get_llm_service

logger = logging.getLogger("app.model_service")
_downloading: set[str] = set()

class ModelService:
    """
    Manages discovery, downloading, loading, and unloading of MLX models.

    All in-memory model cache operations are delegated to :class:`LLMService`
    which is the sole owner of loaded model state.  This service is responsible
    only for filesystem scanning, HuggingFace downloads, and translating
    lifecycle calls into API response objects.
    """

    # project_root/backend/, two levels up from app/service/
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _MODELS_DIR = _PROJECT_ROOT / "models"
    _CHAT_MODELS_DIR = _MODELS_DIR / "chat"
    _EMBEDDING_MODELS_DIR = _MODELS_DIR / "embedding"
    _DOWNLOADABLE_CHAT_MODELS: List[DownloadableModelInfo] = [
        DownloadableModelInfo(
            name="Qwen3.5-35B-A3B (4-bit)",
            repository="mlx-community/Qwen3.5-35B-A3B-4bit",
            filename="mlx-community/Qwen3.5-35B-A3B-4bit",
        ),
    ]

    _DOWNLOADABLE_EMBEDDING_MODELS: List[DownloadableModelInfo] = [
        DownloadableModelInfo(
            name="Qwen3-Embedding-4B (4-bit DWQ)",
            repository="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            filename="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        ),
    ]

    _instance: ClassVar["ModelService | None"] = None

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    def _models_dir_for(model_type: str) -> Path:
        if model_type == "embedding":
            return ModelService._EMBEDDING_MODELS_DIR
        return ModelService._CHAT_MODELS_DIR


    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _human_readable_size(num_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0  # type: ignore[assignment]
        return f"{num_bytes:.1f} PB"


    def _dir_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


    def _scan_models(base_dir: Path) -> List[ModelInfo]:
        """
        Walk *base_dir* and return a ``ModelInfo`` entry for every subdirectory
        that contains a ``config.json`` file (i.e. a valid MLX model).
        """
        results: List[ModelInfo] = []
        if not base_dir.exists():
            return results

        for config_file in base_dir.rglob("config.json"):
            model_dir = config_file.parent
            try:
                rel_path = model_dir.relative_to(base_dir)
            except ValueError:
                continue

            rel_str = str(rel_path).replace(os.sep, "/")
            is_downloading = rel_str in _downloading

            results.append(ModelInfo(
                name=model_dir.name,
                path=rel_str,
                size="Downloading..." if is_downloading else ModelService._human_readable_size(ModelService._dir_size(model_dir)),
                status="downloading" if is_downloading else "ready_to_use",
            ))

        return results


    async def _download_background(repository: str, model_type: str) -> None:
        """Background coroutine that downloads a Hugging Face model via snapshot_download."""
        from huggingface_hub import snapshot_download  # type: ignore[import]

        base_dir = ModelService._models_dir_for(model_type)
        parts = repository.split("/", 1)
        org, repo = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        local_dir = base_dir / org / repo if org else base_dir / repo

        try:
            logger.info(f"Starting background download ({model_type}): {repository} → {local_dir}")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(repo_id=repository, local_dir=str(local_dir)),
            )
            logger.info(f"Download complete: {repository}")
        except Exception as exc:
            logger.error(f"Download failed for {repository}: {exc}")
        finally:
            _downloading.discard(repository)


    def list_chat_models(self) -> List[ModelInfo]:
        return ModelService._scan_models(ModelService._CHAT_MODELS_DIR)

    def list_embedding_models(self) -> List[ModelInfo]:
        return ModelService._scan_models(ModelService._EMBEDDING_MODELS_DIR)

    def list_downloadable_chat_models(self) -> List[DownloadableModelInfo]:
        return ModelService._DOWNLOADABLE_CHAT_MODELS

    def list_downloadable_embedding_models(self) -> List[DownloadableModelInfo]:
        return ModelService._DOWNLOADABLE_EMBEDDING_MODELS

    async def download_model(self, repository: str) -> ModelDownloadResponse:
        chat_repos = {m.repository for m in ModelService._DOWNLOADABLE_CHAT_MODELS}
        embedding_repos = {m.repository for m in ModelService._DOWNLOADABLE_EMBEDDING_MODELS}
        all_repos = chat_repos | embedding_repos
        if repository not in all_repos:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail=f"Repository '{repository}' is not in the allowed list",
            )

        model_type = "embedding" if repository in embedding_repos else "chat"
        parts = repository.split("/", 1)
        org, repo = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        rel_path = f"{org}/{repo}" if org else repo

        if repository in _downloading:
            return ModelDownloadResponse(
                repository=repository,
                status="downloading",
                path=rel_path,
                message="Model is already being downloaded",
            )

        _downloading.add(repository)
        asyncio.create_task(ModelService._download_background(repository, model_type))

        return ModelDownloadResponse(
            repository=repository,
            status="downloading",
            path=f"{model_type}/{rel_path}",
            message="Model download started in background",
        )

    def load_model(self, model_path: str, model_type: str) -> ModelLoadResponse:
        resolved = ModelService._models_dir_for(model_type) / model_path.replace("/", os.sep)
        if not resolved.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Model '{model_path}' not found in models/{model_type}/")

        path_str = str(resolved)

        if self._llm.is_model_loaded(path_str, model_type):
            return ModelLoadResponse(
                model=model_path,
                model_type=model_type,
                status="already_loaded",
                message="Model is already loaded in memory",
                model_path=path_str,
            )

        try:
            self._llm.load_model(path_str, model_type)
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Error loading model: {exc}")

        return ModelLoadResponse(
            model=model_path,
            model_type=model_type,
            status="loaded",
            message="Model loaded successfully into memory",
            model_path=path_str,
        )

    def unload_model(self, model_path: str, model_type: str) -> ModelUnloadResponse:
        resolved = ModelService._models_dir_for(model_type) / model_path.replace("/", os.sep)
        if not resolved.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Model '{model_path}' not found in models/{model_type}/")

        path_str = str(resolved)
        was_unloaded = self._llm.unload_model(path_str, model_type)

        if not was_unloaded:
            return ModelUnloadResponse(
                model_path=model_path,
                model_type=model_type,
                status="not_loaded",
                message="Model was not loaded in memory",
            )

        return ModelUnloadResponse(
            model_path=model_path,
            model_type=model_type,
            status="unloaded",
            message="Model unloaded from memory and RAM freed",
        )

    def list_loaded_models(self) -> List[LoadedModelInfo]:
        """
        Return API-level info for every model that is loaded or loading.

        Delegates to :meth:`LLMService.list_loaded_models` and converts the
        result to :class:`LoadedModelInfo` response objects.
        ``loaded=True`` means fully in memory; ``loaded=False`` means a load
        is currently in progress.
        """
        return [
            LoadedModelInfo(
                model_name=r["model_name"],
                model_path=r["model_path"],
                model_type=r["model_type"],
                loaded=not r["loading"],
            )
            for r in self._llm.list_loaded_models()
        ]


# Singleton — created on first call to get_model_service()


def get_model_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> "ModelService":
    """FastAPI dependency that provides the :class:`ModelService` singleton."""
    if ModelService._instance is None:
        ModelService._instance = ModelService(llm_service=llm_service)
    return ModelService._instance
