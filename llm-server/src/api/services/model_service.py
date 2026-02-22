"""
Model management service - business logic for listing and downloading models
"""
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import sys

# Custom tqdm class to force progress bar in non-interactive environments (e.g. logs)
class TqdmProgress(tqdm):
    def __init__(self, *args, **kwargs):
        # Force a large interval to reduce log frequency but ensure updates happen
        kwargs.setdefault("mininterval", 2.0)
        # Force output to stdout
        kwargs.setdefault("file", sys.stdout)
        # Force ascii to avoid unicode issues in logs
        kwargs.setdefault("ascii", True)
        # Avoid trying to determine column width which can fail in k8s
        kwargs.setdefault("ncols", 80)
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        # Allow standard update to handle logic
        super().update(n)
        # Force flush to ensure logs appear immediately in K8s
        if hasattr(self.fp, 'flush'):
            self.fp.flush()

class ModelService:
    """Service for handling model listing and downloading (Singleton)"""
    
    _instance: Optional['ModelService'] = None
    
    # Mapping of repository to specific GGUF files to download
    DOWNLOADABLE_CHAT_MODELS = {
        "unsloth/Qwen3-4B-Instruct-2507-GGUF": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF": "Qwen3-30B-A3B-Instruct-2507-Q3_K_L.gguf"
    }
    
    DOWNLOADABLE_EMBEDDING_MODELS = {
        "Qwen/Qwen3-Embedding-4B-GGUF": "Qwen3-Embedding-4B-Q4_K_M.gguf",
        "Qwen/Qwen3-Embedding-8B-GGUF": "Qwen3-Embedding-8B-Q4_K_M.gguf"
    }
    
    # Track active downloads: {filename: {"status": str, "repository": str}}
    _active_downloads: Dict[str, Dict] = {}
    _downloads_lock: threading.Lock = threading.Lock()
    
    def __new__(cls, models_dir: str = "models"):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model service
        
        Args:
            models_dir: Directory where models are stored
        """
        if self._initialized:
            return
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.chat_models_dir = self.models_dir / "chat"
        self.embed_models_dir = self.models_dir / "embed"
        self.chat_models_dir.mkdir(exist_ok=True)
        self.embed_models_dir.mkdir(exist_ok=True)
        self._initialized = True
    
    def list_available_chat_models(self) -> List[Dict[str, str]]:
        """
        List all available chat models in the models/chat directory
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        if not self.chat_models_dir.exists():
            return models
        
        # Look for .gguf files in the models/chat directory
        for model_file in self.chat_models_dir.rglob("*.gguf"):
            models.append({
                "name": model_file.stem,
                "path": str(model_file.relative_to(self.models_dir)),
                "size": self._format_size(model_file.stat().st_size),
                "status": "ready_to_use"
            })
        
        # Add models that are currently downloading
        with self._downloads_lock:
            active = dict(self._active_downloads)
        for filename, download_info in active.items():
            if download_info.get("model_type") == "chat":
                # Check if this model is not already in the list (fully downloaded)
                if not any(m["name"] == Path(filename).stem for m in models):
                    models.append({
                        "name": Path(filename).stem,
                        "path": filename,
                        "size": "Downloading...",
                        "status": "downloading"
                    })
        
        return models
    
    def list_available_embedding_models(self) -> List[Dict[str, str]]:
        """
        List all available embedding models in the models/embed directory
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        if not self.embed_models_dir.exists():
            return models
        
        # Look for .gguf files in the models/embed directory
        for model_file in self.embed_models_dir.rglob("*.gguf"):
            models.append({
                "name": model_file.stem,
                "path": str(model_file.relative_to(self.models_dir)),
                "size": self._format_size(model_file.stat().st_size),
                "status": "ready_to_use"
            })
        
        # Add models that are currently downloading
        with self._downloads_lock:
            active = dict(self._active_downloads)
        for filename, download_info in active.items():
            if download_info.get("model_type") == "embedding":
                # Check if this model is not already in the list (fully downloaded)
                if not any(m["name"] == Path(filename).stem for m in models):
                    models.append({
                        "name": Path(filename).stem,
                        "path": filename,
                        "size": "Downloading...",
                        "status": "downloading"
                    })
        
        return models
    
    def chat_model_exists(self, model_name: str) -> tuple[bool, str]:
        """
        Check if a chat model exists in models/chat/
        
        Args:
            model_name: Name of the model (stem without .gguf extension)
        
        Returns:
            Tuple of (exists: bool, full_path: str)
        """
        if not self.chat_models_dir.exists():
            return False, ""
        
        for model_file in self.chat_models_dir.rglob("*.gguf"):
            if model_file.stem == model_name:
                return True, str(model_file)
        
        return False, ""
    
    def embedding_model_exists(self, model_name: str) -> tuple[bool, str]:
        """
        Check if an embedding model exists in models/embed/
        
        Args:
            model_name: Name of the model (stem without .gguf extension)
        
        Returns:
            Tuple of (exists: bool, full_path: str)
        """
        if not self.embed_models_dir.exists():
            return False, ""
        
        for model_file in self.embed_models_dir.rglob("*.gguf"):
            if model_file.stem == model_name:
                return True, str(model_file)
        
        return False, ""
    
    def list_downloadable_chat_models(self) -> List[Dict[str, str]]:
        """
        List all downloadable chat models
        
        Returns:
            List of dictionaries with model information
        """
        return [
            {
                "name": repo,
                "repository": repo,
                "filename": filename
            }
            for repo, filename in self.DOWNLOADABLE_CHAT_MODELS.items()
        ]
    
    def list_downloadable_embedding_models(self) -> List[Dict[str, str]]:
        """
        List all downloadable embedding models
        
        Returns:
            List of dictionaries with model information
        """
        return [
            {
                "name": repo,
                "repository": repo,
                "filename": filename
            }
            for repo, filename in self.DOWNLOADABLE_EMBEDDING_MODELS.items()
        ]
    
    def validate_repository(self, repository: str) -> None:
        """
        Validate that repository is in the allowed list
        
        Args:
            repository: Hugging Face repository (e.g., "Qwen/Qwen3-8B")
        
        Raises:
            ValueError: If repository is not in the allowed list
        """
        all_allowed = {**self.DOWNLOADABLE_CHAT_MODELS, **self.DOWNLOADABLE_EMBEDDING_MODELS}
        if repository not in all_allowed:
            raise ValueError(
                f"Repository '{repository}' is not in the list of downloadable models. "
                f"Allowed models: {', '.join(all_allowed.keys())}"
            )
    
    async def download_model_async(self, repository: str) -> None:
        """
        Download a model from Hugging Face in the background.
        Chat models are saved to models/chat/, embedding models to models/embed/.
        
        Args:
            repository: Hugging Face repository (e.g., "Qwen/Qwen3-8B")
        """
        # Determine model type and target directory
        if repository in self.DOWNLOADABLE_CHAT_MODELS:
            model_type = "chat"
            target_dir = self.chat_models_dir
            filename = self.DOWNLOADABLE_CHAT_MODELS[repository]
        elif repository in self.DOWNLOADABLE_EMBEDDING_MODELS:
            model_type = "embedding"
            target_dir = self.embed_models_dir
            filename = self.DOWNLOADABLE_EMBEDDING_MODELS[repository]
        else:
            raise ValueError(f"No filename configured for repository {repository}")

        def _download():
            try:
                # Download the specific GGUF file to the appropriate subdirectory
                downloaded_path = hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    tqdm_class=TqdmProgress  # Force progress bar in logs
                )
                
                # Download complete - remove from active downloads
                with self._downloads_lock:
                    self._active_downloads.pop(filename, None)
                
                print(f"Model {repository}/{filename} downloaded successfully to {downloaded_path}")
            except Exception as e:
                # Remove from active downloads on error
                with self._downloads_lock:
                    self._active_downloads.pop(filename, None)
                print(f"Failed to download model '{repository}': {str(e)}")

        # Register the download entry BEFORE dispatching to the thread pool.
        # If the worker thread does this, there is a window between run_in_executor()
        # returning and the thread actually running where _active_downloads is empty,
        # causing the list endpoints to transiently return [].
        with self._downloads_lock:
            self._active_downloads[filename] = {
                "status": "downloading",
                "repository": repository,
                "model_type": model_type
            }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def get_model_service() -> ModelService:
    """
    FastAPI dependency to get the singleton ModelService instance
    
    Returns:
        ModelService singleton instance
    """
    return ModelService()
