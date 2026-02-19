"""
Model management service - business logic for listing and downloading models
"""
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download


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
        self._initialized = True
    
    def list_available_chat_models(self) -> List[Dict[str, str]]:
        """
        List all available chat models in the models directory
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        if not self.models_dir.exists():
            return models
        
        # Look for .gguf files in the models directory
        for model_file in self.models_dir.rglob("*.gguf"):
            if "embed" not in model_file.name.lower():  # Exclude embedding models
                models.append({
                    "name": model_file.stem,
                    "path": str(model_file.relative_to(self.models_dir)),
                    "size": self._format_size(model_file.stat().st_size),
                    "status": "ready_to_use"
                })
        
        # Add models that are currently downloading
        for filename, download_info in self._active_downloads.items():
            if "embed" not in filename.lower():
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
        List all available embedding models in the models directory
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        if not self.models_dir.exists():
            return models
        
        # Look for .gguf files with "embed" in the name
        for model_file in self.models_dir.rglob("*.gguf"):
            if "embed" in model_file.name.lower():
                models.append({
                    "name": model_file.stem,
                    "path": str(model_file.relative_to(self.models_dir)),
                    "size": self._format_size(model_file.stat().st_size),
                    "status": "ready_to_use"
                })
        
        # Add models that are currently downloading
        for filename, download_info in self._active_downloads.items():
            if "embed" in filename.lower():
                # Check if this model is not already in the list (fully downloaded)
                if not any(m["name"] == Path(filename).stem for m in models):
                    models.append({
                        "name": Path(filename).stem,
                        "path": filename,
                        "size": "Downloading...",
                        "status": "downloading"
                    })
        
        return models
    
    def model_exists(self, model_name: str, model_type: str = "chat") -> tuple[bool, str]:
        """
        Check if a model exists in the models directory
        
        Args:
            model_name: Name of the model (stem without .gguf extension)
            model_type: Type of model - "chat" or "embedding"
        
        Returns:
            Tuple of (exists: bool, full_path: str)
        """
        if not self.models_dir.exists():
            return False, ""
        
        # Search for the model file
        for model_file in self.models_dir.rglob("*.gguf"):
            if model_file.stem == model_name:
                # Check if it matches the expected type
                is_embedding = "embed" in model_file.name.lower()
                if (model_type == "embedding" and is_embedding) or (model_type == "chat" and not is_embedding):
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
        Download a model from Hugging Face in the background
        
        Args:
            repository: Hugging Face repository (e.g., "Qwen/Qwen3-8B")
        """
        service_instance = self
        
        def _download():
            try:
                # Get the filename for this repository
                all_models = {**service_instance.DOWNLOADABLE_CHAT_MODELS, **service_instance.DOWNLOADABLE_EMBEDDING_MODELS}
                filename = all_models.get(repository)
                
                if not filename:
                    raise ValueError(f"No filename configured for repository {repository}")
                
                # Track download status
                service_instance._active_downloads[filename] = {
                    "status": "downloading",
                    "repository": repository
                }
                
                # Download the specific GGUF file
                downloaded_path = hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    local_dir=str(service_instance.models_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                
                # Download complete - remove from active downloads
                if filename in service_instance._active_downloads:
                    del service_instance._active_downloads[filename]
                
                print(f"Model {repository}/{filename} downloaded successfully to {downloaded_path}")
            except Exception as e:
                # Remove from active downloads on error
                if 'filename' in locals() and filename in service_instance._active_downloads:
                    del service_instance._active_downloads[filename]
                print(f"Failed to download model '{repository}': {str(e)}")
        
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
