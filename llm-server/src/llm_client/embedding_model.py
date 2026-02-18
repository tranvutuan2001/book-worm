from typing import Optional, Dict
from llama_cpp import Llama
from src.llm_client.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """Manager for loading and caching embedding models"""
    
    _instance: Optional['EmbeddingModelManager'] = None
    _models: Dict[str, Llama] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_path: str) -> Llama:
        """
        Get or create an embedding model instance for the given path
        
        Args:
            model_path: Full path to the model file
        
        Returns:
            Llama model instance configured for embeddings
        """
        if model_path not in self._models:
            logger.info(f"Loading embedding model from {model_path}")
            self._models[model_path] = Llama(
                model_path=model_path,
                n_ctx=settings.n_ctx,
                n_gpu_layers=settings.n_gpu_layers,
                n_threads=settings.n_threads,
                verbose=settings.verbose,
                embedding=True,  # Enable embedding mode
            )
            logger.info("Embedding model loaded successfully")
        return self._models[model_path]
    
    def unload_model(self, model_path: str):
        """Unload a specific model from cache and free memory"""
        if model_path in self._models:
            # Delete the model instance to free memory
            model = self._models[model_path]
            del model
            del self._models[model_path]
            
            # Force garbage collection to ensure memory is freed
            import gc
            gc.collect()
            
            logger.info(f"Unloaded embedding model and freed memory: {model_path}")
            return True
        return False
    
    def is_loaded(self, model_path: str) -> bool:
        """Check if a model is currently loaded"""
        return model_path in self._models
    
    def list_loaded_models(self) -> list[str]:
        """List all currently loaded model paths"""
        return list(self._models.keys())
    
    @classmethod
    def reset(cls):
        """Reset the manager instance (useful for testing)"""
        if cls._instance:
            cls._instance._models.clear()
        cls._instance = None


def load_embedding_model(model_path: str) -> Llama:
    """
    Load an embedding model by path
    
    Args:
        model_path: Full path to the model file
    
    Returns:
        Llama model instance configured for embeddings
    """
    manager = EmbeddingModelManager()
    return manager.get_model(model_path)
