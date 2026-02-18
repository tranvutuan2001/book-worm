from typing import Optional, Dict
from llama_cpp import Llama
from src.llm_client.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMModelManager:
    """Manager for loading and caching LLM models"""
    
    _instance: Optional['LLMModelManager'] = None
    _models: Dict[str, Llama] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_path: str) -> Llama:
        """
        Get or create a model instance for the given path
        
        Args:
            model_path: Full path to the model file
        
        Returns:
            Llama model instance
        """
        if model_path not in self._models:
            logger.info(f"Loading chat model from {model_path}")
            self._models[model_path] = Llama(
                model_path=model_path,
                n_ctx=settings.n_ctx,
                n_gpu_layers=settings.n_gpu_layers,
                n_threads=settings.n_threads,
                verbose=settings.verbose,
                chat_format="qwen",
            )
            logger.info("Chat model loaded successfully")
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
            
            logger.info(f"Unloaded chat model and freed memory: {model_path}")
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


def load_chat_model(model_path: str) -> Llama:
    """
    Load a chat model by path
    
    Args:
        model_path: Full path to the model file
    
    Returns:
        Llama model instance
    """
    manager = LLMModelManager()
    return manager.get_model(model_path)
