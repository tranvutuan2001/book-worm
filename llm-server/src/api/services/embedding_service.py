"""
Embedding service - business logic layer for text embeddings
"""
from typing import List, Union
from llama_cpp import Llama


class EmbeddingService:
    """Service for handling text embedding logic"""
    
    @staticmethod
    def generate_embeddings(
        model: Llama,
        texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts.
        
        Args:
            model: Llama model instance configured for embeddings
            texts: Single text string or list of texts to embed
        
        Returns:
            List of embedding vectors (list of floats)
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings for each text
        embeddings = []
        for text in texts:
            # llama-cpp-python's embed method returns a list of floats
            embedding = model.embed(text)
            embeddings.append(embedding)
        
        return embeddings
    
    @staticmethod
    def count_tokens(model: Llama, texts: Union[str, List[str]]) -> int:
        """
        Count the total number of tokens in the input texts.
        
        Args:
            model: Llama model instance
            texts: Single text string or list of texts
        
        Returns:
            Total token count
        """
        if isinstance(texts, str):
            texts = [texts]
        
        total_tokens = 0
        for text in texts:
            tokens = model.tokenize(text.encode('utf-8'))
            total_tokens += len(tokens)
        
        return total_tokens
