"""
Test suite for embedding controller and service.

Tests cover:
- Single text embedding
- Batch text embedding
- Token counting
- Error handling
- Response format validation
"""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from src.api.controllers.embedding_controller import create_embeddings
from src.api.services.embedding_service import EmbeddingService
from src.api.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_embedding(dimension: int = 768) -> List[float]:
    """Create a mock embedding vector"""
    return [0.1 * i for i in range(dimension)]


def create_mock_model(dimension: int = 768) -> MagicMock:
    """Create a mock Llama model for embeddings"""
    mock_model = MagicMock()
    mock_model.embed.return_value = create_mock_embedding(dimension)
    mock_model.tokenize.return_value = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
    return mock_model


# ============================================================================
# Service Layer Tests
# ============================================================================

class TestEmbeddingService:
    """Test the EmbeddingService with mocked llama-cpp-python"""
    
    def test_generate_single_embedding(self):
        """Test generating embedding for single text"""
        service = EmbeddingService()
        mock_model = create_mock_model()
        
        text = "Hello world"
        embeddings = service.generate_embeddings(mock_model, text)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 768
        
        # Verify model.embed was called
        mock_model.embed.assert_called_once_with(text)
        
        print("✓ Service generates single embedding correctly")
    
    def test_generate_batch_embeddings(self):
        """Test generating embeddings for multiple texts"""
        service = EmbeddingService()
        mock_model = create_mock_model()
        
        texts = ["Hello world", "How are you?", "Python is awesome"]
        embeddings = service.generate_embeddings(mock_model, texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 768
        
        # Verify embed was called for each text
        assert mock_model.embed.call_count == 3
        
        print("✓ Service generates batch embeddings correctly")
    
    def test_count_tokens_single_text(self):
        """Test token counting for single text"""
        service = EmbeddingService()
        mock_model = create_mock_model()
        
        text = "Hello world"
        token_count = service.count_tokens(mock_model, text)
        
        assert isinstance(token_count, int)
        assert token_count == 8  # Based on mock
        
        mock_model.tokenize.assert_called_once()
        
        print("✓ Service counts tokens correctly for single text")
    
    def test_count_tokens_batch(self):
        """Test token counting for batch of texts"""
        service = EmbeddingService()
        mock_model = create_mock_model()
        
        texts = ["Hello", "world"]
        token_count = service.count_tokens(mock_model, texts)
        
        assert isinstance(token_count, int)
        assert token_count == 16  # 8 tokens * 2 texts
        
        assert mock_model.tokenize.call_count == 2
        
        print("✓ Service counts tokens correctly for batch")
    
    def test_embedding_normalization(self):
        """Test that string input is normalized to list"""
        service = EmbeddingService()
        mock_model = create_mock_model()
        
        # Pass string, should be converted to list
        embeddings = service.generate_embeddings(mock_model, "single text")
        
        assert len(embeddings) == 1
        
        print("✓ Service normalizes string input to list")


# ============================================================================
# Controller Tests
# ============================================================================

class TestEmbeddingController:
    """Test embedding controller endpoint"""
    
    @pytest.mark.asyncio
    async def test_single_text_embedding(self):
        """Test endpoint with single text input"""
        request = EmbeddingRequest(
            input="The quick brown fox jumps over the lazy dog"
        )
        
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        # Validate response structure
        assert isinstance(response, EmbeddingResponse)
        assert response.object == "list"
        assert len(response.data) == 1
        
        # Validate embedding data
        embedding_data = response.data[0]
        assert embedding_data.object == "embedding"
        assert embedding_data.index == 0
        assert isinstance(embedding_data.embedding, list)
        assert len(embedding_data.embedding) == 768
        
        # Validate usage
        assert response.usage.prompt_tokens == 8
        assert response.usage.total_tokens == 8
        
        print("✓ Controller handles single text embedding")
    
    @pytest.mark.asyncio
    async def test_batch_text_embeddings(self):
        """Test endpoint with batch text input"""
        request = EmbeddingRequest(
            input=["Hello world", "How are you?", "Python is great"]
        )
        
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        # Validate response
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 3
        
        # Check each embedding
        for i, embedding_data in enumerate(response.data):
            assert embedding_data.object == "embedding"
            assert embedding_data.index == i
            assert len(embedding_data.embedding) == 768
        
        # Validate usage (3 texts * 8 tokens each)
        assert response.usage.prompt_tokens == 24
        assert response.usage.total_tokens == 24
        
        print("✓ Controller handles batch embeddings")
    
    @pytest.mark.asyncio
    async def test_encoding_format_float(self):
        """Test with explicit float encoding format"""
        request = EmbeddingRequest(
            input="Test text",
            encoding_format="float"
        )
        
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        assert isinstance(response, EmbeddingResponse)
        assert isinstance(response.data[0].embedding, list)
        assert isinstance(response.data[0].embedding[0], float)
        
        print("✓ Controller handles float encoding format")
    
    @pytest.mark.asyncio
    async def test_encoding_format_base64_error(self):
        """Test that base64 encoding raises error"""
        request = EmbeddingRequest(
            input="Test text",
            encoding_format="base64"
        )
        
        mock_model = create_mock_model()
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_embeddings(request, mock_model)
        
        assert exc_info.value.status_code == 400
        assert "base64" in str(exc_info.value.detail).lower()
        
        print("✓ Controller rejects base64 encoding format")
    
    @pytest.mark.asyncio
    async def test_model_name_in_response(self):
        """Test that model name is included in response"""
        request = EmbeddingRequest(input="Test")
        
        mock_model = create_mock_model()
        
        with patch('src.api.controllers.embedding_controller.settings') as mock_settings:
            mock_settings.embedding_model_path = "/path/to/embedding-model.gguf"
            
            response = await create_embeddings(request, mock_model)
            
            assert response.model == "embedding-model.gguf"
        
        print("✓ Controller includes model name in response")
    
    @pytest.mark.asyncio
    async def test_empty_input_validation(self):
        """Test that empty input is handled"""
        # Empty string
        request = EmbeddingRequest(input="")
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        assert len(response.data) == 1
        
        print("✓ Controller handles empty input")
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batch of texts"""
        texts = [f"Sample text number {i}" for i in range(50)]
        request = EmbeddingRequest(input=texts)
        
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        assert len(response.data) == 50
        assert mock_model.embed.call_count == 50
        
        # Verify all indices are correct
        for i in range(50):
            assert response.data[i].index == i
        
        print("✓ Controller handles large batches")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestEmbeddingErrorHandling:
    """Test error handling in embedding service and controller"""
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self):
        """Test handling of model errors"""
        request = EmbeddingRequest(input="Test text")
        
        mock_model = MagicMock()
        mock_model.embed.side_effect = Exception("Model crashed")
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_embeddings(request, mock_model)
        
        assert exc_info.value.status_code == 500
        assert "error" in str(exc_info.value.detail).lower()
        
        print("✓ Controller handles model errors")
    
    @pytest.mark.asyncio
    async def test_tokenization_error_handling(self):
        """Test handling of tokenization errors"""
        request = EmbeddingRequest(input="Test text")
        
        mock_model = MagicMock()
        mock_model.embed.return_value = create_mock_embedding()
        mock_model.tokenize.side_effect = Exception("Tokenization failed")
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_embeddings(request, mock_model)
        
        assert exc_info.value.status_code == 500
        
        print("✓ Controller handles tokenization errors")


# ============================================================================
# Integration Tests
# ============================================================================

class TestEmbeddingIntegration:
    """Integration tests for complete embedding flow"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_single_embedding(self):
        """Test complete flow for single embedding"""
        request = EmbeddingRequest(
            input="Machine learning is fascinating",
            encoding_format="float"
        )
        
        mock_model = create_mock_model(dimension=1024)
        mock_model.tokenize.return_value = [1, 2, 3, 4, 5]
        
        response = await create_embeddings(request, mock_model)
        
        # Comprehensive validation
        assert isinstance(response, EmbeddingResponse)
        assert response.object == "list"
        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 1024
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5
        
        print("✓ End-to-end single embedding flow works")
    
    @pytest.mark.asyncio
    async def test_end_to_end_batch_embedding(self):
        """Test complete flow for batch embeddings"""
        texts = [
            "Natural language processing",
            "Computer vision",
            "Reinforcement learning"
        ]
        
        request = EmbeddingRequest(input=texts)
        
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        # Validate all embeddings
        assert len(response.data) == 3
        for i, text in enumerate(texts):
            assert response.data[i].index == i
            assert len(response.data[i].embedding) == 768
        
        print("✓ End-to-end batch embedding flow works")
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_use_case(self):
        """Test use case: computing semantic similarity"""
        # Simulate embedding similar sentences
        sentences = [
            "The cat sits on the mat",
            "A feline rests on the rug"
        ]
        
        request = EmbeddingRequest(input=sentences)
        mock_model = create_mock_model()
        
        response = await create_embeddings(request, mock_model)
        
        # Both should have embeddings
        assert len(response.data) == 2
        embedding1 = response.data[0].embedding
        embedding2 = response.data[1].embedding
        
        # Can compute similarity (dot product, cosine, etc.)
        assert len(embedding1) == len(embedding2)
        
        print("✓ Semantic similarity use case works")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "="*70)
    print("Running Embedding Tests")
    print("="*70 + "\n")
    
    # Test service layer
    print("--- Service Layer Tests ---")
    service_tests = TestEmbeddingService()
    service_tests.test_generate_single_embedding()
    service_tests.test_generate_batch_embeddings()
    service_tests.test_count_tokens_single_text()
    service_tests.test_count_tokens_batch()
    service_tests.test_embedding_normalization()
    
    # Test controller
    print("\n--- Controller Tests ---")
    controller_tests = TestEmbeddingController()
    asyncio.run(controller_tests.test_single_text_embedding())
    asyncio.run(controller_tests.test_batch_text_embeddings())
    asyncio.run(controller_tests.test_encoding_format_float())
    asyncio.run(controller_tests.test_encoding_format_base64_error())
    asyncio.run(controller_tests.test_model_name_in_response())
    asyncio.run(controller_tests.test_empty_input_validation())
    asyncio.run(controller_tests.test_large_batch_processing())
    
    # Test error handling
    print("\n--- Error Handling Tests ---")
    error_tests = TestEmbeddingErrorHandling()
    asyncio.run(error_tests.test_model_error_handling())
    asyncio.run(error_tests.test_tokenization_error_handling())
    
    # Test integration
    print("\n--- Integration Tests ---")
    integration_tests = TestEmbeddingIntegration()
    asyncio.run(integration_tests.test_end_to_end_single_embedding())
    asyncio.run(integration_tests.test_end_to_end_batch_embedding())
    asyncio.run(integration_tests.test_semantic_similarity_use_case())
    
    print("\n" + "="*70)
    print("All Embedding Tests Passed! ✓")
    print("="*70 + "\n")
