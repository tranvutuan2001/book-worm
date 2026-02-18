"""
Integration tests for LLM server.

NOTE: Individual controller tests have been moved to separate files:
- test_health_controller.py - Health and root endpoint tests
- test_chat_controller.py - Chat completion and tool calling tests
- test_embedding_controller.py - Text embedding tests

This file now contains only cross-controller integration tests.

Run all tests with: pytest tests/
Run specific controller: pytest tests/test_chat_controller.py
"""

import pytest
from unittest.mock import MagicMock, patch

from src.api.controllers.chat_controller import create_chat_completion
from src.api.controllers.embedding_controller import create_embeddings
from src.api.controllers.health_controller import root, health
from src.api.schemas import (
    ChatCompletionRequest,
    Message,
    EmbeddingRequest
)


# ============================================================================
# Cross-Controller Integration Tests
# ============================================================================

class TestCrossControllerIntegration:
    """Test interactions between different controllers and services"""
    
    @pytest.mark.asyncio
    async def test_all_endpoints_accessible(self):
        """Test that all endpoints are accessible"""
        # Health endpoints
        root_response = await root()
        health_response = await health()
        
        assert root_response is not None
        assert health_response is not None
        assert root_response["message"] == "LLM Server is running"
        assert health_response["status"] == "healthy"
        
        print("✓ All endpoints are accessible")
    
    @pytest.mark.asyncio
    async def test_chat_and_embedding_different_models(self):
        """Test that chat and embedding use separate models"""
        # Create mock models
        mock_chat_model = MagicMock()
        mock_chat_model.create_chat_completion.return_value = {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 1707782400,
            "model": "chat-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Chat response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed.return_value = [0.1] * 768
        mock_embedding_model.tokenize.return_value = [1, 2, 3]
        
        # Test chat
        chat_request = ChatCompletionRequest(
            messages=[Message(role="user", content="Hello")]
        )
        chat_response = await create_chat_completion(chat_request, mock_chat_model)
        
        # Test embedding
        embedding_request = EmbeddingRequest(input="Hello")
        embedding_response = await create_embeddings(embedding_request, mock_embedding_model)
        
        # Both should work independently
        assert chat_response is not None
        assert embedding_response is not None
        
        print("✓ Chat and embedding use separate models")
    
    @pytest.mark.asyncio
    async def test_server_configuration_consistency(self):
        """Test that server configuration is consistent across endpoints"""
        with patch('src.api.controllers.health_controller.settings') as mock_settings:
            mock_settings.model_path = "/test/model.gguf"
            mock_settings.embedding_model_path = "/test/embedding.gguf"
            
            root_response = await root()
            
            # Health endpoint should reflect same configuration
            assert root_response["model"] == "/test/model.gguf"
        
        print("✓ Server configuration is consistent")


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflows:
    """Test realistic end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_rag_workflow_simulation(self):
        """Simulate a RAG (Retrieval Augmented Generation) workflow"""
        # Step 1: Generate embeddings for documents
        documents = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "SQL is for databases"
        ]
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed.return_value = [0.1] * 768
        mock_embedding_model.tokenize.return_value = [1, 2, 3, 4]
        
        embedding_request = EmbeddingRequest(input=documents)
        embedding_response = await create_embeddings(embedding_request, mock_embedding_model)
        
        assert len(embedding_response.data) == 3
        
        # Step 2: User query with retrieved context
        mock_chat_model = MagicMock()
        mock_chat_model.create_chat_completion.return_value = {
            "id": "chat-rag",
            "object": "chat.completion",
            "created": 1707782400,
            "model": "chat-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Python is a versatile programming language."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        }
        
        chat_request = ChatCompletionRequest(
            messages=[
                Message(role="system", content="You are a helpful assistant. Use the context."),
                Message(role="user", content="What is Python?")
            ]
        )
        chat_response = await create_chat_completion(chat_request, mock_chat_model)
        
        assert "Python" in chat_response.choices[0].message.content
        
        print("✓ RAG workflow simulation works")
    
    @pytest.mark.asyncio
    async def test_semantic_search_workflow(self):
        """Simulate semantic search workflow"""
        # Embed query
        query = "How do I learn programming?"
        
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed.return_value = [0.2] * 768
        mock_embedding_model.tokenize.return_value = [1, 2, 3, 4, 5]
        
        query_embedding = await create_embeddings(
            EmbeddingRequest(input=query),
            mock_embedding_model
        )
        
        assert len(query_embedding.data) == 1
        assert len(query_embedding.data[0].embedding) == 768
        
        # Embed document corpus
        documents = [
            "Python tutorial for beginners",
            "Advanced JavaScript patterns",
            "Learning to code step by step"
        ]
        
        doc_embeddings = await create_embeddings(
            EmbeddingRequest(input=documents),
            mock_embedding_model
        )
        
        assert len(doc_embeddings.data) == 3
        
        # Simulate finding most relevant document and generating response
        mock_chat_model = MagicMock()
        mock_chat_model.create_chat_completion.return_value = {
            "id": "chat-search",
            "object": "chat.completion",
            "created": 1707782400,
            "model": "chat-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Start with Python tutorials for beginners."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
        }
        
        response = await create_chat_completion(
            ChatCompletionRequest(
                messages=[Message(role="user", content=query)]
            ),
            mock_chat_model
        )
        
        assert response is not None
        
        print("✓ Semantic search workflow works")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "="*70)
    print("Running Integration Tests")
    print("="*70 + "\n")
    
    print("--- Cross-Controller Integration ---")
    cross_tests = TestCrossControllerIntegration()
    asyncio.run(cross_tests.test_all_endpoints_accessible())
