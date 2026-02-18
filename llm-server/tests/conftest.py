"""
Shared test fixtures and utilities for all test modules.

This module provides common test fixtures, mock helpers, and utilities
that can be reused across different test files.
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import MagicMock

from src.api.schemas import Tool, FunctionDefinition


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_model():
    """Create a mock LLM model for chat completions"""
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1707782400,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Test response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    return mock_model


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model"""
    mock_model = MagicMock()
    mock_model.embed.return_value = [0.1 * i for i in range(768)]
    mock_model.tokenize.return_value = [1, 2, 3, 4, 5]
    return mock_model


# ============================================================================
# Tool Definition Fixtures
# ============================================================================

@pytest.fixture
def weather_tool() -> Tool:
    """Weather tool definition fixture"""
    return Tool(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        )
    )


@pytest.fixture
def calculator_tool() -> Tool:
    """Calculator tool definition fixture"""
    return Tool(
        type="function",
        function=FunctionDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        )
    )


# ============================================================================
# Response Mock Helpers
# ============================================================================

def create_mock_chat_response(
    content: str = None,
    tool_calls: List[Dict[str, Any]] = None,
    finish_reason: str = "stop"
) -> Dict[str, Any]:
    """
    Create a mock chat completion response.
    
    Args:
        content: Response text content
        tool_calls: List of tool calls
        finish_reason: Completion finish reason
    
    Returns:
        Mock response dictionary
    """
    message = {"role": "assistant"}
    
    if content:
        message["content"] = content
    else:
        message["content"] = None
    
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1707782400,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }


def create_mock_tool_call(
    function_name: str,
    arguments: Dict[str, Any],
    call_id: str = "call_test123"
) -> Dict[str, Any]:
    """
    Create a mock tool call structure.
    
    Args:
        function_name: Name of the function to call
        arguments: Function arguments as dictionary
        call_id: Tool call ID
    
    Returns:
        Tool call dictionary
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments)
        }
    }


def create_mock_embedding_vector(dimension: int = 768) -> List[float]:
    """
    Create a mock embedding vector.
    
    Args:
        dimension: Vector dimension
    
    Returns:
        List of floats representing the embedding
    """
    return [0.1 * i for i in range(dimension)]


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_chat_response_structure(response: Dict[str, Any]) -> bool:
    """
    Validate that a chat response has the correct structure.
    
    Args:
        response: Response dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["id", "object", "created", "model", "choices", "usage"]
    if not all(field in response for field in required_fields):
        return False
    
    if not response["choices"] or len(response["choices"]) == 0:
        return False
    
    choice = response["choices"][0]
    required_choice_fields = ["index", "message", "finish_reason"]
    if not all(field in choice for field in required_choice_fields):
        return False
    
    return True


def validate_embedding_response_structure(response: Dict[str, Any]) -> bool:
    """
    Validate that an embedding response has the correct structure.
    
    Args:
        response: Response dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["object", "data", "model", "usage"]
    if not all(field in response for field in required_fields):
        return False
    
    if not isinstance(response["data"], list) or len(response["data"]) == 0:
        return False
    
    for item in response["data"]:
        if not all(field in item for field in ["object", "embedding", "index"]):
            return False
        if not isinstance(item["embedding"], list):
            return False
    
    return True


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_messages(count: int = 3) -> List[Dict[str, str]]:
    """
    Generate test chat messages.
    
    Args:
        count: Number of messages to generate
    
    Returns:
        List of message dictionaries
    """
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": f"Test message {i + 1}"
        })
    return messages


def generate_test_texts(count: int = 5) -> List[str]:
    """
    Generate test text strings for embeddings.
    
    Args:
        count: Number of texts to generate
    
    Returns:
        List of text strings
    """
    return [f"Test text number {i + 1}" for i in range(count)]


# ============================================================================
# Pytest Configuration
# ============================================================================

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks between tests"""
    yield
    # Cleanup happens here if needed


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
