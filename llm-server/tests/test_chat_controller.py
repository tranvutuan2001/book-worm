"""
Test suite for chat completion controller and service.

Tests cover:
- Tool/function calling
- Multi-turn conversations
- Text-only responses
- Message format conversion
- Error handling
"""

import pytest
import json
from typing import Dict, Any
from unittest.mock import MagicMock

from src.api.controllers.chat_controller import create_chat_completion
from src.api.services.chat_service import ChatCompletionService
from src.api.schemas import (
    ChatCompletionRequest,
    Message,
    Tool,
    FunctionDefinition,
    ToolCall,
    ChatCompletionResponse,
    LlamaChatCompletionResponse
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_weather_tool() -> Tool:
    """Create a weather tool definition for testing"""
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
                        "description": "The city and country, e.g. Tokyo, Japan"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        )
    )


def create_calculator_tool() -> Tool:
    """Create a calculator tool definition for testing"""
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
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        )
    )


def create_search_tool() -> Tool:
    """Create a search tool definition for testing"""
    return Tool(
        type="function",
        function=FunctionDefinition(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    )


def mock_llama_response_with_tool_call(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Create a mock llama-cpp-python response with tool calls"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1707782400,
        "model": "qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        }
    }


def mock_llama_response_text_only(text: str) -> Dict[str, Any]:
    """Create a mock llama-cpp-python response with text only"""
    return {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1707782400,
        "model": "qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        }
    }


def validate_tool_call_structure(tool_call: ToolCall) -> bool:
    """Validate that a tool call has the correct structure"""
    if not tool_call:
        return False
    
    if not tool_call.id or not isinstance(tool_call.id, str):
        return False
    
    if tool_call.type != "function":
        return False
    
    if not tool_call.function or not isinstance(tool_call.function, dict):
        return False
    
    if "name" not in tool_call.function or "arguments" not in tool_call.function:
        return False
    
    # Validate arguments is valid JSON string
    try:
        json.loads(tool_call.function["arguments"])
    except (json.JSONDecodeError, TypeError):
        return False
    
    return True


# ============================================================================
# Service Layer Tests
# ============================================================================

class TestChatCompletionService:
    """Test the ChatCompletionService with mocked llama-cpp-python"""
    
    def test_parse_response_with_tool_call(self):
        """Test parsing llama-cpp-python response with tool calls"""
        service = ChatCompletionService()
        
        mock_response = mock_llama_response_with_tool_call(
            "get_weather",
            {"location": "Tokyo, Japan", "unit": "celsius"}
        )
        
        result = service.parse_model_response(LlamaChatCompletionResponse(**mock_response))
        
        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test123"
        assert len(result.choices) == 1
        
        choice = result.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.tool_calls is not None
        assert len(choice.tool_calls) == 1
        
        tool_call = choice.tool_calls[0]
        assert validate_tool_call_structure(tool_call)
        assert tool_call.function["name"] == "get_weather"
        
        arguments = json.loads(tool_call.function["arguments"])
        assert arguments["location"] == "Tokyo, Japan"
        assert arguments["unit"] == "celsius"
        
        print("✓ Service parses tool call response correctly")
    
    def test_parse_response_text_only(self):
        """Test parsing llama-cpp-python response without tool calls"""
        service = ChatCompletionService()
        
        mock_response = mock_llama_response_text_only(
            "Python is a versatile programming language."
        )
        
        result = service.parse_model_response(LlamaChatCompletionResponse(**mock_response))
        
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].tool_calls is None
        assert "Python" in result.choices[0].message.content
        
        print("✓ Service parses text-only response correctly")
    
    def test_generate_completion_converts_messages(self):
        """Test that generate_completion properly converts messages to dict format"""
        service = ChatCompletionService()
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_text_only("Hello!")
        
        messages = [
            Message(role="user", content="Hi there")
        ]
        
        response = service.generate_completion(
            model=mock_model,
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            tools=None
        )
        
        mock_model.create_chat_completion.assert_called_once()
        call_args = mock_model.create_chat_completion.call_args
        
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hi there"}]
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["tools"] is None
        
        print("✓ Service converts messages correctly")
    
    def test_generate_completion_with_tools(self):
        """Test that generate_completion properly converts tools"""
        service = ChatCompletionService()
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_with_tool_call(
            "get_weather", {"location": "Paris"}
        )
        
        messages = [Message(role="user", content="Weather?")]
        tools = [create_weather_tool()]
        
        response = service.generate_completion(
            model=mock_model,
            messages=messages,
            temperature=0.5,
            max_tokens=150,
            tools=tools
        )
        
        call_args = mock_model.create_chat_completion.call_args
        tools_arg = call_args.kwargs["tools"]
        
        assert tools_arg is not None
        assert len(tools_arg) == 1
        assert tools_arg[0]["type"] == "function"
        assert tools_arg[0]["function"]["name"] == "get_weather"
        assert "description" in tools_arg[0]["function"]
        assert "parameters" in tools_arg[0]["function"]
        
        print("✓ Service converts tools correctly")


# ============================================================================
# Controller Tests
# ============================================================================

class TestChatController:
    """Test chat completion controller endpoint"""
    
    @pytest.mark.asyncio
    async def test_weather_query_with_tool_call(self):
        """Test weather query that triggers tool call"""
        request = ChatCompletionRequest(
            messages=[
                Message(role="user", content="What's the weather in Paris?")
            ],
            tools=[create_weather_tool()],
            temperature=0.7,
            max_tokens=150
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_with_tool_call(
            "get_weather",
            {"location": "Paris, France", "unit": "celsius"}
        )
        
        response = await create_chat_completion(request, mock_model)
        
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1
        
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.tool_calls is not None
        assert len(choice.tool_calls) == 1
        
        tool_call = choice.tool_calls[0]
        assert tool_call.function["name"] == "get_weather"
        
        arguments = json.loads(tool_call.function["arguments"])
        assert "location" in arguments
        assert "Paris" in arguments["location"]
        
        print("✓ Controller handles weather query with tool call")
    
    @pytest.mark.asyncio
    async def test_calculator_query(self):
        """Test calculator query"""
        request = ChatCompletionRequest(
            messages=[
                Message(role="user", content="What is 25 plus 17?")
            ],
            tools=[create_calculator_tool()],
            temperature=0.1,
            max_tokens=100
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_with_tool_call(
            "calculate",
            {"operation": "add", "a": 25, "b": 17}
        )
        
        response = await create_chat_completion(request, mock_model)
        
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        
        tool_call = choice.tool_calls[0]
        assert tool_call.function["name"] == "calculate"
        
        arguments = json.loads(tool_call.function["arguments"])
        assert arguments["operation"] == "add"
        assert arguments["a"] == 25
        assert arguments["b"] == 17
        
        print("✓ Controller handles calculator query")
    
    @pytest.mark.asyncio
    async def test_general_question_no_tools(self):
        """Test general question without tool usage"""
        request = ChatCompletionRequest(
            messages=[
                Message(role="user", content="What is Python?")
            ],
            tools=[create_weather_tool(), create_calculator_tool()],
            max_tokens=150
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_text_only(
            "Python is a versatile programming language created by Guido van Rossum."
        )
        
        response = await create_chat_completion(request, mock_model)
        
        choice = response.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.tool_calls is None
        assert "Python" in choice.message.content
        
        print("✓ Controller handles general question without tools")
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with tool usage"""
        request = ChatCompletionRequest(
            messages=[
                Message(role="user", content="What's the weather in London?")
            ],
            tools=[create_weather_tool()],
            max_tokens=150
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_with_tool_call(
            "get_weather",
            {"location": "London, UK", "unit": "celsius"}
        )
        
        response = await create_chat_completion(request, mock_model)
        assert response.choices[0].finish_reason == "tool_calls"
        
        # Second turn: add tool result
        request.messages.extend([
            Message(role="assistant", content=""),
            Message(
                role="tool",
                name="get_weather",
                content='{"temperature": 15, "condition": "partly cloudy"}',
                tool_call_id="call_123"
            )
        ])
        
        mock_model.create_chat_completion.return_value = mock_llama_response_text_only(
            "The weather in London is currently 15°C and partly cloudy."
        )
        
        response = await create_chat_completion(request, mock_model)
        
        assert response.choices[0].finish_reason == "stop"
        assert "15" in response.choices[0].message.content
        
        print("✓ Controller handles multi-turn conversation")
    
    @pytest.mark.asyncio
    async def test_multiple_tools_available(self):
        """Test request with multiple tools available"""
        request = ChatCompletionRequest(
            messages=[
                Message(role="user", content="Search for the weather in Tokyo")
            ],
            tools=[
                create_weather_tool(),
                create_calculator_tool(),
                create_search_tool()
            ],
            max_tokens=200
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = mock_llama_response_with_tool_call(
            "get_weather",
            {"location": "Tokyo, Japan"}
        )
        
        response = await create_chat_completion(request, mock_model)
        
        call_args = mock_model.create_chat_completion.call_args
        assert len(call_args.kwargs["tools"]) == 3
        
        tool_call = response.choices[0].tool_calls[0]
        assert tool_call.function["name"] == "get_weather"
        
        print("✓ Controller handles multiple tools")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestChatErrorHandling:
    """Test error handling in chat service and controller"""
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self):
        """Test that model errors are properly handled"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="Hello")],
            max_tokens=100
        )
        
        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = Exception("Model error")
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(request, mock_model)
        
        assert exc_info.value.status_code == 500
        assert "Model error" in str(exc_info.value.detail)
        
        print("✓ Controller handles model errors")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "="*70)
    print("Running Chat Controller Tests")
    print("="*70 + "\n")
    
    # Test service layer
    print("--- Service Layer Tests ---")
    service_tests = TestChatCompletionService()
    service_tests.test_parse_response_with_tool_call()
    service_tests.test_parse_response_text_only()
    service_tests.test_generate_completion_converts_messages()
    service_tests.test_generate_completion_with_tools()
    
    # Test controller
    print("\n--- Controller Tests ---")
    controller_tests = TestChatController()
    asyncio.run(controller_tests.test_weather_query_with_tool_call())
    asyncio.run(controller_tests.test_calculator_query())
    asyncio.run(controller_tests.test_general_question_no_tools())
    asyncio.run(controller_tests.test_multi_turn_conversation())
    asyncio.run(controller_tests.test_multiple_tools_available())
    
    # Test error handling
    print("\n--- Error Handling Tests ---")
    error_tests = TestChatErrorHandling()
    asyncio.run(error_tests.test_model_error_handling())
    
    print("\n" + "="*70)
    print("All Chat Controller Tests Passed! ✓")
    print("="*70 + "\n")
