"""
Chat completion service - business logic layer
"""
from typing import List, Optional
from llama_cpp import Llama

from src.api.schemas import (
    ToolCall, 
    ChatCompletionResponse, 
    Choice, 
    Message, 
    Usage, 
    Tool,
    LlamaChatCompletionResponse
)


class ChatCompletionService:
    """Service for handling chat completion logic"""
    
    @staticmethod
    def generate_completion(
        model: Llama,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Tool]] = None
    ) -> LlamaChatCompletionResponse:
        """
        Generate completion from the model using chat format.
        
        Args:
            model: Llama model instance
            messages: List of OpenAI-format messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions
        
        Returns:
            Raw response dict from llama-cpp-python
        """
        # Convert messages to dict format for llama-cpp-python
        messages_dict = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in messages
        ]
        
        # Prepare tools if provided
        tools_dict = None
        if tools:
            tools_dict = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    }
                }
                for tool in tools
            ]
        
        # Use create_chat_completion for automatic format conversion
        response = model.create_chat_completion(
            messages=messages_dict,
            tools=tools_dict,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Validate and return as typed response
        return LlamaChatCompletionResponse(**response)

    @staticmethod
    def parse_model_response(response: LlamaChatCompletionResponse) -> ChatCompletionResponse:
        """
        Convert llama-cpp-python response to our ChatCompletionResponse schema.
        
        Args:
            response: Typed response from llama-cpp-python
        
        Returns:
            ChatCompletionResponse object
        """
        # Extract the choice from llama-cpp-python response
        choice_data = response.choices[0]
        message_data = choice_data.message
        
        # Convert tool_calls if present
        tool_calls = None
        if message_data.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function={
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                )
                for tc in message_data.tool_calls
            ]
        
        # Build response message
        response_message = Message(
            role=message_data.role,
            content=message_data.content or "",
        )
        
        # Build choice
        choice = Choice(
            index=choice_data.index,
            message=response_message,
            finish_reason=choice_data.finish_reason,
            tool_calls=tool_calls,
        )
        
        # Build usage
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        
        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[choice],
            usage=usage,
        )


