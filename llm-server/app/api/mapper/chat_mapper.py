from typing import Any
from app.api.dto.chat_completion_request import ChatCompletionRequest
from app.services.commands.generate_text_command import GenerateTextCommand
from app.domain.value_objects.message import Message, ToolCall, ToolCallFunction
from app.domain.value_objects.message_role import MessageRole

class ChatMapper:
    """Mapper to convert Chat DTOs to Service Commands."""
    
    @staticmethod
    def to_generate_text_command(request: ChatCompletionRequest) -> GenerateTextCommand:
        domain_messages = []
        for msg in request.messages:
            content = ChatMapper._map_content(msg.content) if hasattr(msg, "content") else None
            
            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    ) for tc in msg.tool_calls
                ]
            
            domain_messages.append(Message(
                role=MessageRole(msg.role),
                content=content,
                tool_calls=tool_calls,
                tool_call_id=getattr(msg, "tool_call_id", None),
                name=getattr(msg, "name", None)
            ))
            
        return GenerateTextCommand(
            messages=domain_messages,
            max_completion_tokens=request.max_completion_tokens,
            frequency_penalty=request.frequency_penalty,
            response_format=request.response_format.model_dump() if request.response_format else None,
            tools=[t.model_dump() for t in request.tools] if request.tools else None
        )

    @staticmethod
    def _map_content(content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif hasattr(part, "type") and part.type == "text":
                    text_parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return "".join(text_parts) if text_parts else None
        return str(content)
