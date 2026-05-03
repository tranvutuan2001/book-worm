from openai import AsyncOpenAI, OpenAIError
from app.domain.value_objects.message import Message, ToolCall, ToolCallFunction
from app.domain.value_objects.message_role import MessageRole
from app.domain.protocols.llm_provider import LLMProvider
from app.domain.exceptions.llm_exception import LLMGenerationException

class OpenAILLMProvider(LLMProvider):
    """OpenAI-based implementation of the LLMProvider."""
    
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def generate(self, messages: list[Message], max_completion_tokens: int, tools: list[dict[str, object]] | None = None) -> Message:
        try:
            formatted_messages = []
            for m in messages:
                msg_dict = {"role": m.role.value}
                if m.content is not None:
                    msg_dict["content"] = m.content
                if m.tool_calls:
                    msg_dict["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
                if m.tool_call_id:
                    msg_dict["tool_call_id"] = m.tool_call_id
                if m.name:
                    msg_dict["name"] = m.name
                formatted_messages.append(msg_dict)
            
            kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": max_completion_tokens
            }
            if tools:
                kwargs["tools"] = tools

            response = await self.client.chat.completions.create(**kwargs)
            
            response_msg = response.choices[0].message
            
            domain_tool_calls = None
            if response_msg.tool_calls:
                domain_tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    )
                    for tc in response_msg.tool_calls
                ]
            
            return Message(
                role=MessageRole(response_msg.role),
                content=response_msg.content,
                tool_calls=domain_tool_calls
            )
        except OpenAIError as e:
            raise LLMGenerationException(
                message=str(e),
                provider="openai",
                original_error=e
            )
