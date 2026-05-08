from typing import Any
import mlx_lm
from mlx_lm.sample_utils import make_logits_processors
from app.domain.value_objects.message import Message, ToolCall, ToolCallFunction
from app.domain.value_objects.message_role import MessageRole
from app.domain.protocols.llm_provider import LLMProvider
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.infrastructure.mlx_provider.mlx_model import MLXModel

class MLXLLMProvider(LLMProvider):
    """MLX-based implementation of the LLMProvider."""
    
    def __init__(self, mlx_model: MLXModel):
        self.mlx_model = mlx_model

    async def generate(
        self, 
        messages: list[Message], 
        max_completion_tokens: int, 
        frequency_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None
    ) -> Message:
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
                
            if hasattr(self.mlx_model.tokenizer, "apply_chat_template"):
                kwargs = {
                    "conversation": formatted_messages,
                    "tokenize": False,
                    "add_generation_prompt": True
                }
                
                if tools:
                    kwargs["tools"] = tools

                prompt = self.mlx_model.tokenizer.apply_chat_template(**kwargs)
            else:
                prompt = "\n".join([f"{m.role.value}: {m.content or ''}" for m in messages])
                prompt += "\nassistant: "

            print("\n******** Tools: \n", tools)
            print("\n******** Prompt: \n", prompt)
            
            logits_processors = []
            if frequency_penalty is not None:
                logits_processors = make_logits_processors(frequency_penalty=frequency_penalty)

            # Running directly to avoid 'no Stream' errors with asyncio.to_thread
            response = mlx_lm.generate(
                model=self.mlx_model.model,
                tokenizer=self.mlx_model.tokenizer,
                prompt=prompt,
                max_tokens=max_completion_tokens,
                verbose=False,
                logits_processors=logits_processors
            )

            print("\n******** Response: \n", response)
            
            from app.infrastructure.mlx_provider.mlx_response_parser import MLXResponseParser
            clean_content, tool_calls_data = MLXResponseParser.parse(
                response, 
                model_path=self.mlx_model.model_path
            )

            print("\n******** Clean Content: \n", clean_content)
            print("\n******** Tool Calls Data: \n", tool_calls_data)
            
            domain_tool_calls = None
            if tool_calls_data:
                domain_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type="function",
                        function=ToolCallFunction(
                            name=tc["name"],
                            arguments=tc["arguments"]
                        )
                    )
                    for tc in tool_calls_data
                ]
                
            return Message(
                role=MessageRole.ASSISTANT,
                content=clean_content,
                tool_calls=domain_tool_calls
            )
        except Exception as e:
            raise LLMGenerationException(
                message=str(e),
                provider="mlx",
                original_error=e
            )
