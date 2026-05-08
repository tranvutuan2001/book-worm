from typing import Any
import mlx_lm
from mlx_lm.sample_utils import make_logits_processors
from app.domain.value_objects.message import Message, ToolCall, ToolCallFunction
from app.domain.value_objects.message_role import MessageRole
from app.domain.protocols.llm_provider import LLMProvider
from app.domain.exceptions.llm_exception import LLMGenerationException
from app.infrastructure.mlx_provider.mlx_model import MLXModel

import mlx.core as mx
import numpy as np

class _XGrammarLogitsProcessor:
    def __init__(self, matcher, vocab_size: int, eos_token_id: int):
        import xgrammar as xgr
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.bitmask_np = self.bitmask.numpy()
        self.last_token_idx = None
        
    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        input_ids_list = input_ids.tolist()
        
        if self.last_token_idx is None:
            self.last_token_idx = len(input_ids_list)
        
        while self.last_token_idx < len(input_ids_list):
            token = input_ids_list[self.last_token_idx]
            if not self.matcher.is_terminated():
                try:
                    self.matcher.accept_token(token)
                except Exception:
                    pass
            self.last_token_idx += 1
            
        if self.matcher.is_terminated():
            # If XGrammar is done, force the model to generate the EOS token.
            mask = mx.zeros(logits.shape, dtype=mx.bool_)
            if self.eos_token_id is not None and self.eos_token_id < logits.shape[-1]:
                mask[..., self.eos_token_id] = True
            return mx.where(mask, logits, mx.array([-float('inf')], dtype=logits.dtype))
            
        self.matcher.fill_next_token_bitmask(self.bitmask)
        
        uint8_view = self.bitmask_np.view(np.uint8)
        bits = np.unpackbits(uint8_view, bitorder='little')[:self.vocab_size]
        
        mask = mx.array(bits, dtype=mx.bool_)
        
        # Apply mask: where bit is 0 (rejected), set to -inf
        return mx.where(mask, logits, mx.array([-float('inf')], dtype=logits.dtype))

class MLXLLMProvider(LLMProvider):
    
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

            if response_format and response_format.get("type") == "json_schema":
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    import xgrammar as xgr
                    
                    if hasattr(self.mlx_model.tokenizer, "vocab_size"):
                        vocab_size = self.mlx_model.tokenizer.vocab_size
                    else:
                        vocab_size = len(self.mlx_model.tokenizer)
                        
                    hf_tokenizer = getattr(self.mlx_model.tokenizer, "_tokenizer", self.mlx_model.tokenizer)
                    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                        hf_tokenizer, 
                        vocab_size=vocab_size
                    )
                    compiler = xgr.GrammarCompiler(tokenizer_info)
                    grammar = compiler.compile_json_schema(schema)
                    matcher = xgr.GrammarMatcher(grammar)
                    
                    eos_token_id = getattr(self.mlx_model.tokenizer, "eos_token_id", None)
                    processor = _XGrammarLogitsProcessor(matcher, vocab_size, eos_token_id)
                    logits_processors.append(processor)

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
