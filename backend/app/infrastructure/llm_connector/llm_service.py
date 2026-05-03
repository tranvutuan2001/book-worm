"""
LLM inference service — powered by a remote Multi-Provider LLM Server.

This module provides a unified interface for chat completions (with tool-calling)
and text embeddings by communicating with an external LLM server.
"""

import asyncio
import json
import logging
import inspect
from typing import Any, Callable

from langfuse.openai import openai

from app.domain.entity.agent import Agent as DomainAgent
from app.domain.entity.message import Message
from app.domain.enum.role import Role
from app.infrastructure.llm_connector.dto.chat_message import ChatMessage
from app.infrastructure.llm_connector.dto.tool_call import ToolCall
from app.infrastructure.llm_connector.dto.tool_call_function import ToolCallFunction
from app.infrastructure.llm_connector.mapper.completion_request_mapper import CompletionRequestMapper

logger = logging.getLogger("app.infra.llm_service")


class LLMService:
    """
    Handles LLM inference via a remote server.

    Responsibilities
    ----------------
    * :meth:`agent_complete_chat` — chat completion with tool-calling support.
    * :meth:`embed_text`          — text embedding.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        # We use Langfuse wrapper to get observability out of the box
        # API key is required by the client but may not be used by the remote server
        self._client = openai.AsyncOpenAI(
            base_url=f"{self._base_url}",
            api_key="none",
        )

    async def agent_complete_chat(
        self,
        message_list: list[Message],
        agent: DomainAgent,
    ) -> str:
        """
        Run a full chat turn with tool-calling support via the remote LLM server.

        Args:
            message_list: Conversation history as ``Message`` objects.
            agent:        Domain agent carrying the system prompt, tools,
                          and model settings.

        Returns:
            The final assistant text response.
        """
        # Map initial request using the mapper
        request = CompletionRequestMapper.map_to_completion_request(message_list, agent)
        messages = request.messages

        # Conversation loop (to handle multiple tool calls if needed)
        for _ in range(10): # Limit to 10 steps to prevent infinite loops
            # Prepare request parameters from the request model and current messages
            kwargs = request.model_dump(exclude_none=True)
            kwargs["messages"] = [m.model_dump(exclude_none=True) for m in messages]

            # Retry loop for the specific HTTP request
            response = None
            for attempt in range(agent.max_retries + 1):
                try:
                    logger.info(f"LLM request step {_+1}, attempt {attempt+1}")
                    response = await self._client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    if attempt == agent.max_retries:
                        logger.error(f"LLM request failed after {agent.max_retries} retries: {e}")
                        raise
                    logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(1) # Basic backoff

            if not response:
                return ""

            choice = response.choices[0]
            content = choice.message.content
            tool_calls = choice.message.tool_calls
            
            logger.info(f"LLM response: content='{content}', tool_calls={tool_calls}")

            if not tool_calls:
                return content or ""
            
            # Handle tool calls
            # Add assistant message to history using the model
            assistant_message = ChatMessage(
                role="assistant",
                content=content,
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        function=ToolCallFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    ) for tc in tool_calls
                ]
            )
            messages.append(assistant_message)
            
            for tool_call in tool_calls:
                await self._execute_tool(tool_call, agent, messages)
            
            # Continue conversation loop with updated messages
            continue

        return ""

    async def _execute_tool(self, tool_call, agent: DomainAgent, messages: list[ChatMessage]):
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        try:
            tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError:
            tool_args = {}
            
        tool_func = next((t for t in agent.tools if t.__name__ == tool_name), None)
        if tool_func:
            from pydantic_ai import RunContext
            ctx = RunContext(deps=None, model=None, usage=None, prompt=None)
            
            try:
                sig = inspect.signature(tool_func)
                # Check if it's async
                if inspect.iscoroutinefunction(tool_func):
                    if "ctx" in sig.parameters:
                        result = await tool_func(ctx, **tool_args)
                    else:
                        result = await tool_func(**tool_args)
                else:
                    if "ctx" in sig.parameters:
                        result = tool_func(ctx, **tool_args)
                    else:
                        result = tool_func(**tool_args)
                
                logger.info(f"Tool {tool_name} result: {result}")
                messages.append(ChatMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    content=str(result)
                ))
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                messages.append(ChatMessage(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    content=f"Error: {str(e)}"
                ))
        else:
            messages.append(ChatMessage(
                role="tool",
                tool_call_id=tool_call.id,
                name=tool_name,
                content=f"Error: Tool {tool_name} not found."
            ))


    async def embed_text(self, text: str, model: str | None = None, *args, **kwargs) -> list[float]:
        """
        Create a text embedding using the remote embedding server.

        Args:
            text: The text to embed.
            model: Optional model name (handled natively by the OpenAI SDK)

        Returns:
            A list of floats representing the embedding vector.
        """
        response = await self._client.embeddings.create(
            input=text,
            model=model or "",
            name="",
            metadata={},
            **kwargs
        )
        return response.data[0].embedding
