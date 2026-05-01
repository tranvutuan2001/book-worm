"""
LLM inference service — powered by a remote Multi-Provider LLM Server.

This module provides a unified interface for chat completions (with tool-calling)
and text embeddings by communicating with an external LLM server.
"""

import asyncio
import json
import logging
from typing import Any, Callable

import httpx

from app.domain.entity.agent import Agent as DomainAgent
from app.domain.entity.message import Message
from app.domain.enums import Role

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
        messages = [
            {"role": "system", "content": agent.system_prompt}
        ]
        for msg in message_list:
            messages.append({"role": msg.role.value, "content": msg.content})

        tools_schema = self._prepare_tools(agent.tools) if agent.tools else None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Conversation loop (to handle multiple tool calls if needed)
            for _ in range(10): # Limit to 10 steps to prevent infinite loops
                payload = {
                    "messages": messages,
                    "max_tokens": agent.model_settings.max_tokens or 1024,
                    "temperature": agent.model_settings.temperature,
                    "tools": tools_schema,
                }

                # Retry loop for the specific HTTP request
                data = None
                for attempt in range(agent.max_retries + 1):
                    try:
                        logger.info(f"LLM request step {_+1}, attempt {attempt+1}")
                        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
                        response = await client.post(f"{self._base_url}/generate", json=payload)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except (httpx.HTTPStatusError, httpx.RequestError) as e:
                        if attempt == agent.max_retries:
                            logger.error(f"LLM request failed after {agent.max_retries} retries: {e}")
                            raise
                        logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1) # Basic backoff

                if not data:
                    return ""

                content = data.get("content")
                tool_calls = data.get("tool_calls")
                
                logger.info(f"LLM response: content='{content}', tool_calls={tool_calls}")

                if not tool_calls:
                    return content or ""
                
                # Handle tool calls
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    
                    tool_func = next((t for t in agent.tools if t.__name__ == tool_name), None)
                    if tool_func:
                        from pydantic_ai import RunContext
                        ctx = RunContext(deps=None, model=None, usage=None, prompt=None)
                        
                        try:
                            # Use inspect to see if it takes ctx
                            import inspect
                            sig = inspect.signature(tool_func)
                            if "ctx" in sig.parameters:
                                result = tool_func(ctx, **tool_args)
                            else:
                                result = tool_func(**tool_args)
                            
                            logger.info(f"Tool {tool_name} result: {result}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id"),
                                "name": tool_name,
                                "content": str(result)
                            })
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id"),
                                "name": tool_name,
                                "content": f"Error: {str(e)}"
                            })
                
                # Continue conversation loop with updated messages
                continue

        return ""

    def _prepare_tools(self, tools: list[Callable[..., Any]]) -> list[dict[str, Any]]:
        """Convert Python callables to OpenAI-compatible tool schemas."""
        import inspect
        from docstring_parser import parse

        schemas = []
        for tool in tools:
            doc = parse(tool.__doc__ or "")
            sig = inspect.signature(tool)
            
            properties = {}
            required = []
            
            for name, param in sig.parameters.items():
                if name == "ctx": # Skip Pydantic AI context
                    continue
                
                param_doc = next((p for p in doc.params if p.arg_name == name), None)
                
                # Basic type mapping
                p_type = "string"
                if param.annotation == float or param.annotation == int:
                    p_type = "number"
                elif param.annotation == bool:
                    p_type = "boolean"
                
                properties[name] = {
                    "type": p_type,
                    "description": param_doc.description if param_doc else ""
                }
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": doc.short_description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    async def embed_text(self, text: str) -> list[float]:
        """
        Create a text embedding using the remote embedding server.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        payload = {
            "input": text,
            "model_name": None # Server uses its own
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self._base_url}/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
