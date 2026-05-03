import inspect
from typing import Any, Callable
from docstring_parser import parse

from app.domain.entity.agent import Agent as DomainAgent
from app.domain.entity.message import Message as DomainMessage
from app.infrastructure.llm_connector.dto.chat_message import ChatMessage
from app.infrastructure.llm_connector.dto.tool_definition import ToolDefinition
from app.infrastructure.llm_connector.dto.tool_function_schema import ToolFunctionSchema
from app.infrastructure.llm_connector.dto.completion_request import CompletionRequest


class CompletionRequestMapper:
    """Maps domain Agent and Message entities to a CompletionRequest DTO."""

    @staticmethod
    def map_to_completion_request(
        message_list: list[DomainMessage],
        agent: DomainAgent,
    ) -> CompletionRequest:
        """
        Map domain conversation history and agent configuration to
        a CompletionRequest.
        """
        # 1. Map messages
        chat_messages = [
            ChatMessage(role="system", content=agent.system_prompt)
        ]
        for msg in message_list:
            chat_messages.append(
                ChatMessage(role=msg.role.value, content=msg.content)
            )

        # 2. Map tools
        tools = None
        if agent.tools:
            tools = CompletionRequestMapper._build_tool_definitions(agent.tools)

        return CompletionRequest(
            model="",  # Server selects automatically
            messages=chat_messages,
            temperature=agent.model_settings.temperature,
            tools=tools,
            name="",
            metadata={},
        )

    @staticmethod
    def _build_tool_definitions(tools: list[Callable[..., Any]]) -> list[ToolDefinition]:
        """Convert Python callables to ToolDefinition DTOs."""
        schemas = []
        for tool in tools:
            doc = parse(tool.__doc__ or "")
            sig = inspect.signature(tool)

            properties = {}
            required = []

            for name, param in sig.parameters.items():
                if name == "ctx":  # Skip Pydantic AI context
                    continue

                param_doc = next((p for p in doc.params if p.arg_name == name), None)

                # Basic type mapping
                p_type = "string"
                if param.annotation == float or param.annotation == int:
                    p_type = "number"
                elif param.annotation == bool:
                    p_type = "boolean"
                elif param.annotation == list or param.annotation == list[str]:
                    p_type = "array"

                properties[name] = {
                    "type": p_type,
                    "description": param_doc.description if param_doc else ""
                }
                if param.default is inspect.Parameter.empty:
                    required.append(name)

            schemas.append(
                ToolDefinition(
                    type="function",
                    function=ToolFunctionSchema(
                        name=tool.__name__,
                        description=doc.short_description or "",
                        parameters={
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    )
                )
            )
        return schemas
