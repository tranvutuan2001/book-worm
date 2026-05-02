from __future__ import annotations
from typing import Any, Callable
from app.domain.enum.agent_type import AgentType
from app.domain.value_object.chat_model_setting import ChatModelSettings
from app.domain.entity.agent import (
    Agent, 
    AGENT_SUMMARY_SYSTEM_PROMPT, 
    AGENT_VERIFY_SYSTEM_PROMPT, 
    AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT
)

class AgentFactory:
    """Creates pre-configured :class:`Agent` instances for each job type.

    All methods are ``@staticmethod`` — no factory instance is needed.
    """

    @staticmethod
    def summary(
        *,
        system_prompt: str | None = None,
        model_settings: ChatModelSettings | None = None,
        max_retries: int = 3,
    ) -> Agent:
        return Agent(
            agent_type=AgentType.SUMMARY,
            system_prompt=system_prompt or AGENT_SUMMARY_SYSTEM_PROMPT,
            tools=[],
            model_settings=model_settings or ChatModelSettings(),
            max_retries=max_retries,
        )

    @staticmethod
    def verify(
        *,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        model_settings: ChatModelSettings | None = None,
        max_retries: int = 3,
    ) -> Agent:
        return Agent(
            agent_type=AgentType.VERIFY,
            system_prompt=system_prompt or AGENT_VERIFY_SYSTEM_PROMPT,
            tools=tools or [],
            model_settings=model_settings or ChatModelSettings(),
            max_retries=max_retries,
        )

    @staticmethod
    def document_assistant(
        *,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        model_settings: ChatModelSettings | None = None,
        max_retries: int = 3,
    ) -> Agent:
        return Agent(
            agent_type=AgentType.DOCUMENT_ASSISTANT,
            system_prompt=system_prompt or AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
            tools=tools or [],
            model_settings=model_settings or ChatModelSettings(),
            max_retries=max_retries,
        )
