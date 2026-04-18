"""
Agent domain entity.

:class:`Agent` is a pure data-holder that carries everything :class:`LLMService`
needs to run an LLM job: agent type, system prompt, tools, generation settings,
and retry count.

To construct an :class:`Agent` use :class:`~src.domain.entity.agent_factory.AgentFactory`
rather than instantiating this dataclass directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from src.config.config import (
    AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
    AGENT_SUMMARY_SYSTEM_PROMPT,
    AGENT_VERIFY_SYSTEM_PROMPT,
)
from src.domain.enums import AgentType
from src.domain.value_object.chat_model_setting import ChatModelSettings


@dataclass
class Agent:
    """A configured LLM agent ready to be executed by :class:`LLMService`.

    Attributes:
        agent_type:     Functional category of the agent (see :class:`AgentType`).
        system_prompt:  Instruction prepended to every conversation this agent
                        handles.
        tools:          Plain Python callables the agent may invoke during a run.
        model_settings: Token-limit, temperature, and other generation knobs.
        max_retries:    Maximum retries on output-validation errors.
    """

    agent_type: AgentType
    system_prompt: str
    tools: list[Callable[..., Any]] = field(default_factory=list)
    model_settings: ChatModelSettings = field(default_factory=ChatModelSettings)
    max_retries: int = 3




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
        """Create a summarization agent.

        The agent condenses text into a high-density, structured representation.
        The default system prompt is
        :data:`~src.config.config.AGENT_SUMMARY_SYSTEM_PROMPT`.

        Args:
            system_prompt:  Optional override for the default prompt.
            model_settings: Generation settings; defaults to
                            :class:`~src.domain.value_object.chat_model_setting.ChatModelSettings`.
            max_retries:    Validation-error retries.

        Returns:
            A configured :class:`Agent` of type :attr:`~src.domain.enums.AgentType.SUMMARY`.
        """
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
        """Create a verification agent.

        The agent examines a (task, result) pair and must answer **yes** or
        **no**.  The default system prompt is
        :data:`~src.config.config.AGENT_VERIFY_SYSTEM_PROMPT`.

        Args:
            system_prompt:  Optional override for the default prompt.
            tools:          Optional retrieval callables for fact-checking.
            model_settings: Generation settings; defaults to
                            :class:`~src.domain.value_object.chat_model_setting.ChatModelSettings`.
            max_retries:    Validation-error retries.

        Returns:
            A configured :class:`Agent` of type :attr:`~src.domain.enums.AgentType.VERIFY`.
        """
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
        """Create a document-assistant agent.

        The agent answers user questions about a document by invoking
        retrieval tools as needed.  The default system prompt is
        :data:`~src.config.config.AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT`.

        Args:
            system_prompt:  Optional override / enrichment of the default prompt
                            (e.g. append the document name).
            tools:          Retrieval callables (chunk search, summary fetch,
                            etc.).
            model_settings: Generation settings; defaults to
                            :class:`~src.domain.value_object.chat_model_setting.ChatModelSettings`.
            max_retries:    Validation-error retries.

        Returns:
            A configured :class:`Agent` of type
            :attr:`~src.domain.enums.AgentType.DOCUMENT_ASSISTANT`.
        """
        return Agent(
            agent_type=AgentType.DOCUMENT_ASSISTANT,
            system_prompt=system_prompt or AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT,
            tools=tools or [],
            model_settings=model_settings or ChatModelSettings(),
            max_retries=max_retries,
        )
