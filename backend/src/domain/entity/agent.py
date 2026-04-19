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

from src.domain.enums import AgentType
from src.domain.value_object.chat_model_setting import ChatModelSettings

# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------

AGENT_SUMMARY_SYSTEM_PROMPT: str = (
    "You are a summarization expert.\n"
    "Condense the provided text into a high-density, structured summary.\n"
    "Focus on hard facts, key concepts, names, and metrics.\n"
    "Use a professional, note-taking style.\n"
    "Output ONLY the summary — no meta-commentary."
)

AGENT_VERIFY_SYSTEM_PROMPT: str = (
    "You are a strict verification assistant.\n"
    "Your task is to check whether the provided answer correctly addresses "
    "the given task.\n"
    "CRITICAL RULES:\n"
    "  - Use only information verifiable with the provided tools.\n"
    "  - Remove any claims that cannot be verified.\n"
    "  - Do NOT add information from your own knowledge.\n"
    "  - Do NOT make assumptions beyond what tools confirm.\n"
    "  - If uncertain, remove the questionable content rather than keeping it.\n"
    "Return only the fact-checked final answer with no meta-commentary."
)

AGENT_COMPLETION_CHECK_SYSTEM_PROMPT: str = (
    "You are a text-completeness checker.\n"
    "Given a piece of text, determine whether it is a complete, finished answer "
    "or whether it appears to have been cut off mid-sentence or mid-thought due to "
    "reaching a token limit.\n"
    "CRITICAL RULES:\n"
    "  - Answer ONLY with the single word 'yes' if the text is complete.\n"
    "  - Answer ONLY with the single word 'no' if the text appears truncated or cut off.\n"
    "  - Do NOT add any explanation, punctuation, or extra words.\n"
    "  - Base your judgment solely on whether the text ends naturally."
)

AGENT_DOCUMENT_ASSISTANT_SYSTEM_PROMPT: str = (
    "You are a knowledgeable assistant in a document-analyzing system.\n"
    "Answer in the language of the question.\n"
    "Use the tools to retrieve the information needed to answer.\n"
    "All answers must be grounded in knowledge retrieved from the tools.\n"
    "Do not fabricate answers that are not supported by the tools.\n"
    "At the end of your response, briefly cite which part of the document "
    "informed your answer.\n"
    "Format your answer for human readability.\n"
    'If the answer cannot be found even after using the tools, respond with:\n'
    '"The provided data is not sufficient to answer this question."'
)


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
