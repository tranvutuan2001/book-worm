"""
Chat service — orchestrates the document Q&A use case.

Receives a ``Conversation`` domain object, retrieves relevant context from the
document store via LLM tools, and returns a plain string answer.

This module has NO dependency on FastAPI; exceptions raised here are translated
to HTTP responses by the API route layer.
"""

import logging
import time

from langfuse import observe
from app.util.exceptions import DocumentNotFoundError, LLMError
from app.config.app_setting import app_setting
from app.domain.entity.agent_factory import AgentFactory

from app.domain.entity.message import Message
from app.domain.enum.role import Role
from app.infrastructure.llm_connector import LLMService
from app.infrastructure.logging_config import (
    end_request_logging,
    get_request_logger,
    start_request_logging,
)
from app.domain.enum.tool_type import ToolType
from app.services.tools.tool_factory import ToolFactory
from app.services.commands.ask_question_command import AskQuestionCommand

logger = logging.getLogger("app.service")

_CHAT_TOOL_TYPES = (
    ToolType.DOCUMENT_SEARCH,
    ToolType.DOCUMENT_SUMMARY,
    ToolType.DOCUMENT_TITLE,
)


class ChatService:
    """Handles the document Q&A use case."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @observe()
    async def ask(self, command: AskQuestionCommand) -> str:
        """Answer the user's latest question about a document.

        Args:
            command: AskQuestionCommand containing messages and document info.

        Returns:
            Verified answer string.

        Raises:
            DocumentNotFoundError: If the document folder does not exist.
            LLMError: If the LLM call fails.
        """
        user_query = (
            command.messages[-1].content
            if command.messages
            else "No query"
        )
        
        start_request_logging(endpoint="/ask", user_query=user_query)
        req_logger = get_request_logger("app.api")

        req_logger.info(
            "Processing %d messages for document: %s",
            len(command.messages),
            command.document_name,
        )

        try:
            self._validate_document(command.document_name)

            tools = [
                ToolFactory.create(t, command.document_name)
                for t in _CHAT_TOOL_TYPES
            ]
            answer = await self._generate_answer(command.messages, tools)
            verified = await self._verify_answer(
                command.messages,
                answer,
                tools,
            )

            end_request_logging(response_summary=verified, success=True)
            return verified

        except (DocumentNotFoundError, LLMError):
            raise
        except Exception as exc:
            end_request_logging(response_summary=str(exc), success=False)
            logger.error("Unexpected error in ask(): %s", exc, exc_info=True)
            raise LLMError(f"Unexpected error during chat: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_document(self, document_name: str | None) -> None:
        if not document_name:
            raise DocumentNotFoundError("Document name is required")
        doc_path = app_setting.data_storage_path / document_name
        if not doc_path.exists():
            raise DocumentNotFoundError(
                f"Document '{document_name}' not found at {doc_path}"
            )

    async def _generate_answer(
        self,
        message_list: list[Message],
        tools: list,
    ) -> str:
        try:
            agent = AgentFactory.document_assistant(
                tools=tools,
            )
            return await self._llm.agent_complete_chat(
                message_list=message_list,
                domain_agent=agent,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            raise LLMError(f"Failed to generate answer: {exc}") from exc

    async def _verify_answer(
        self,
        message_list: list[Message],
        answer: str,
        tools: list,
    ) -> str:
        req_logger = get_request_logger("app.api")

        context = (
            "\n".join(f"{m.role.value}: {m.content}" for m in message_list)
            if message_list
            else "N/A"
        )
        current_query = message_list[-1].content if message_list else "N/A"

        verification_prompt = (
            f"Conversation Context:\n{context}\n\n"
            f"Current User Query: {current_query}\n\n"
            f"Answer to Review:\n{answer}\n\n"
            "Verify every factual claim using tools. Return the verified final answer."
        )

        verification_message = [
            Message(
                id="verification",
                role=Role.USER,
                content=verification_prompt,
                timestamp=int(time.time() * 1000),
            )
        ]

        req_logger.info("Starting verification step…")
        try:
            verify_agent = AgentFactory.verify(
                tools=tools,
            )
            verified = await self._llm.agent_complete_chat(
                message_list=verification_message,
                domain_agent=verify_agent,
            )
            req_logger.info("Verification complete (%d chars)", len(verified))
            return verified
        except Exception as exc:
            logger.error("Verification LLM call failed: %s", exc, exc_info=True)
            raise LLMError(f"Answer verification failed: {exc}") from exc

