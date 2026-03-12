"""
Chat service — orchestrates the document Q&A use case.

Receives a ``Conversation`` domain object, retrieves relevant context from the
document store via LLM tools, and returns a plain string answer.

This module has NO dependency on FastAPI; exceptions raised here are translated
to HTTP responses by the API route layer.
"""

import logging
import time
import traceback
from typing import List

from src.core.exceptions import DocumentNotFoundError, LLMError
from src.core.config import DATA_DIR
from src.domain.entity.conversation import Conversation
from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector.llm_client import LLMService, _llm_service
from src.infra.logging_config import (
    end_request_logging,
    get_request_logger,
    start_request_logging,
)
from src.infra.session_manager import session_manager
from src.service.tools.document_retrieval_tool import (
    get_document_summary,
    get_the_most_relevant_chunks,
)

logger = logging.getLogger("app.service")

_SYSTEM_PROMPT = """
You are a knowledgeable assistant in a document-analyzing system.
Answer in the language of the question.
Use the tools to get necessary information to complete the chat.
All answers must be based on the knowledge retrieved from the tools.
Do not make up answers that are not supported by the tools.
At the end of your response, briefly explain how you arrived at the answer by
citing the document.
If the answer cannot be found even after using the tools, respond with:
"The provided data is not sufficient to answer this question."
Format your answer for human readability.
""".strip()

_VERIFICATION_SYSTEM_PROMPT = """
You are a strict verification assistant. Your job is to fact-check answers
against the document using tools.

CRITICAL RULES:
- ONLY use information that you can verify with the provided tools.
- REMOVE any claims that cannot be verified by checking the actual document.
- DO NOT add information from your own knowledge.
- DO NOT make assumptions beyond what the tools show.
- If uncertain, remove the questionable content rather than keeping it.
- Check whether the answer addresses the user query given the conversation context.
- If the answer is accurate and verified, return it as-is.

Return only the fact-checked final answer with no meta-commentary.
""".strip()


class ChatService:
    """Handles the document Q&A use case."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, conversation: Conversation) -> str:
        """Answer the user's latest question about a document.

        Args:
            conversation: Full conversation payload including document name.

        Returns:
            Verified answer string.

        Raises:
            DocumentNotFoundError: If the document folder does not exist.
            LLMError: If the LLM call fails.
        """
        user_query = (
            conversation.message_list[-1].content
            if conversation.message_list
            else "No query"
        )
        request_id = start_request_logging(endpoint="/ask", user_query=user_query)
        req_logger = get_request_logger("app.api")

        req_logger.info(
            "Processing %d messages for document: %s",
            len(conversation.message_list),
            conversation.document_name,
        )

        try:
            self._validate_document(conversation.document_name)

            with session_manager.session_context(
                conversation.document_name, conversation.embedding_model
            ) as session_id:
                req_logger.info("Session: %s", session_id)
                answer = self._generate_answer(conversation)
                verified = await self._verify_answer(
                    conversation.message_list, answer, conversation.chat_model
                )

            end_request_logging(response_summary=verified, success=True)
            return verified

        except (DocumentNotFoundError, LLMError):
            raise
        except Exception as exc:
            end_request_logging(response_summary=str(exc), success=False)
            logger.error("Unexpected error in ask(): %s\n%s", exc, traceback.format_exc())
            raise LLMError(f"Unexpected error during chat: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_document(self, document_name: str | None) -> None:
        if not document_name:
            raise DocumentNotFoundError("Document name is required")
        doc_path = DATA_DIR / document_name
        if not doc_path.exists():
            raise DocumentNotFoundError(
                f"Document '{document_name}' not found at {doc_path}"
            )

    def _generate_answer(self, conversation: Conversation) -> str:
        try:
            return self._llm.complete_chat(
                message_list=conversation.message_list,
                system_prompt=_SYSTEM_PROMPT,
                tools=[get_the_most_relevant_chunks, get_document_summary],
                model_path=conversation.chat_model,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise LLMError(f"Failed to generate answer: {exc}") from exc

    async def _verify_answer(
        self, message_list: List[Message], answer: str, chat_model: str
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
            verified = self._llm.complete_chat(
                message_list=verification_message,
                system_prompt=_VERIFICATION_SYSTEM_PROMPT,
                tools=[get_the_most_relevant_chunks, get_document_summary],
                model_path=chat_model,
            )
            req_logger.info("Verification complete (%d chars)", len(verified))
            return verified
        except Exception as exc:
            logger.error("Verification LLM call failed: %s", exc)
            raise LLMError(f"Answer verification failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Singleton & dependency factory
# ---------------------------------------------------------------------------

_chat_service: ChatService = ChatService(llm_service=_llm_service)


def get_chat_service() -> ChatService:
    """FastAPI dependency that provides the shared ``ChatService`` instance."""
    return _chat_service
