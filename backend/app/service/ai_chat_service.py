from time import time
from typing import List
from fastapi import HTTPException
from app.constant import Role
from app.domain.entity.conversation import Conversation
from app.controller.dto import AskResponse
from app.domain.entity.message import Message
from app.infra.llm_connector.llm_client import complete_chat
from app.infra.llm_tool.document_analyzing_tool import get_the_most_relevant_chunks, get_document_summary
from app.infra.session_manager import session_manager
from app.infra.logging_config import start_request_logging, end_request_logging, get_request_logger
from pathlib import Path
import time
import traceback

class AIChatService:
    async def ask(self, payload: Conversation) -> AskResponse:
        # Start request logging
        user_query = payload.message_list[-1].content if payload.message_list else "No query"
        request_id = start_request_logging(endpoint="/ask", user_query=user_query)
        
        logger = get_request_logger("app.api")
        logger.info(f"Processing request with {len(payload.message_list)} messages for document: {payload.document_name}")
        
        try:
            # Check if document name is provided
            if not payload.document_name:
                error_msg = "Document name is required but not provided"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=400, detail="Document name is required")
            
            # Check if document exists
            doc_path = Path(f"0_data/{payload.document_name}")
            if not doc_path.exists():
                error_msg = f"Document '{payload.document_name}' not found at path: {doc_path}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=404, detail=f"Document '{payload.document_name}' not found")
            
            logger.info(f"Document found: {payload.document_name}")
            
            # Use session context manager to ensure proper session handling
            with session_manager.session_context(payload.document_name) as session_id:
                message_list = payload.message_list
                logger.info(f"Using session ID: {session_id}")
                
                system_prompt = f"""
                    You are a knowledgeable assistant in a document-analyzing system.
                    Answer in the language of the question.
                    Use the tools to get necessary information to complete the chat.
                    All answers must be based on the knowledge retrieved from the tools.
                    Do not make up answers that are not supported by the tools.
                    At the end of you response, briefly explain how you came up with the answer by providing cite from the document.
                    If the answer cannot be found even after using the tools, respond with "The provided data is not sufficient to answer this question."
                    Format your answer for human-readability.
                """

                logger.info("Starting initial LLM completion...")
                try:
                    answer = complete_chat(message_list=message_list, system_prompt=system_prompt,
                                        tools=[get_the_most_relevant_chunks, get_document_summary], model_name=payload.chat_model)
                    logger.info(f"Initial answer generated: {len(answer)} characters")
                except Exception as e:
                    error_msg = f"Failed to generate initial answer: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    print(f"Stack trace: {traceback.format_exc()}")
                    raise

                try:
                    verified_answer = await self._verify_answer(message_list, answer, payload.chat_model)
                    logger.info("Answer verification completed successfully")
                except Exception as e:
                    error_msg = f"Failed during answer verification: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    print(f"Stack trace: {traceback.format_exc()}")
                    raise
                
                return AskResponse(
                    message=verified_answer, 
                    conversation_id="a27z", 
                    timestamp=123456789
                )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is (they're already logged above)
            raise
        except Exception as e:
            error_msg = f"Unexpected error during chat request: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            end_request_logging(response_summary=f"Error: {str(e)}", success=False)
            raise
    

    async def _verify_answer(self, message_list: List[Message], answer: str, chat_model: str) -> str:
        logger = get_request_logger("app.api")
        
        try:
            context_messages = "\n".join([
                        f"{msg.role.value}: {msg.content}"
                        for msg in message_list
                    ]) if message_list else "N/A"

            current_user_query = message_list[-1].content if message_list else "N/A"

            verification_prompt = f"""
                Conversation Context (previous messages):
                {context_messages}
                
                Current User Query: {current_user_query}
                
                Answer to Review:
                {answer}
                
                Verify the answer above by checking each factual claim against the document using the tools. Return the verified final answer.
            """

            verification_message = [Message(
                id="verification",
                role=Role.USER,
                content=verification_prompt,
                timestamp=int(time.time() * 1000)
            )]

            logger.info("Starting verification step...")
            try:
                verified_answer = complete_chat(
                    message_list=verification_message,
                    system_prompt="""You are a strict verification assistant. Your job is to fact-check answers against the document using tools.

                CRITICAL RULES:
                - ONLY use information that you can verify with the provided tools
                - REMOVE any claims that cannot be verified by checking the actual document with tools
                - DO NOT add information from your own knowledge
                - DO NOT make assumptions or inferences beyond what the tools show
                - If uncertain, remove the questionable content rather than keeping it
                - Check if the answer addresses the user query based on the conversation context
                - If the answer is accurate and verified, return it as-is
                
                Return only the fact-checked final answer with no meta-commentary.""",
            tools=[get_the_most_relevant_chunks, get_document_summary],
            model_name=chat_model
        )
                logger.info(f"Verification completed: {len(verified_answer)} characters")
            except Exception as e:
                error_msg = f"LLM verification call failed: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise
            
            # End request logging with success
            end_request_logging(response_summary=verified_answer, success=True)

            return verified_answer
            
        except Exception as e:
            error_msg = f"Error during answer verification process: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise

ai_service = AIChatService()