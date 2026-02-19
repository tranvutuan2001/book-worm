import os
import logging
import traceback
from typing import List, TypeVar
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.domain.entity.message import Message
from langchain.tools import BaseTool
from app.infra.llm_connector.LLMLoggingHandler import LLMLoggingHandler
from openai import OpenAI

logger = logging.getLogger('app.llm_connector')

T = TypeVar('T', bound=BaseModel)
# select base_url depending on environment
env = os.getenv("ENVIRONMENT", "").lower()
base_url = "http://llm-server:8000/v1" if env == "production" else "http://localhost:8001/v1"

def complete_chat(model_name: str, message_list: List[Message], system_prompt: str, tools: List[BaseTool], timeout: int = 1500) -> str:
    """
    Complete a chat interaction with LLM using provided tools.
    All LLM steps and tool calls are automatically logged via LLMLoggingHandler.
    """
    try:
        # Initialize the logging callback handler
        logging_handler = LLMLoggingHandler()

        logger.info(f"Initializing LLM with model: {model_name}, timeout: {timeout}s")
        
        try:
            llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key="custom-api-key",
                temperature=0.1,
                timeout=timeout,
                callbacks=[logging_handler],
            )
        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        try:
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
            )
            logger.info(f"Agent created with {len(tools)} tools")
        except Exception as e:
            error_msg = f"Failed to create agent: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        try:
            messages = [
                {"role": message.role.value, "content": message.content}
                for message in message_list
            ]
            logger.info(f"Invoking agent with {len(messages)} messages")
        except Exception as e:
            error_msg = f"Failed to format messages: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise

        # Invoke agent with callback handler - this will automatically log all steps
        try:
            response = agent.invoke(
                input={"messages": messages},
                config={"callbacks": [logging_handler]}
            )
            logger.info("Agent invocation completed successfully")
        except TimeoutError as e:
            error_msg = f"LLM request timed out after {timeout}s: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Agent invocation failed: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise

        try:
            last_message = response["messages"][-1]
            return last_message.content
        except (KeyError, IndexError) as e:
            error_msg = f"Failed to extract response content: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Response structure: {response}")
            raise
    
    except Exception as e:
        error_msg = f"Unexpected error in complete_chat: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise


def embed_text(model_name: str, text: str):
    try:
        logger.info(f"Creating embedding with model: {model_name}, text length: {len(text)}")
        
        try:
            client = OpenAI(base_url=base_url, api_key="custom-api-key")
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client for embedding: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise

        try:
            response = client.embeddings.create(input=text, model=model_name, encoding_format='float')
            logger.info("Embedding created successfully")
            return response.data[0].embedding
        except TimeoutError as e:
            error_msg = f"Embedding request timed out: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Embedding API call failed: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise
    
    except Exception as e:
        error_msg = f"Unexpected error in embed_text: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise
