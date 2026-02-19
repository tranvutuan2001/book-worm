import os
from typing import List, TypeVar
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.domain.entity.message import Message
from langchain.tools import BaseTool
from app.infra.llm_connector.LLMLoggingHandler import LLMLoggingHandler
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)
# select base_url depending on environment
env = os.getenv("ENVIRONMENT", "").lower()
base_url = "http://llm-server:8000/v1" if env == "production" else "http://localhost:8001/v1"

def complete_chat(model_name: str, message_list: List[Message], system_prompt: str, tools: List[BaseTool], timeout: int = 1500) -> str:
    """
    Complete a chat interaction with LLM using provided tools.
    All LLM steps and tool calls are automatically logged via LLMLoggingHandler.
    """

    # Initialize the logging callback handler
    logging_handler = LLMLoggingHandler()

    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="custom-api-key",
        temperature=0.1,
        timeout=timeout,
        callbacks=[logging_handler],
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    messages = [
        {"role": message.role.value, "content": message.content}
        for message in message_list
    ]

    # Invoke agent with callback handler - this will automa tically log all steps
    response = agent.invoke(
        input={"messages": messages},
        config={"callbacks": [logging_handler]}
    )

    last_message = response["messages"][-1]

    return last_message.content


def embed_text(model_name: str, text: str):
    client = OpenAI(base_url=base_url, api_key="custom-api-key")

    response = client.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding
