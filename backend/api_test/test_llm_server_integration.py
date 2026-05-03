import pytest
import logging
from app.infrastructure.llm_connector.llm_service import LLMService
from app.domain.entity.agent_factory import AgentFactory
from app.domain.entity.message import Message
from app.domain.enum.role import Role
from app.config.app_setting import app_setting

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
import uuid

def create_msg(role: Role, content: str) -> Message:
    return Message(
        id=str(uuid.uuid4()),
        role=role,
        content=content,
        timestamp=int(time.time() * 1000)
    )

@pytest.fixture
def llm_service():
    return LLMService(base_url=app_setting.llm_server_url)

@pytest.mark.asyncio
async def test_simple_generation(llm_service):
    """Test basic text generation without tools."""
    agent = AgentFactory.document_assistant(
        system_prompt="You are a helpful assistant. Reply with only the word 'ACK'.",
    )
    messages = [create_msg(Role.USER, "Please acknowledge.")]
    
    reply = await llm_service.agent_complete_chat(
        message_list=messages,
        domain_agent=agent
    )
    
    logger.info(f"Simple generation reply: {reply}")
    assert "ACK" in reply.upper()

@pytest.mark.asyncio
async def test_embedding(llm_service):
    """Test text embedding generation."""
    text = "This is a test sentence for embedding."
    embedding = await llm_service.embed_text(text=text)
    
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    logger.info(f"Embedding generated with dimension: {len(embedding)}")

@pytest.mark.asyncio
async def test_tool_calling(llm_service):
    """Test generation with tool calling loop."""
    
    def get_weather(ctx, location: str) -> str:
        """
        Get the current weather in a given location.

        Args:
            location: The name of the city, e.g. 'Paris, France'.
        """
        if "Paris" in location:
            return "Sunny, 25°C"
        return "Unknown"

    agent = AgentFactory.document_assistant(
        system_prompt="You are a weather assistant. Use the tools provided to answer questions. If you need to know the weather, use the get_weather tool.",
        tools=[get_weather]
    )
    messages = [create_msg(Role.USER, "What is the weather like in Paris?")]
    
    reply = await llm_service.agent_complete_chat(
        message_list=messages,
        domain_agent=agent
    )
    
    logger.info(f"Tool calling reply: {reply}")
    assert "25" in reply
    assert "Paris" in reply or "Sunny" in reply

@pytest.mark.asyncio
async def test_multi_turn_chat(llm_service):
    """Test multi-turn conversation memory."""
    agent = AgentFactory.document_assistant(
        system_prompt="You are a helpful assistant.",
    )
    messages = [
        create_msg(Role.USER, "My name is Bob."),
        create_msg(Role.ASSISTANT, "Hello Bob! How can I help you today?"),
        create_msg(Role.USER, "What is my name?")
    ]
    
    reply = await llm_service.agent_complete_chat(
        message_list=messages,
        domain_agent=agent
    )
    
    logger.info(f"Multi-turn reply: {reply}")
    assert "Bob" in reply
