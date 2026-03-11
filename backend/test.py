import json
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from app.infra.llm_connector.llm_client import LLMService
from app.infra.llm_connector.parsing_service import ParsingService
from app.domain.entity.message import Message
from app.constant import Role

_llm_client = LLMService(ParsingService())

CHAT_MODEL = "./models/chat/mlx-community/Qwen3.5-35B-A3B-4bit"
EMBED_MODEL = "./models/embedding/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

# ---------------------------------------------------------------------------
# Dummy tools for tool-binding test
# ---------------------------------------------------------------------------

class GetWeatherInput(BaseModel):
    city: str = Field(description="The city to get the weather for")

class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Returns the current weather for a given city."
    args_schema: Type[BaseModel] = GetWeatherInput

    def _run(self, city: str) -> str:
        # Stub – return fake data so we can verify the LLM calls the tool
        return json.dumps({"city": city, "temperature": "22°C", "condition": "Sunny"})


class CalculatorInput(BaseModel):
    expression: str = Field(description="A simple arithmetic expression to evaluate, e.g. '2 + 3 * 4'")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluates a simple arithmetic expression and returns the result."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"


# # ---------------------------------------------------------------------------
# # Test 1 – plain chat (no tools)
# # ---------------------------------------------------------------------------
# print("\n" + "=" * 60)
# print("TEST 1: Plain chat (no tools)")
# print("=" * 60)
# res = _llm_client.complete_chat(
#     model_name=CHAT_MODEL,
#     message_list=[Message(id="msg_1", content="What is Domain-Driven Design in one sentence?", role=Role.USER, timestamp=1674567890)],
#     system_prompt="You are a helpful assistant. Be concise.",
#     tools=[],
# )
# print("Response:", res)


# ---------------------------------------------------------------------------
# Test 2 – chat with tools bound
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Chat with tool binding (weather + calculator)")
print("=" * 60)
tools = [GetWeatherTool(), CalculatorTool()]
res_tools = _llm_client.complete_chat(
    model_path=CHAT_MODEL,
    message_list=[Message(id="msg_2", content="What is the weather in Paris? Also, what is 123 * 456?", role=Role.USER, timestamp=1674567891)],
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
    tools=tools,
)
print("Response:", res_tools)


# ---------------------------------------------------------------------------
# Test 3 – text embedding
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: Text embedding")
print("=" * 60)

texts = [
    "Domain-Driven Design is an approach to software development.",
    "The quick brown fox jumps over the lazy dog.",
]
for text in texts:
    embedding = _llm_client.embed_text(EMBED_MODEL, text)
    print(f"Text : {text!r}")
    print(f"Dims : {len(embedding)}")
    print(f"First5: {[round(v, 4) for v in embedding[:5]]}")
    print()
