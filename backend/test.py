"""
Integration tests for LLMService.

Covers:
  1. Simple chat  — complete_chat(), single-turn and multi-turn
  2. Tool calling — agent_complete_chat() with custom inline tools

Run from the project root:
    python test.py
"""

import math
import time

from langchain.tools import tool

from src.container import container
from src.config.config import DEFAULT_CHAT_MODEL
from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_MODEL = DEFAULT_CHAT_MODEL

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------


@tool(description="Add two numbers together and return the result.")
def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


@tool(description="Multiply two numbers together and return the result.")
def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return a * b


@tool(description="Return the square root of a non-negative number.")
def square_root(x: float) -> float:
    """Return sqrt(x)."""
    return math.sqrt(x)


@tool(description="Return today's date as a string in YYYY-MM-DD format.")
def get_current_date() -> str:
    """Return the current date."""
    return time.strftime("%Y-%m-%d")


TOOLS = [add, multiply, square_root, get_current_date]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: Role, content: str, idx: int = 1) -> Message:
    return Message(role=role, content=content, id=f"msg_{idx}", timestamp=int(time.time()))


def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _result(label: str, reply: str) -> None:
    preview = reply[:400] + ("…" if len(reply) > 400 else "")
    print(f"[OK] {label}\n     {preview}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup_logging()

llm_service = container.llm_service()
llm_manager = container.llm_manager()

_header("Loading model")
llm_manager.load_model(CHAT_MODEL, "chat")
print(f"  chat model: {CHAT_MODEL}")

# ---------------------------------------------------------------------------
# Test 1 — Simple chat (no tools)
# ---------------------------------------------------------------------------

_header("Test 1 — Simple chat")

# 1a. Single-turn factual question
reply = llm_service.complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is the capital of France?")],
    system_prompt="You are a concise assistant. Answer in one sentence.",
)
_result("1a. Single-turn factual question", reply)

# 1b. Multi-turn — model must recall earlier context
reply = llm_service.complete_chat(
    model_path=CHAT_MODEL,
    message_list=[
        _msg(Role.USER,      "My name is Alice.", idx=1),
        _msg(Role.ASSISTANT, "Nice to meet you, Alice!", idx=2),
        _msg(Role.USER,      "What is my name?", idx=3),
    ],
    system_prompt="You are a concise assistant.",
)
_result("1b. Multi-turn memory", reply)

# ---------------------------------------------------------------------------
# Test 2 — Tool calling
# ---------------------------------------------------------------------------

_header("Test 2 — Tool calling")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to math and date tools. "
    "Always use the appropriate tool to compute answers rather than guessing."
)

# 2a. Requires add tool
reply = llm_service.agent_complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is 347 + 658?")],
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
_result("2a. add(347, 658) — expected 1005", reply)

# 2b. Requires multiply tool
reply = llm_service.agent_complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is 123 multiplied by 456?")],
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
_result("2b. multiply(123, 456) — expected 56088", reply)

# 2c. Requires square_root tool
reply = llm_service.agent_complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is the square root of 144?")],
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
_result("2c. square_root(144) — expected 12.0", reply)

# 2d. Requires chaining two tools: multiply then add
reply = llm_service.agent_complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is (6 * 7) + 50?")],
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
_result("2d. multiply(6,7) then add(42,50) — expected 92", reply)

# 2e. Requires get_current_date tool
reply = llm_service.agent_complete_chat(
    model_path=CHAT_MODEL,
    message_list=[_msg(Role.USER, "What is today's date?")],
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)
_result("2e. get_current_date()", reply)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

_header("All tests completed")