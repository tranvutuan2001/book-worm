"""
Integration tests for LLMService.

Covers:
  1. Simple chat  — agent_complete_chat(), single-turn and multi-turn
  2. Tool calling — agent_complete_chat() with custom inline tools

Tools are plain Python functions with ``ctx: RunContext[None]`` as the first
parameter, matching the Pydantic AI convention.

Run from the project root:
    python test.py
"""

import math
import time
from pydantic_ai import RunContext
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


def add(ctx: RunContext[None], a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    print(f"Tool add() called with a={a}, b={b}")
    return a + b


def multiply(ctx: RunContext[None], a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b


def square_root(ctx: RunContext[None], x: float) -> float:
    """Return the square root of a non-negative number."""
    return math.sqrt(x)


TOOLS = [add, multiply, square_root]

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
def test1() -> bool:
    _header("Test 1 — Simple chat (no tools)")
    # 1a. Single-turn factual question
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, "What is the capital of France?")],
        system_prompt="You are a concise assistant. Answer in one sentence.",
        tools=[],
    )
    _result("1a. Single-turn factual question", reply)

    # 1b. Multi-turn — model must recall earlier context
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[
            _msg(Role.USER,      "My name is Alice.", idx=1),
            _msg(Role.ASSISTANT, "Nice to meet you, Alice!", idx=2),
            _msg(Role.USER,      "What is my name?", idx=3),
        ],
        system_prompt="You are a concise assistant.",
        tools=[],
    )
    _result("1b. Multi-turn memory", reply)
    return True

# ---------------------------------------------------------------------------
# Test 2 — Tool calling
# ---------------------------------------------------------------------------

def test2() -> bool:
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

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
test1()
test2()
_header("All tests completed")