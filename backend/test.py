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
from src.domain.entity.agent import AgentFactory
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
        agent=AgentFactory.document_assistant(
            system_prompt="You are a concise assistant. Answer in one sentence.",
        ),
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
        agent=AgentFactory.document_assistant(
            system_prompt="You are a concise assistant.",
        ),
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
        agent=AgentFactory.document_assistant(
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        ),
    )

    _result("2a. add(347, 658) — expected 1005", reply)

# ---------------------------------------------------------------------------
# Test 3 — Summary agent
# ---------------------------------------------------------------------------

def test3() -> bool:
    _header("Test 3 — Summary agent")

    long_text = (
        "The Python programming language was created by Guido van Rossum and first "
        "released in 1991. It emphasises code readability and uses significant "
        "indentation. Python's design philosophy, documented in 'The Zen of Python', "
        "includes aphorisms such as 'Beautiful is better than ugly' and 'Simple is "
        "better than complex'. The language supports multiple programming paradigms, "
        "including procedural, object-oriented, and functional programming. It has a "
        "comprehensive standard library and an active community that publishes third-"
        "party packages on the Python Package Index (PyPI). Python is widely used in "
        "web development, data science, artificial intelligence, scientific computing, "
        "and automation."
    )

    # 3a. Default summary system prompt
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, f"Summarise the following text:\n\n{long_text}")],
        agent=AgentFactory.summary(),
    )
    _result("3a. Summary with default prompt", reply)

    # 3b. Custom system prompt override
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, f"Summarise in exactly two bullet points:\n\n{long_text}")],
        agent=AgentFactory.summary(
            system_prompt="You are a concise summariser. Always respond with exactly two bullet points.",
        ),
    )
    _result("3b. Summary with custom prompt (two bullet points)", reply)
    return True


# ---------------------------------------------------------------------------
# Test 4 — Verify agent
# ---------------------------------------------------------------------------

def test4() -> bool:
    _header("Test 4 — Verify agent")

    # 4a. Correct answer — expect "yes"
    task_correct = (
        "Task: What is the capital of Germany?\n"
        "Answer: Berlin"
    )
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, task_correct)],
        agent=AgentFactory.verify(),
    )
    _result("4a. Verify correct answer (expect yes)", reply)

    # 4b. Incorrect answer — expect "no"
    task_wrong = (
        "Task: What is the capital of France?\n"
        "Answer: Lyon"
    )
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, task_wrong)],
        agent=AgentFactory.verify(),
    )
    _result("4b. Verify wrong answer (expect no)", reply)

    # 4c. Verify with custom prompt override
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, "Task: Is 7 a prime number?\nAnswer: Yes")],
        agent=AgentFactory.verify(
            system_prompt=(
                "You are a strict fact-checker. "
                "Respond with only 'yes' if the answer is correct, or 'no' if it is wrong."
            ),
        ),
    )
    _result("4c. Verify with custom prompt (expect yes)", reply)
    return True


# ---------------------------------------------------------------------------
# Test 5 — Document assistant agent (extended)
# ---------------------------------------------------------------------------

def test5() -> bool:
    _header("Test 5 — Document assistant agent (extended)")

    # 5a. With math tools — sqrt
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, "What is the square root of 144?")],
        agent=AgentFactory.document_assistant(
            system_prompt=(
                "You are a helpful assistant. "
                "Always use the square_root tool when asked about square roots."
            ),
            tools=TOOLS,
        ),
    )
    _result("5a. square_root(144) — expected 12", reply)

    # 5b. Multi-step tool use — multiply then add
    reply = llm_service.agent_complete_chat(
        model_path=CHAT_MODEL,
        message_list=[_msg(Role.USER, "What is (3 × 7) + 5?")],
        agent=AgentFactory.document_assistant(
            system_prompt=(
                "You are a helpful math assistant. "
                "Use tools step-by-step to compute the answer."
            ),
            tools=TOOLS,
        ),
    )
    _result("5b. multiply(3,7) + add(21,5) — expected 26", reply)
    return True

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
test1()
test2()
test3()
test4()
test5()
_header("All tests completed")