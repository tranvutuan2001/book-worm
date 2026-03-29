"""
LM Studio connectivity tests.

Tests:
1. Chat completion (qwen3.5-9b-mlx)
2. Embedding generation (qwen3-embedding-0.6b-dwq)
3. Tool / function-calling via the agent loop

Run:
    python test_lm_studio.py
"""

from __future__ import annotations

import sys
import textwrap
from typing import List, Type

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.core import config
from src.infra.llm_connector.external_llm.lm_studio_chat import LMStudioChatModel
from src.infra.llm_connector.external_llm.lm_studio_embedding import LMStudioEmbeddingModel
from src.infra.llm_connector.llm_manager import LLMManager
from src.infra.llm_connector.llm_service import LLMService
from src.infra.llm_connector.local_llm.parsing_service import ParsingService
from src.domain.entity.message import Message
from src.domain.enums import Role

# ---------------------------------------------------------------------------
# Model identifiers as they appear in LM Studio
# ---------------------------------------------------------------------------

CHAT_MODEL = "qwen3.5-9b-mlx"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

BASE_URL = config.LM_STUDIO_BASE_URL   # http://localhost:1234/v1
API_KEY = config.LM_STUDIO_API_KEY     # lm-studio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS}  {label}{suffix}")


def _err(label: str, exc: Exception) -> None:
    print(f"  {FAIL}  {label}")
    print(f"         {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 1 — Chat completion
# ---------------------------------------------------------------------------

def test_chat() -> bool:
    _section("Test 1 · Chat completion")
    try:
        model = LMStudioChatModel(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=CHAT_MODEL,
            max_tokens=256,
            temperature=0.0,
        )
        response = model.invoke([HumanMessage(content="Reply with exactly: hello from lm-studio")])
        reply: str = response.content.strip()
        _ok("Received response", f"'{reply[:120]}'")
        return True
    except Exception as exc:
        _err("Chat completion failed", exc)
        return False


# ---------------------------------------------------------------------------
# Test 2 — Embedding generation
# ---------------------------------------------------------------------------

def test_embedding() -> bool:
    _section("Test 2 · Embedding generation")
    try:
        model = LMStudioEmbeddingModel(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=EMBEDDING_MODEL,
        )
        vector: List[float] = model.embed("What is compound interest?")
        dim = len(vector)
        norm = sum(x ** 2 for x in vector) ** 0.5
        _ok("Received embedding", f"dim={dim}  ‖v‖={norm:.4f}")
        return True
    except Exception as exc:
        _err("Embedding failed", exc)
        return False


# ---------------------------------------------------------------------------
# Test 3 — Tool / function-calling
# ---------------------------------------------------------------------------

# Define a trivial calculator tool so no external state is needed.

class _CalcInput(BaseModel):
    a: float = Field(description="First operand")
    b: float = Field(description="Second operand")
    op: str = Field(description="Operator: add, subtract, multiply, divide")


@tool(args_schema=_CalcInput)
def calculator(a: float, b: float, op: str) -> str:
    """Perform a basic arithmetic operation and return the result as a string."""
    if op == "add":
        return str(a + b)
    if op == "subtract":
        return str(a - b)
    if op == "multiply":
        return str(a * b)
    if op == "divide":
        if b == 0:
            return "Error: division by zero"
        return str(a / b)
    return f"Unknown operator '{op}'"


def test_tools() -> bool:
    _section("Test 3 · Tool / function-calling (agent loop)")
    try:
        # Build a minimal LLMService backed by LM Studio without starting FastAPI.
        parsing_service = ParsingService()
        manager = LLMManager(parsing_service=parsing_service)
        service = LLMService(llm_manager=manager)

        messages = [
            Message(role=Role.USER, content="What is 17 multiplied by 6? Use the calculator tool.", id="1", timestamp=0),
        ]

        answer = service.agent_complete_chat(
            model_path=CHAT_MODEL,
            message_list=messages,
            system_prompt=(
                "You are a helpful assistant. "
                "When the user asks for a calculation, use the calculator tool."
            ),
            tools=[calculator],
            max_iterations=10,
            max_tokens=512,
            temperature=0.0,
        )

        _ok("Agent returned answer", f"'{str(answer)[:200]}'")
        if "102" in str(answer):
            _ok("Answer contains expected result (102)")
        else:
            print(f"  \033[93m[WARN]\033[0m  Expected '102' in answer — verify manually.")
        return True
    except Exception as exc:
        _err("Tool execution failed", exc)
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nLM Studio connectivity test suite")
    print(f"  base_url : {BASE_URL}")
    print(f"  chat     : {CHAT_MODEL}")
    print(f"  embed    : {EMBEDDING_MODEL}")

    results = {
        "chat":      test_chat(),
        "embedding": test_embedding(),
        "tools":     test_tools(),
    }

    _section("Summary")
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
