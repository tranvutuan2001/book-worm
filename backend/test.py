import logging
import os

from src.infra.logging_config import setup_logging
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

from src.infra.llm_connector.llm_service import get_llm_service
from src.service.tools.document_retrieval_tool import get_document_summary
from src.domain.entity.message import Message
from src.domain.enums import Role
from src.core.config import DEFAULT_CHAT_MODEL

DOCUMENT_NAME = "The Simple Path to Wealth_ Your - J Collins_20260312_000311"

# 1. Retrieve the document summary as context
doc_summary = get_document_summary(DOCUMENT_NAME)

# 2. Build the message list
user_question = "What is the main investment strategy recommended in this book?"

messages = [
    Message(
        id="msg_1",
        content=(
            f"Rewrite the following raw document summary into a clean, "
                f"comprehensive summary:\n\n{doc_summary}"
        ),
        role=Role.USER,
        timestamp=int(time.time()),
    )
]

# # 3. Define a system prompt that includes the document summary as context
# from src.service.pdf_summarization_service import _STEP1_SYSTEM

# # 4. Run the chat completion
# llm_client = get_llm_service()

# print(f"=== Running chat completion with model: {DEFAULT_CHAT_MODEL} ===")
# res = llm_client.complete_chat(
#     model_path=DEFAULT_CHAT_MODEL,
#     message_list=messages,
#     system_prompt=_STEP1_SYSTEM,
#     template_name="qwen",
#     max_tokens=9000,
# )

setup_logging()
logger = logging.getLogger("app.test")
logger.info(doc_summary)

# print("=== Response ===")
# print(res)
