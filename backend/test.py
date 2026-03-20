from src.domain.entity.message import Message
from src.domain.enums import Role
from src.infra.llm_connector import LLMService, get_llm_service
from src.infra.logging_config import setup_logging

llm_service = get_llm_service()

llm_service.load_model("models/chat/mlx-community/Qwen3.5-9B-MLX-4bit", "chat")

setup_logging()

res = llm_service.complete_chat(
    model_path="models/chat/mlx-community/Qwen3.5-9B-MLX-4bit",
    message_list=[Message(role = Role.USER, content = "hi", id="1", timestamp=0)],
    system_prompt="",
)

print(res)