from dependency_injector import containers, providers
from app.infrastructure.openai_adapter import OpenAIProvider
from app.infrastructure.anthropic_adapter import AnthropicProvider
from app.infrastructure.mlx_adapter import MLXProvider
from app.services.conversation import ConversationService
from app.config import settings

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Provider Selection Logic
    llm_provider = providers.Selector(
        config.llm_backend,
        openai=providers.Singleton(
            OpenAIProvider,
            api_key=config.openai_api_key,
            model=config.openai_model
        ),
        anthropic=providers.Singleton(
            AnthropicProvider,
            api_key=config.anthropic_api_key,
            model=config.anthropic_model
        ),
        mlx=providers.Singleton(
            MLXProvider,
            model_path=config.mlx_model_path
        )
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        llm_provider=llm_provider
    )
