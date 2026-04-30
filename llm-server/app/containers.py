from dependency_injector import containers, providers
from app.infrastructure.openai_adapter import OpenAIProvider
from app.infrastructure.anthropic_adapter import AnthropicProvider
from app.infrastructure.mlx_adapter import MLXProvider
from app.services.conversation import ConversationService
from app.services.embedding import EmbeddingService
from app.config import settings

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Provider Selection Logic for LLM
    llm_provider = providers.Selector(
        config.llm.backend,
        openai=providers.Singleton(
            OpenAIProvider,
            api_key=config.llm.openai_key,
            model=config.llm.openai_model
        ),
        anthropic=providers.Singleton(
            AnthropicProvider,
            api_key=config.llm.anthropic_key,
            model=config.llm.anthropic_model
        ),
        mlx=providers.Singleton(
            MLXProvider,
            model_path=config.llm.mlx_path
        )
    )
    
    # Provider Selection Logic for Embeddings
    embedding_provider = providers.Selector(
        config.llm.backend,
        openai=providers.Singleton(
            OpenAIProvider,
            api_key=config.llm.openai_key,
            model=config.llm.openai_model
        ),
        mlx=providers.Singleton(
            MLXProvider,
            model_path=config.llm.mlx_path
        ),
        anthropic=providers.Singleton(
            # Anthropic doesn't support embeddings, fallback or raise
            OpenAIProvider, # Or a dummy provider
            api_key=config.llm.openai_key
        )
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        llm_provider=llm_provider
    )

    embedding_service = providers.Factory(
        EmbeddingService,
        embedding_provider=embedding_provider
    )
