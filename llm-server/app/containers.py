from dependency_injector import containers, providers
from openai import AsyncOpenAI
from app.infrastructure.openai_llm_provider import OpenAILLMProvider
from app.infrastructure.openai_embedding_provider import OpenAIEmbeddingProvider
from app.infrastructure.mlx_model import MLXModel
from app.infrastructure.mlx_llm_provider import MLXLLMProvider
from app.infrastructure.mlx_embedding_provider import MLXEmbeddingProvider
from app.services.conversation import ConversationService
from app.services.embedding import EmbeddingService
from app.config import settings

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Common Resources
    openai_client = providers.Singleton(
        AsyncOpenAI,
        api_key=config.llm.openai_key
    )
    
    mlx_model = providers.Singleton(
        MLXModel,
        model_path=config.llm.mlx_path
    )
    
    # Provider Selection Logic for LLM
    llm_provider = providers.Selector(
        config.llm.backend,
        openai=providers.Singleton(
            OpenAILLMProvider,
            client=openai_client,
            model=config.llm.openai_model
        ),
        mlx=providers.Singleton(
            MLXLLMProvider,
            mlx_model=mlx_model
        )
    )
    
    # Provider Selection Logic for Embeddings
    embedding_provider = providers.Selector(
        config.llm.backend,
        openai=providers.Singleton(
            OpenAIEmbeddingProvider,
            client=openai_client
        ),
        mlx=providers.Singleton(
            MLXEmbeddingProvider,
            mlx_model=mlx_model
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
