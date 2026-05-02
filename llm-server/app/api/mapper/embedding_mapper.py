from app.api.dto.embedding_request import EmbeddingRequest
from app.services.commands.generate_embedding_command import GenerateEmbeddingCommand

class EmbeddingMapper:
    """Mapper to convert Embedding DTOs to Service Commands."""
    
    @staticmethod
    def to_generate_embedding_command(request: EmbeddingRequest) -> GenerateEmbeddingCommand:
        texts = [request.input] if isinstance(request.input, str) else request.input
        return GenerateEmbeddingCommand(texts=texts)
