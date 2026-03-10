import logging
from typing import List

import mlx.core as mx

from app.infra.llm_connector.mlx_base import MLXModelBase

logger = logging.getLogger("app.llm_connector")


class MLXEmbeddingModel(MLXModelBase):
    """
    Wrapper around a locally-stored MLX embedding model.

    Produces L2-normalised dense embeddings by running the transformer
    backbone (without the LM head) and taking the last-token hidden state.

    Example::

        model = MLXEmbeddingModel("/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        vector = model.embed("What is domain-driven design?")

    The model is loaded lazily on the first call to :meth:`embed` and then
    cached in memory for the lifetime of the process.
    """

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path: Local path (or HF hub name) to the MLX embedding model
                        directory.  Docker-style ``/models/...`` paths are
                        automatically remapped to the project-local
                        ``models/...`` directory when running outside Docker.
        """
        self._model_path = model_path

    def embed(self, text: str) -> List[float]:
        """
        Embed *text* and return a normalised float vector.

        The transformer backbone is called directly (no LM head) so that the
        output represents the hidden state rather than next-token logits.  The
        last-token position is used as the sentence embedding, which matches
        the convention for decoder-only embedding models such as Qwen3-
        Embedding.

        Args:
            text: The text to embed.

        Returns:
            A ``List[float]`` of length equal to the model's hidden dimension,
            normalised to unit L2 norm.
        """
        model, tokenizer = self._load_model(self._model_path)
        tokens = tokenizer.encode(text, return_tensors="mlx")
        hidden = model.model(tokens)          # (1, seq_len, hidden_dim)
        last = hidden[0, -1, :]              # last-token hidden state
        norm = mx.sqrt((last * last).sum())
        normalised = last / norm
        mx.eval(normalised)
        return normalised.tolist()



