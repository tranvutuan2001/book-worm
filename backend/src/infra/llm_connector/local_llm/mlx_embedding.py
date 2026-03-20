import logging
from typing import Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load


@runtime_checkable
class _Tokenizer(Protocol):
    """Structural interface for the tokenizer returned by ``mlx_lm.load``."""

    def encode(self, text: str, **kwargs: object) -> mx.array: ...


# (backbone, tokenizer) pair as returned by mlx_lm.load
_ModelPair = tuple[nn.Module, _Tokenizer]

logger = logging.getLogger("app.llm_connector")


class MLXEmbeddingModel:
    """
    Wrapper around a locally-stored MLX embedding model.

    Produces L2-normalised dense embeddings by running the transformer
    backbone (without the LM head) and taking the last-token hidden state.

    Example::

        model = MLXEmbeddingModel("/models/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
        vector = model.embed("What is domain-driven design?")

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
        logger.info(f"Loading MLX embedding model from: {model_path}")
        self._model_pair: _ModelPair = mlx_load(model_path)
        logger.info(f"MLX embedding model loaded successfully: {model_path}")

    def embed(self, text: str) -> list[float]:
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
        model, tokenizer = self._model_pair
        tokens = tokenizer.encode(text, return_tensors="mlx")
        hidden = model.model(tokens)          # (1, seq_len, hidden_dim)
        last = hidden[0, -1, :]              # last-token hidden state
        norm = mx.sqrt((last * last).sum())
        normalised = last / norm
        mx.eval(normalised)
        return normalised.tolist()



