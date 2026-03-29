"""
LM Studio embedding model — OpenAI-compatible HTTP API adapter.

``LMStudioEmbeddingModel`` mirrors the ``embed(text) -> List[float]``
contract of ``MLXEmbeddingModel`` so that ``LLMService`` can swap between
local and remote backends without any inference-site changes.
"""

from __future__ import annotations

import logging
from typing import List

import openai

logger = logging.getLogger("app.llm_connector")


class LMStudioEmbeddingModel:
    """
    Embedding model backed by LM Studio's OpenAI-compatible ``/v1/embeddings``
    endpoint.

    Exposes the same ``embed(text) -> List[float]`` interface as
    ``MLXEmbeddingModel`` so that ``LLMService`` treats both backends
    identically.

    Example::

        model = LMStudioEmbeddingModel(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="text-embedding-nomic-embed-text-v1.5",
        )
        vector = model.embed("What is domain-driven design?")
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        """
        Args:
            base_url: LM Studio API base URL, e.g. ``http://localhost:1234/v1``.
            api_key:  LM Studio API key (any non-empty string is accepted).
            model:    Embedding model identifier as listed in LM Studio.
        """
        self._model = model
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        logger.info(
            f"[LMStudioEmbeddingModel] Configured for model='{model}' at {base_url}"
        )

    def embed(self, text: str) -> List[float]:
        """
        Embed *text* and return a float vector.

        The vector is returned as-is from LM Studio; most embedding models
        served by LM Studio already produce normalised vectors, but callers
        that require strict unit-norm output should normalise on their side.

        Args:
            text: The text to embed.

        Returns:
            A ``List[float]`` embedding vector.
        """
        logger.info(f"[LMStudioEmbeddingModel] Embedding text (model='{self._model}')")
        response = self._client.embeddings.create(input=text, model=self._model)
        vector: List[float] = response.data[0].embedding
        logger.info("[LMStudioEmbeddingModel] Embedding received")
        return vector
