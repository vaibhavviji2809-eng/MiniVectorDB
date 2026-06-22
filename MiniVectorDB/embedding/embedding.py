"""Embedding engine built from tokenizer and encoder."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from .encoder import Encoder
from .tokenizer import Tokenizer


class EmbeddingEngine:
    def __init__(self, dimension: int = 64) -> None:
        self.tokenizer = Tokenizer()
        self.encoder = Encoder(dimension=dimension)

    @lru_cache(maxsize=1024)
    def embed_text(self, text: str) -> List[float]:
        return self.encoder.encode(self.tokenizer.tokenize(text))

    def embed(self, text: str) -> List[float]:
        return self.embed_text(text)

