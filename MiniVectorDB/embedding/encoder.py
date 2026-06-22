"""Simple deterministic text encoder."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, List


class Encoder:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def encode(self, tokens: Iterable[str]) -> List[float]:
        counts = Counter(tokens)
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            vector[self._index(token)] += float(count)
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

