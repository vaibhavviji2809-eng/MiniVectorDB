"""Product quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


def _chunk(vector: Sequence[float], parts: int) -> List[List[float]]:
    size = max(1, len(vector) // parts)
    return [list(vector[index : index + size]) for index in range(0, len(vector), size)]


@dataclass
class ProductQuantizer:
    parts: int = 4
    codebooks: List[Dict[int, List[float]]] = field(default_factory=list)

    def fit(self, vectors: List[Sequence[float]]) -> "ProductQuantizer":
        self.codebooks = []
        if not vectors:
            return self
        dimension = len(vectors[0])
        chunk_size = max(1, dimension // self.parts)
        for index in range(self.parts):
            start = index * chunk_size
            end = dimension if index == self.parts - 1 else start + chunk_size
            means: Dict[int, List[float]] = {}
            for code, vector in enumerate(vectors):
                means[code] = list(vector[start:end])
            self.codebooks.append(means)
        return self

    def encode(self, vector: Sequence[float]) -> List[int]:
        chunks = _chunk(vector, self.parts)
        codes: List[int] = []
        for part, chunk in enumerate(chunks):
            candidates = self.codebooks[part] if part < len(self.codebooks) else {}
            if not candidates:
                codes.append(0)
                continue
            codes.append(
                min(
                    candidates,
                    key=lambda code: sum((a - b) ** 2 for a, b in zip(chunk, candidates[code])),
                )
            )
        return codes

    def decode(self, codes: List[int]) -> List[float]:
        decoded: List[float] = []
        for part, code in enumerate(codes):
            decoded.extend(self.codebooks[part].get(code, []))
        return decoded
