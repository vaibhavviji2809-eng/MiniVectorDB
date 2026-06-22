"""Scalar quantization for compact numeric storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class ScalarQuantizer:
    bits: int = 8
    min_values: List[float] = field(default_factory=list)
    max_values: List[float] = field(default_factory=list)

    @property
    def levels(self) -> int:
        return max(2, 2**self.bits)

    def fit(self, vectors: List[Sequence[float]]) -> "ScalarQuantizer":
        if not vectors:
            self.min_values = []
            self.max_values = []
            return self
        dimension = len(vectors[0])
        self.min_values = [min(vector[index] for vector in vectors) for index in range(dimension)]
        self.max_values = [max(vector[index] for vector in vectors) for index in range(dimension)]
        return self

    def encode(self, vector: Sequence[float]) -> List[int]:
        codes: List[int] = []
        for index, value in enumerate(vector):
            min_v = self.min_values[index] if index < len(self.min_values) else value
            max_v = self.max_values[index] if index < len(self.max_values) else value
            span = max(max_v - min_v, 1e-9)
            normalized = (value - min_v) / span
            codes.append(int(round(normalized * (self.levels - 1))))
        return codes

    def decode(self, codes: Sequence[int]) -> List[float]:
        decoded: List[float] = []
        for index, code in enumerate(codes):
            min_v = self.min_values[index] if index < len(self.min_values) else 0.0
            max_v = self.max_values[index] if index < len(self.max_values) else 0.0
            span = max(max_v - min_v, 1e-9)
            decoded.append(min_v + (code / max(self.levels - 1, 1)) * span)
        return decoded

