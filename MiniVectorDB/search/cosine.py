"""Cosine similarity."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left)) or 1.0
    right_norm = sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


class CosineDistance:
    def score(self, left: Iterable[float], right: Iterable[float]) -> float:
        return cosine_similarity(list(left), list(right))

