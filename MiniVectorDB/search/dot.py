"""Dot product distance."""

from __future__ import annotations

from typing import Iterable, Sequence


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class DotProduct:
    def score(self, left: Iterable[float], right: Iterable[float]) -> float:
        return dot_product(list(left), list(right))

