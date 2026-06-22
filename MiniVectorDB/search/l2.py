"""Euclidean distance."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


def l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


class L2Distance:
    def score(self, left: Iterable[float], right: Iterable[float]) -> float:
        return -l2_distance(list(left), list(right))

