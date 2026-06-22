"""Brute force vector search baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..search.cosine import cosine_similarity
from ..search.dot import dot_product
from ..search.l2 import l2_distance
from ..storage.page import VectorRecord


def _score(metric: str, query: Sequence[float], vector: Sequence[float]) -> float:
    if metric == "dot":
        return dot_product(query, vector)
    if metric == "l2":
        return -l2_distance(query, vector)
    return cosine_similarity(query, vector)


@dataclass
class ScoredRecord:
    record: VectorRecord
    score: float

    def to_dict(self) -> dict:
        payload = self.record.to_dict()
        payload["score"] = self.score
        return payload


class BruteForceIndex:
    def search(
        self,
        query_vector: Sequence[float],
        records: List[VectorRecord],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[ScoredRecord]:
        scored = [
            ScoredRecord(record=record, score=_score(metric, query_vector, record.vector))
            for record in records
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

