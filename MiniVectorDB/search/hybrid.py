"""Hybrid keyword + vector search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..embedding.tokenizer import Tokenizer
from ..storage.page import VectorRecord
from .cosine import cosine_similarity
from .dot import dot_product
from .l2 import l2_distance


def _metric_score(metric: str, query: Sequence[float], vector: Sequence[float]) -> float:
    if metric == "dot":
        return dot_product(query, vector)
    if metric == "l2":
        return -l2_distance(query, vector)
    return cosine_similarity(query, vector)


@dataclass
class RankedRecord:
    record: VectorRecord
    score: float

    def to_dict(self) -> dict:
        payload = self.record.to_dict()
        payload["score"] = self.score
        return payload


class HybridSearcher:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        self.alpha = alpha
        self.beta = beta
        self.tokenizer = Tokenizer()

    def keyword_score(self, query: str, record: VectorRecord) -> float:
        query_tokens = set(self.tokenizer.tokenize(query))
        if not query_tokens:
            return 0.0
        searchable = " ".join(
            [record.text or "", " ".join(f"{k} {v}" for k, v in record.metadata.items())]
        )
        record_tokens = self.tokenizer.tokenize(searchable)
        if not record_tokens:
            return 0.0
        overlap = len(query_tokens & set(record_tokens))
        return overlap / max(len(query_tokens), 1)

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        records: List[VectorRecord],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[RankedRecord]:
        ranked = []
        for record in records:
            semantic = _metric_score(metric, query_vector, record.vector)
            keyword = self.keyword_score(query, record)
            score = self.alpha * semantic + self.beta * keyword
            ranked.append(RankedRecord(record=record, score=score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

