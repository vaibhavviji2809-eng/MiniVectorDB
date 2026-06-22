"""Hybrid keyword + vector search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import Counter
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
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, k1: float = 1.5, b: float = 0.75) -> None:
        self.alpha = alpha
        self.beta = beta
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer()

    def _corpus_stats(self, corpus: List[VectorRecord]) -> tuple[Counter, float]:
        doc_freq = Counter()
        total_length = 0
        for doc in corpus:
            tokens = set(
                self.tokenizer.tokenize(
                    " ".join([doc.text or "", " ".join(f"{k} {v}" for k, v in doc.metadata.items())])
                )
            )
            total_length += len(tokens)
            for token in tokens:
                doc_freq[token] += 1
        avg_doc_len = total_length / max(len(corpus), 1)
        return doc_freq, avg_doc_len

    def keyword_score(
        self,
        query: str,
        record: VectorRecord,
        corpus: List[VectorRecord],
        corpus_stats: tuple[Counter, float],
    ) -> float:
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return 0.0
        searchable = " ".join(
            [record.text or "", " ".join(f"{k} {v}" for k, v in record.metadata.items())]
        )
        record_tokens = self.tokenizer.tokenize(searchable)
        if not record_tokens:
            return 0.0
        doc_freq, avg_doc_len = corpus_stats
        doc_len = len(record_tokens)
        term_freq = Counter(record_tokens)
        score = 0.0
        for token in query_tokens:
            df = doc_freq.get(token, 0)
            if not df:
                continue
            idf = math.log((len(corpus) - df + 0.5) / (df + 0.5) + 1.0)
            tf = term_freq.get(token, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(avg_doc_len, 1e-9)))
            score += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-9))
        return score

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        records: List[VectorRecord],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[RankedRecord]:
        corpus_stats = self._corpus_stats(records)
        ranked = []
        for record in records:
            semantic = _metric_score(metric, query_vector, record.vector)
            keyword = self.keyword_score(query, record, records, corpus_stats)
            score = self.alpha * semantic + self.beta * keyword
            ranked.append(RankedRecord(record=record, score=score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
