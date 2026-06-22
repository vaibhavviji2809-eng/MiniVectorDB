"""Inverted file index with simple k-means clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..search.cosine import cosine_similarity
from ..search.dot import dot_product
from ..search.l2 import l2_distance
from ..storage.page import VectorRecord


def _distance(metric: str, left: Sequence[float], right: Sequence[float]) -> float:
    if metric == "dot":
        return -dot_product(left, right)
    if metric == "l2":
        return l2_distance(left, right)
    return 1.0 - cosine_similarity(left, right)


def _mean(vectors: List[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dimension = len(vectors[0])
    summed = [0.0] * dimension
    for vector in vectors:
        for index, value in enumerate(vector):
            summed[index] += value
    return [value / len(vectors) for value in summed]


@dataclass
class IVFIndex:
    n_clusters: int = 8
    centroids: List[List[float]] = field(default_factory=list)
    posting_lists: Dict[int, List[VectorRecord]] = field(default_factory=dict)

    def build(self, records: List[VectorRecord]) -> "IVFIndex":
        if not records:
            self.centroids = []
            self.posting_lists = {}
            return self
        self.n_clusters = max(1, min(self.n_clusters, len(records)))
        self.centroids = [list(record.vector) for record in records[: self.n_clusters]]
        for _ in range(5):
            clusters: Dict[int, List[VectorRecord]] = {i: [] for i in range(self.n_clusters)}
            for record in records:
                cluster_id = min(
                    range(self.n_clusters),
                    key=lambda index: _distance("cosine", record.vector, self.centroids[index]),
                )
                clusters[cluster_id].append(record)
            self.centroids = [
                _mean([record.vector for record in clusters[index]]) or self.centroids[index]
                for index in range(self.n_clusters)
            ]
        self.posting_lists = {index: [] for index in range(self.n_clusters)}
        for record in records:
            cluster_id = min(
                range(self.n_clusters),
                key=lambda index: _distance("cosine", record.vector, self.centroids[index]),
            )
            self.posting_lists.setdefault(cluster_id, []).append(record)
        return self

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[VectorRecord]:
        if not self.centroids:
            return []
        cluster_id = min(
            range(len(self.centroids)),
            key=lambda index: _distance(metric, query_vector, self.centroids[index]),
        )
        candidates = self.posting_lists.get(cluster_id, [])
        scored = sorted(
            candidates,
            key=lambda record: -_distance(metric, query_vector, record.vector),
        )
        return scored[:top_k]
