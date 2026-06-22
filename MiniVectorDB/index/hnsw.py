"""Simplified HNSW-style navigable small-world graph search."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..search.cosine import cosine_similarity
from ..search.dot import dot_product
from ..search.l2 import l2_distance
from ..storage.page import VectorRecord


def _score(metric: str, left: Sequence[float], right: Sequence[float]) -> float:
    if metric == "dot":
        return dot_product(left, right)
    if metric == "l2":
        return -l2_distance(left, right)
    return cosine_similarity(left, right)


@dataclass
class HNSWIndex:
    max_level: int = 0
    entry_point: Optional[str] = None
    layers: Dict[int, Dict[str, List[str]]] = field(default_factory=dict)
    records: Dict[str, VectorRecord] = field(default_factory=dict)

    def build(self, records: List[VectorRecord]) -> "HNSWIndex":
        self.records = {record.id: record for record in records}
        self.layers = {}
        if not records:
            self.entry_point = None
            self.max_level = 0
            return self

        levels = {record.id: random.randint(0, 3) for record in records}
        self.max_level = max(levels.values())
        self.entry_point = records[0].id
        for level in range(self.max_level + 1):
            self.layers[level] = {record.id: [] for record in records if levels[record.id] >= level}

        for level in range(self.max_level + 1):
            level_records = [record for record in records if levels[record.id] >= level]
            for record in level_records:
                ranked = sorted(
                    (other for other in level_records if other.id != record.id),
                    key=lambda other: _score("cosine", record.vector, other.vector),
                    reverse=True,
                )
                self.layers[level][record.id] = [other.id for other in ranked[:8]]
        return self

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[VectorRecord]:
        if not self.entry_point:
            return []
        current = self.entry_point
        for level in range(self.max_level, 0, -1):
            neighbors = self.layers.get(level, {}).get(current, [])
            best = current
            best_score = _score(metric, query_vector, self.records[current].vector)
            for neighbor_id in neighbors:
                candidate_score = _score(metric, query_vector, self.records[neighbor_id].vector)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best = neighbor_id
            current = best

        visited = {current}
        frontier = [current]
        candidates: List[VectorRecord] = []
        while frontier and len(visited) < 64:
            node = frontier.pop(0)
            candidates.append(self.records[node])
            for neighbor_id in self.layers.get(0, {}).get(node, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    frontier.append(neighbor_id)

        candidates.sort(
            key=lambda record: _score(metric, query_vector, record.vector),
            reverse=True,
        )
        return candidates[:top_k]

