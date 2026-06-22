"""Graph utilities for navigable vector search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..search.cosine import cosine_similarity
from ..storage.page import VectorRecord


@dataclass
class GraphIndex:
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def build(self, records: List[VectorRecord], fanout: int = 8) -> "GraphIndex":
        self.adjacency = {record.id: [] for record in records}
        for record in records:
            ranked = sorted(
                (other for other in records if other.id != record.id),
                key=lambda other: cosine_similarity(record.vector, other.vector),
                reverse=True,
            )
            self.adjacency[record.id] = [other.id for other in ranked[:fanout]]
        return self

