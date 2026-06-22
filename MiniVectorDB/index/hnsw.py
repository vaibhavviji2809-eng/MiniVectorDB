"""HNSW-style navigable graph with layered insertion and pruning."""

from __future__ import annotations

import heapq
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def _stable_level(node_id: str, max_level: int = 8) -> int:
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    level = 0
    while level < max_level and value & 0x3 == 0:
        level += 1
        value >>= 2
    return level


@dataclass
class HNSWIndex:
    """A simplified but dynamically maintainable HNSW graph."""

    m: int = 8
    ef_construction: int = 64
    ef_search: int = 32
    max_level_cap: int = 8
    entry_point: Optional[str] = None
    max_level: int = -1
    layers: Dict[int, Dict[str, Set[str]]] = field(default_factory=dict)
    node_levels: Dict[str, int] = field(default_factory=dict)
    records: Dict[str, VectorRecord] = field(default_factory=dict)

    def clear(self) -> None:
        self.entry_point = None
        self.max_level = -1
        self.layers.clear()
        self.node_levels.clear()
        self.records.clear()

    def build(self, records: List[VectorRecord]) -> "HNSWIndex":
        self.clear()
        for record in records:
            self.add_record(record)
        return self

    def add_record(self, record: VectorRecord) -> None:
        if record.id in self.records:
            self.remove_record(record.id)

        self.records[record.id] = record
        node_level = _stable_level(record.id, self.max_level_cap)
        self.node_levels[record.id] = node_level
        self.max_level = max(self.max_level, node_level)

        for level in range(node_level + 1):
            self.layers.setdefault(level, {})
            self.layers[level].setdefault(record.id, set())

        if self.entry_point is None:
            self.entry_point = record.id
            return

        if self.entry_point not in self.records:
            self.entry_point = record.id
            return

        current = self.entry_point
        if self.max_level > node_level:
            for level in range(self.max_level, node_level, -1):
                current = self._greedy_descent(record.vector, current, level)

        for level in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(
                record.vector, current, level, self.ef_construction, "cosine"
            )
            if not candidates:
                continue
            selected = [node_id for _, node_id in candidates[: self.m]]
            self._connect(record.id, selected, level)
            current = selected[0]

        if node_level > self.max_level:
            self.entry_point = record.id
            self.max_level = node_level

    def remove_record(self, record_id: str) -> bool:
        if record_id not in self.records:
            return False

        self.records.pop(record_id, None)
        self.node_levels.pop(record_id, None)
        for level, adjacency in self.layers.items():
            adjacency.pop(record_id, None)
            for neighbors in adjacency.values():
                neighbors.discard(record_id)
        self._prune_all()

        if record_id == self.entry_point:
            self.entry_point = self._choose_entry_point()
        self.max_level = max(self.node_levels.values(), default=-1)
        return True

    def maintain(self) -> None:
        self._prune_all()
        self.max_level = max(self.node_levels.values(), default=-1)
        if self.entry_point not in self.records:
            self.entry_point = self._choose_entry_point()

    def _choose_entry_point(self) -> Optional[str]:
        if not self.node_levels:
            return None
        return max(self.node_levels.items(), key=lambda item: (item[1], item[0]))[0]

    def _connect(self, node_id: str, neighbors: List[str], level: int) -> None:
        adjacency = self.layers.setdefault(level, {})
        adjacency.setdefault(node_id, set()).update(neighbors)
        for neighbor_id in neighbors:
            if neighbor_id not in self.records:
                continue
            self.layers.setdefault(level, {}).setdefault(neighbor_id, set()).add(node_id)
        self._prune_node(node_id, level)
        for neighbor_id in neighbors:
            self._prune_node(neighbor_id, level)

    def _prune_node(self, node_id: str, level: int) -> None:
        adjacency = self.layers.get(level, {})
        neighbors = adjacency.get(node_id)
        if not neighbors:
            return
        keep = self.m if level == 0 else max(self.m, self.m // 2 + 1)
        ranked = sorted(
            neighbors,
            key=lambda other_id: _score(
                "cosine", self.records[node_id].vector, self.records[other_id].vector
            ),
            reverse=True,
        )
        adjacency[node_id] = set(ranked[:keep])

    def _prune_all(self) -> None:
        for level in list(self.layers):
            for node_id in list(self.layers[level]):
                self._prune_node(node_id, level)

    def _greedy_descent(
        self, query_vector: Sequence[float], start: str, level: int, metric: str
    ) -> str:
        current = start
        current_score = _score(metric, query_vector, self.records[current].vector)
        improved = True
        while improved:
            improved = False
            for neighbor_id in self.layers.get(level, {}).get(current, set()):
                candidate_score = _score(metric, query_vector, self.records[neighbor_id].vector)
                if candidate_score > current_score:
                    current = neighbor_id
                    current_score = candidate_score
                    improved = True
        return current

    def _search_layer(
        self,
        query_vector: Sequence[float],
        entry_point: str,
        level: int,
        ef: int,
        metric: str,
    ) -> List[Tuple[float, str]]:
        visited: Set[str] = {entry_point}
        candidates: List[Tuple[float, str]] = []
        explored: List[Tuple[float, str]] = []
        entry_score = _score(metric, query_vector, self.records[entry_point].vector)
        heapq.heappush(candidates, (-entry_score, entry_point))

        while candidates and len(explored) < ef:
            neg_score, node_id = heapq.heappop(candidates)
            score = -neg_score
            explored.append((score, node_id))
            for neighbor_id in self.layers.get(level, {}).get(node_id, set()):
                if neighbor_id in visited or neighbor_id not in self.records:
                    continue
                visited.add(neighbor_id)
                neighbor_score = _score(metric, query_vector, self.records[neighbor_id].vector)
                heapq.heappush(candidates, (-neighbor_score, neighbor_id))

        explored.sort(key=lambda item: item[0], reverse=True)
        return explored

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metric: str = "cosine",
        ef_search: Optional[int] = None,
    ) -> List[VectorRecord]:
        if not self.entry_point or not self.records:
            return []

        ef = ef_search or self.ef_search
        current = self.entry_point
        for level in range(self.max_level, 0, -1):
            current = self._greedy_descent(query_vector, current, level, metric)

        best = self._search_layer(query_vector, current, 0, ef=max(ef, top_k * 2), metric=metric)
        ranked_ids = [node_id for _, node_id in best[:top_k]]
        if len(ranked_ids) < top_k:
            # Fall back to a broader scan if the graph is sparse.
            fallback = sorted(
                self.records.values(),
                key=lambda record: _score(metric, query_vector, record.vector),
                reverse=True,
            )
            return fallback[:top_k]

        scored = [
            (node_id, _score(metric, query_vector, self.records[node_id].vector))
            for node_id in ranked_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [self.records[node_id] for node_id, _ in scored[:top_k]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_level_cap": self.max_level_cap,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "node_levels": dict(self.node_levels),
            "layers": {
                str(level): {node_id: sorted(neighbors) for node_id, neighbors in adjacency.items()}
                for level, adjacency in self.layers.items()
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Dict[str, object], records: Optional[List[VectorRecord]] = None
    ) -> "HNSWIndex":
        index = cls(
            m=int(payload.get("m", 8)),
            ef_construction=int(payload.get("ef_construction", 64)),
            ef_search=int(payload.get("ef_search", 32)),
            max_level_cap=int(payload.get("max_level_cap", 8)),
        )
        index.entry_point = payload.get("entry_point") or None
        index.max_level = int(payload.get("max_level", -1))
        index.node_levels = {
            str(node_id): int(level) for node_id, level in dict(payload.get("node_levels", {})).items()
        }
        index.layers = {}
        for level_str, adjacency in dict(payload.get("layers", {})).items():
            level = int(level_str)
            index.layers[level] = {
                str(node_id): set(map(str, neighbors))
                for node_id, neighbors in dict(adjacency).items()
            }
        if records:
            index.records = {record.id: record for record in records}
        return index
