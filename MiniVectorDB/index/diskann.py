"""DiskANN-style graph index with disk-backed adjacency and memory cache."""

from __future__ import annotations

import json
import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
class DiskANNIndex:
    """Simplified DiskANN-style index."""

    graph_degree: int = 32
    cache_size: int = 256
    directory: Optional[Path] = None
    entry_point: Optional[str] = None
    graph: Dict[str, List[str]] = field(default_factory=dict)
    records: Dict[str, VectorRecord] = field(default_factory=dict)
    cache: "OrderedDict[str, VectorRecord]" = field(default_factory=OrderedDict)

    def build(self, records: List[VectorRecord], directory: Optional[str | Path] = None) -> "DiskANNIndex":
        if directory is not None:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {record.id: record for record in records}
        if not records:
            self.graph = {}
            self.entry_point = None
            return self

        self.entry_point = records[0].id
        self.graph = {}
        for record in records:
            ranked = sorted(
                (other for other in records if other.id != record.id),
                key=lambda other: _score("cosine", record.vector, other.vector),
                reverse=True,
            )
            self.graph[record.id] = [other.id for other in ranked[: self.graph_degree]]
        self._persist()
        return self

    def _persist(self) -> None:
        if self.directory is None:
            return
        payload = {
            "entry_point": self.entry_point,
            "graph": self.graph,
            "records": [record.to_dict() for record in self.records.values()],
        }
        (self.directory / "diskann_graph.json").write_text(json.dumps(payload), encoding="utf-8")

    def _load_record(self, record_id: str) -> Optional[VectorRecord]:
        if record_id in self.cache:
            self.cache.move_to_end(record_id)
            return self.cache[record_id]
        record = self.records.get(record_id)
        if record is None and self.directory is not None:
            path = self.directory / "diskann_graph.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                for item in payload.get("records", []):
                    if item["id"] == record_id:
                        record = VectorRecord.from_dict(item)
                        self.records[record_id] = record
                        break
        if record is None:
            return None
        self.cache[record_id] = record
        self.cache.move_to_end(record_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return record

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        metric: str = "cosine",
        ef_search: int = 64,
    ) -> List[VectorRecord]:
        if not self.entry_point:
            return []

        entry = self._load_record(self.entry_point)
        if entry is None:
            return []

        visited = {entry.id}
        candidates: List[Tuple[float, str]] = [(-_score(metric, query_vector, entry.vector), entry.id)]
        frontier: List[Tuple[float, str]] = []

        while candidates and len(frontier) < ef_search:
            neg_score, node_id = heapq.heappop(candidates)
            score = -neg_score
            frontier.append((score, node_id))
            for neighbor_id in self.graph.get(node_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                record = self._load_record(neighbor_id)
                if record is None:
                    continue
                neighbor_score = _score(metric, query_vector, record.vector)
                heapq.heappush(candidates, (-neighbor_score, neighbor_id))

        frontier.sort(key=lambda item: item[0], reverse=True)
        ordered_ids = [node_id for _, node_id in frontier[:top_k]]
        ranked_records = [self._load_record(node_id) for node_id in ordered_ids]
        return [record for record in ranked_records if record is not None]

