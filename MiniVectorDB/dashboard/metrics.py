"""Snapshot models for dashboard reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CollectionSnapshot:
    name: str
    vectors: int
    segments: int
    memory_mb: float
    latency_ms: float
    qps: float
    index_type: str
    recall: float


@dataclass
class ShardSnapshot:
    shard_id: str
    collections: List[str] = field(default_factory=list)
    vectors: int = 0


@dataclass
class DashboardSnapshot:
    collections: List[CollectionSnapshot] = field(default_factory=list)
    shards: List[ShardSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collections": [asdict(item) for item in self.collections],
            "shards": [asdict(item) for item in self.shards],
        }

