"""Dashboard report generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..core import MiniVectorDB
from .metrics import CollectionSnapshot, DashboardSnapshot, ShardSnapshot


def build_snapshot(db: MiniVectorDB, shard_ids: Iterable[str] | None = None) -> DashboardSnapshot:
    collections = []
    for collection in db.collections.values():
        collections.append(
            CollectionSnapshot(
                name=collection.name,
                vectors=collection.count(),
                segments=len(collection.segments),
                memory_mb=max(collection.count() * collection.dimension * 4 / (1024 * 1024), 0.0),
                latency_ms=collection.statistics.average_latency_ms,
                qps=(
                    (collection.statistics.searches * 1000.0)
                    / max(sum(collection.statistics.latencies_ms), 1e-9)
                )
                if collection.statistics.latencies_ms
                else 0.0,
                index_type=collection.planner.choose_strategy(collection.count()),
                recall=1.0,
            )
        )

    shards = []
    if shard_ids is not None:
        for shard_id in shard_ids:
            shards.append(ShardSnapshot(shard_id=shard_id, collections=list(db.collections)))

    return DashboardSnapshot(collections=collections, shards=shards)


def to_json(db: MiniVectorDB, path: str | Path | None = None) -> str:
    payload = build_snapshot(db).to_dict()
    text = json.dumps(payload, indent=2)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
