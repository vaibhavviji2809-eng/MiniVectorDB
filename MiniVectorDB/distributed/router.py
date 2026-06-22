"""Query router across shards."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .shard import Shard


@dataclass
class Router:
    shards: List[Shard] = field(default_factory=list)

    def add_shard(self, shard: Shard) -> None:
        self.shards.append(shard)

    def _target_shards(self, key: str | None = None) -> List[Shard]:
        if not self.shards:
            return []
        if key is None:
            return list(self.shards)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        shard_index = int.from_bytes(digest[:4], "big") % len(self.shards)
        return [self.shards[shard_index]]

    def insert(self, collection: str, key: str | None = None, **payload: Any):
        targets = self._target_shards(key)
        if not targets:
            raise ValueError("no shards configured")
        return targets[0].insert(collection, **payload)

    def search(self, collection: str, key: str | None = None, **payload: Any):
        targets = self._target_shards(key)
        results = []
        for shard in targets:
            results.extend(shard.search(collection, **payload))
        return self._merge_results(results, payload.get("top_k", 5))

    async def search_async(self, collection: str, key: str | None = None, **payload: Any):
        targets = self._target_shards(key)
        if not targets:
            return []
        shard_results = await asyncio.gather(
            *[shard.search_async(collection, **payload) for shard in targets]
        )
        merged = [item for batch in shard_results for item in batch]
        return self._merge_results(merged, payload.get("top_k", 5))

    def _merge_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for item in results:
            record_id = str(item.get("id"))
            current = seen.get(record_id)
            if current is None or item.get("score", 0.0) > current.get("score", 0.0):
                seen[record_id] = item
        merged = sorted(seen.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        return merged[:top_k]
