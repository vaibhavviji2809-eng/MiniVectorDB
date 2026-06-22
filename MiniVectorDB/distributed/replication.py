"""Primary-replica replication helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List

from .shard import Shard


@dataclass
class ReplicaSet:
    primary: Shard
    replicas: List[Shard] = field(default_factory=list)

    def insert(self, collection: str, **payload: Any):
        result = self.primary.insert(collection, **payload)
        for replica in self.replicas:
            replica.insert(collection, **payload)
        return result

    def search(self, collection: str, **payload: Any):
        return self.primary.search(collection, **payload)

    async def search_async(self, collection: str, **payload: Any):
        return await asyncio.to_thread(self.search, collection, **payload)

    def delete(self, collection: str, record_id: str) -> bool:
        deleted = self.primary.delete(collection, record_id)
        for replica in self.replicas:
            replica.delete(collection, record_id)
        return deleted

    async def insert_async(self, collection: str, **payload: Any):
        return await asyncio.to_thread(self.insert, collection, **payload)
