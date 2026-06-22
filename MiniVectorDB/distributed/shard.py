"""Shard abstraction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..core import MiniVectorDB


@dataclass
class Shard:
    shard_id: str
    db: MiniVectorDB = field(default_factory=MiniVectorDB)

    def insert(self, collection: str, **payload: Any):
        return self.db.insert(collection, **payload)

    def search(self, collection: str, **payload: Any):
        return self.db.search(collection, **payload)

    async def search_async(self, collection: str, **payload: Any):
        return await asyncio.to_thread(self.search, collection, **payload)

    def delete(self, collection: str, record_id: str) -> bool:
        return self.db.delete(collection, record_id)

    async def insert_async(self, collection: str, **payload: Any):
        return await asyncio.to_thread(self.insert, collection, **payload)
