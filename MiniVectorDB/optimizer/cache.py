"""Simple LRU cache implementations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class LRUCache(Generic[K, V]):
    capacity: int = 128
    data: "OrderedDict[K, V]" = field(default_factory=OrderedDict)

    def get(self, key: K) -> Optional[V]:
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key: K, value: V) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


EmbeddingCache = LRUCache
SearchCache = LRUCache
MetadataCache = LRUCache
ResultCache = LRUCache

