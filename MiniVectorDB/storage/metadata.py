"""Metadata storage and filter indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .page import VectorRecord


def _match_condition(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        for op, expected in condition.items():
            if op == "gt" and not (value > expected):
                return False
            if op == "gte" and not (value >= expected):
                return False
            if op == "lt" and not (value < expected):
                return False
            if op == "lte" and not (value <= expected):
                return False
            if op == "ne" and not (value != expected):
                return False
            if op == "in" and value not in expected:
                return False
        return True
    return value == condition


@dataclass
class HashIndex:
    mapping: Dict[str, Dict[Any, Set[str]]] = field(default_factory=dict)

    def add(self, record: VectorRecord) -> None:
        for key, value in record.metadata.items():
            self.mapping.setdefault(key, {}).setdefault(value, set()).add(record.id)

    def remove(self, record: VectorRecord) -> None:
        for key, value in record.metadata.items():
            bucket = self.mapping.get(key, {}).get(value)
            if bucket is not None:
                bucket.discard(record.id)

    def query(self, field: str, value: Any) -> Set[str]:
        return set(self.mapping.get(field, {}).get(value, set()))


@dataclass
class BitmapIndex:
    mapping: Dict[str, Dict[Any, Set[str]]] = field(default_factory=dict)

    def add(self, record: VectorRecord) -> None:
        for key, value in record.metadata.items():
            self.mapping.setdefault(key, {}).setdefault(value, set()).add(record.id)

    def remove(self, record: VectorRecord) -> None:
        for key, value in record.metadata.items():
            bucket = self.mapping.get(key, {}).get(value)
            if bucket is not None:
                bucket.discard(record.id)

    def query(self, field: str, value: Any) -> Set[str]:
        return set(self.mapping.get(field, {}).get(value, set()))


@dataclass
class MetadataStore:
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hash_index: HashIndex = field(default_factory=HashIndex)
    bitmap_index: BitmapIndex = field(default_factory=BitmapIndex)
    _counter: int = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def add(self, record: VectorRecord) -> None:
        self.records[record.id] = dict(record.metadata)
        self.hash_index.add(record)
        self.bitmap_index.add(record)

    def remove(self, record_id: str) -> None:
        metadata = self.records.pop(record_id, None)
        if metadata is None:
            return
        record = VectorRecord(id=record_id, vector=[], metadata=metadata)
        self.hash_index.remove(record)
        self.bitmap_index.remove(record)

    def get(self, record_id: str) -> Dict[str, Any]:
        return dict(self.records.get(record_id, {}))

    def filter_ids(self, filters: Dict[str, Any]) -> Optional[Set[str]]:
        if not filters:
            return None
        candidate_ids: Optional[Set[str]] = None
        for field, condition in filters.items():
            if isinstance(condition, dict) and set(condition).issubset(
                {"gt", "gte", "lt", "lte", "ne", "in"}
            ):
                matches = {
                    record_id
                    for record_id, metadata in self.records.items()
                    if field in metadata and _match_condition(metadata[field], condition)
                }
            else:
                matches = self.hash_index.query(field, condition)
            candidate_ids = matches if candidate_ids is None else candidate_ids & matches
        return candidate_ids or set()

    def to_dict(self) -> Dict[str, Any]:
        return {"records": self.records, "counter": self._counter}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MetadataStore":
        store = cls(records={k: dict(v) for k, v in payload.get("records", {}).items()})
        store._counter = int(payload.get("counter", 0))
        for record_id, metadata in store.records.items():
            record = VectorRecord(id=record_id, vector=[], metadata=metadata)
            store.hash_index.add(record)
            store.bitmap_index.add(record)
        return store

