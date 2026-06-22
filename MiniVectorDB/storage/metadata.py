"""Metadata storage and filter indexes."""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
            if op in {"prefix", "starts_with"} and not str(value).startswith(str(expected)):
                return False
        return True
    return value == condition


@dataclass
class BloomFilter:
    size: int = 4096
    hash_functions: int = 4
    bits: List[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bits:
            self.bits = [False] * self.size

    def _hashes(self, value: str) -> List[int]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return [
            int.from_bytes(digest[index : index + 4], "big") % self.size
            for index in range(0, self.hash_functions * 4, 4)
        ]

    def add(self, value: str) -> None:
        for index in self._hashes(value):
            self.bits[index] = True

    def might_contain(self, value: str) -> bool:
        return all(self.bits[index] for index in self._hashes(value))


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
class RangeIndex:
    values: Dict[str, List[Tuple[float, str]]] = field(default_factory=dict)

    def add(self, record: VectorRecord) -> None:
        for field, value in record.metadata.items():
            if isinstance(value, (int, float)):
                bucket = self.values.setdefault(field, [])
                bisect.insort(bucket, (float(value), record.id))

    def remove(self, record: VectorRecord) -> None:
        for field, value in record.metadata.items():
            if not isinstance(value, (int, float)):
                continue
            bucket = self.values.get(field, [])
            try:
                bucket.remove((float(value), record.id))
            except ValueError:
                continue

    def query(self, field: str, condition: Dict[str, Any]) -> Set[str]:
        bucket = self.values.get(field, [])
        if not bucket:
            return set()
        values = [value for value, _ in bucket]
        left = 0
        right = len(bucket)
        if "gt" in condition:
            left = bisect.bisect_right(values, float(condition["gt"]))
        if "gte" in condition:
            left = bisect.bisect_left(values, float(condition["gte"]))
        if "lt" in condition:
            right = bisect.bisect_left(values, float(condition["lt"]))
        if "lte" in condition:
            right = bisect.bisect_right(values, float(condition["lte"]))
        candidates = {record_id for _, record_id in bucket[left:right]}
        if "ne" in condition:
            excluded = {
                record_id for value, record_id in bucket if value == float(condition["ne"])
            }
            candidates -= excluded
        return candidates


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    record_ids: Set[str] = field(default_factory=set)


@dataclass
class TrieIndex:
    roots: Dict[str, TrieNode] = field(default_factory=dict)

    def add(self, record: VectorRecord) -> None:
        for field, value in record.metadata.items():
            if not isinstance(value, str):
                continue
            node = self.roots.setdefault(field, TrieNode())
            node.record_ids.add(record.id)
            for char in value.lower():
                node = node.children.setdefault(char, TrieNode())
                node.record_ids.add(record.id)

    def remove(self, record: VectorRecord) -> None:
        # Removal is approximate for the trie to keep it compact.
        for field, value in record.metadata.items():
            if not isinstance(value, str):
                continue
            node = self.roots.get(field)
            if node is None:
                continue
            node.record_ids.discard(record.id)

    def prefix_query(self, field: str, prefix: str) -> Set[str]:
        node = self.roots.get(field)
        if node is None:
            return set()
        current = node
        for char in prefix.lower():
            current = current.children.get(char)
            if current is None:
                return set()
        return set(current.record_ids)


@dataclass
class MetadataStore:
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hash_index: HashIndex = field(default_factory=HashIndex)
    bitmap_index: BitmapIndex = field(default_factory=BitmapIndex)
    range_index: RangeIndex = field(default_factory=RangeIndex)
    trie_index: TrieIndex = field(default_factory=TrieIndex)
    bloom_filter: BloomFilter = field(default_factory=BloomFilter)
    _counter: int = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def _bloom_key(self, field: str, value: Any) -> str:
        return f"{field}={value}"

    def add(self, record: VectorRecord) -> None:
        self.records[record.id] = dict(record.metadata)
        self.hash_index.add(record)
        self.bitmap_index.add(record)
        self.range_index.add(record)
        self.trie_index.add(record)
        for field, value in record.metadata.items():
            self.bloom_filter.add(self._bloom_key(field, value))

    def remove(self, record_id: str) -> None:
        metadata = self.records.pop(record_id, None)
        if metadata is None:
            return
        record = VectorRecord(id=record_id, vector=[], metadata=metadata)
        self.hash_index.remove(record)
        self.bitmap_index.remove(record)
        self.range_index.remove(record)
        self.trie_index.remove(record)

    def get(self, record_id: str) -> Dict[str, Any]:
        return dict(self.records.get(record_id, {}))

    def _scan_matches(self, field: str, condition: Any) -> Set[str]:
        return {
            record_id
            for record_id, metadata in self.records.items()
            if field in metadata and _match_condition(metadata[field], condition)
        }

    def filter_ids(self, filters: Dict[str, Any]) -> Optional[Set[str]]:
        if not filters:
            return None

        candidate_ids: Optional[Set[str]] = None
        for field, condition in filters.items():
            if isinstance(condition, dict):
                if set(condition).intersection({"gt", "gte", "lt", "lte", "ne"}):
                    matches = self.range_index.query(field, condition)
                elif "prefix" in condition or "starts_with" in condition:
                    prefix = condition.get("prefix", condition.get("starts_with", ""))
                    matches = {
                        record_id
                        for record_id in self.trie_index.prefix_query(field, str(prefix))
                        if str(self.records.get(record_id, {}).get(field, "")).startswith(str(prefix))
                    }
                elif "in" in condition:
                    matches = set()
                    for value in condition["in"]:
                        matches |= self.hash_index.query(field, value)
                else:
                    matches = self._scan_matches(field, condition)
            else:
                key = self._bloom_key(field, condition)
                if self.bloom_filter.might_contain(key):
                    matches = self.hash_index.query(field, condition) or self.bitmap_index.query(
                        field, condition
                    )
                else:
                    matches = set()

            candidate_ids = matches if candidate_ids is None else candidate_ids & matches
            if not candidate_ids:
                return set()
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
            store.range_index.add(record)
            store.trie_index.add(record)
            for field, value in metadata.items():
                store.bloom_filter.add(store._bloom_key(field, value))
        return store
