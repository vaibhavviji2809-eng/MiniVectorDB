"""Page-level storage primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VectorRecord":
        return cls(
            id=str(payload["id"]),
            vector=list(payload["vector"]),
            metadata=dict(payload.get("metadata", {})),
            text=payload.get("text"),
        )


@dataclass
class Page:
    capacity: int = 128
    records: List[VectorRecord] = field(default_factory=list)

    def is_full(self) -> bool:
        return len(self.records) >= self.capacity

    def insert(self, record: VectorRecord) -> None:
        if self.is_full():
            raise ValueError("page is full")
        self.records.append(record)

    def delete(self, record_id: str) -> bool:
        original = len(self.records)
        self.records = [record for record in self.records if record.id != record_id]
        return len(self.records) != original

    def find(self, record_id: str) -> Optional[VectorRecord]:
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def all_records(self) -> List[VectorRecord]:
        return list(self.records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Page":
        return cls(
            capacity=payload.get("capacity", 128),
            records=[VectorRecord.from_dict(record) for record in payload["records"]],
        )

