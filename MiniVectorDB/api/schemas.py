"""Request and response schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CreateCollectionRequest:
    name: str
    dimension: int = 64
    page_capacity: int = 128
    pages_per_segment: int = 16


@dataclass
class InsertRequest:
    collection: str
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SearchRequest:
    collection: str
    query: Optional[str] = None
    query_vector: Optional[List[float]] = None
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    metric: str = "cosine"
    use_hybrid: bool = False


@dataclass
class DeleteRequest:
    collection: str
    id: str


def to_dict(payload: Any) -> Dict[str, Any]:
    return asdict(payload)

