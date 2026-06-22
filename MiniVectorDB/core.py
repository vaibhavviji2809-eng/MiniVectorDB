"""Core collection and database primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .embedding.embedding import EmbeddingEngine
from .index.brute_force import BruteForceIndex
from .index.hnsw import HNSWIndex
from .index.ivf import IVFIndex
from .optimizer.planner import QueryPlanner
from .search.hybrid import HybridSearcher
from .search.cosine import CosineDistance
from .storage.metadata import MetadataStore
from .storage.page import Page, VectorRecord
from .storage.persistence import PersistenceManager
from .storage.segment import Segment


@dataclass
class Collection:
    name: str
    dimension: int = 64
    page_capacity: int = 128
    pages_per_segment: int = 16
    metadata_store: MetadataStore = field(default_factory=MetadataStore)
    segments: List[Segment] = field(default_factory=list)
    embedding_engine: EmbeddingEngine = field(init=False)
    planner: QueryPlanner = field(default_factory=QueryPlanner)

    def __post_init__(self) -> None:
        self.embedding_engine = EmbeddingEngine(dimension=self.dimension)
        if not self.segments:
            self.segments.append(
                Segment(
                    name=f"{self.name}:segment-0",
                    page_capacity=self.page_capacity,
                    max_pages=self.pages_per_segment,
                )
            )

    def _active_segment(self) -> Segment:
        segment = self.segments[-1]
        if segment.is_full():
            segment = Segment(
                name=f"{self.name}:segment-{len(self.segments)}",
                page_capacity=self.page_capacity,
                max_pages=self.pages_per_segment,
            )
            self.segments.append(segment)
        return segment

    def _all_records(self) -> List[VectorRecord]:
        records: List[VectorRecord] = []
        for segment in self.segments:
            records.extend(segment.all_records())
        return records

    def count(self) -> int:
        return len(self._all_records())

    def insert(
        self,
        vector: Optional[Iterable[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> VectorRecord:
        if vector is None:
            if text is None:
                raise ValueError("insert requires either vector or text")
            vector = self.embedding_engine.embed(text)
        record = VectorRecord(
            id=str(record_id or self.metadata_store.next_id()),
            vector=list(vector),
            metadata=dict(metadata or {}),
            text=text,
        )
        self._active_segment().insert(record)
        self.metadata_store.add(record)
        return record

    def delete(self, record_id: str) -> bool:
        deleted = False
        for segment in self.segments:
            deleted = segment.delete(record_id) or deleted
        if deleted:
            self.metadata_store.remove(record_id)
        return deleted

    def search(
        self,
        query: Optional[str] = None,
        query_vector: Optional[Iterable[float]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        records = self._all_records()
        if not records:
            return []

        filtered_ids = self.metadata_store.filter_ids(filters or {})
        if filtered_ids is not None:
            records = [record for record in records if record.id in filtered_ids]
        if not records:
            return []

        if query_vector is None:
            if query is None:
                raise ValueError("search requires query or query_vector")
            query_vector = self.embedding_engine.embed(query)

        strategy = self.planner.choose_strategy(len(records))
        if use_hybrid:
            ranked = HybridSearcher().search(
                query=query or "",
                query_vector=list(query_vector),
                records=records,
                top_k=top_k,
                metric=metric,
            )
        elif strategy == "ivf":
            ranked = IVFIndex().build(records).search(
                list(query_vector), top_k=top_k, metric=metric
            )
        elif strategy == "hnsw":
            ranked = HNSWIndex().build(records).search(
                list(query_vector), top_k=top_k, metric=metric
            )
        else:
            ranked = BruteForceIndex().search(
                list(query_vector), records=records, top_k=top_k, metric=metric
            )
        return [entry.to_dict() for entry in ranked]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "page_capacity": self.page_capacity,
            "pages_per_segment": self.pages_per_segment,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata_store": self.metadata_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Collection":
        collection = cls(
            name=payload["name"],
            dimension=payload.get("dimension", 64),
            page_capacity=payload.get("page_capacity", 128),
            pages_per_segment=payload.get("pages_per_segment", 16),
        )
        collection.segments = [Segment.from_dict(item) for item in payload["segments"]]
        collection.metadata_store = MetadataStore.from_dict(payload["metadata_store"])
        collection.embedding_engine = EmbeddingEngine(dimension=collection.dimension)
        return collection


@dataclass
class MiniVectorDB:
    collections: Dict[str, Collection] = field(default_factory=dict)

    def create_collection(
        self,
        name: str,
        dimension: int = 64,
        page_capacity: int = 128,
        pages_per_segment: int = 16,
    ) -> Collection:
        collection = Collection(
            name=name,
            dimension=dimension,
            page_capacity=page_capacity,
            pages_per_segment=pages_per_segment,
        )
        self.collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        if name not in self.collections:
            raise KeyError(f"collection '{name}' not found")
        return self.collections[name]

    def insert(
        self,
        collection_name: str,
        vector: Optional[Iterable[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> VectorRecord:
        return self.get_collection(collection_name).insert(
            vector=vector, metadata=metadata, text=text, record_id=record_id
        )

    def search(
        self,
        collection_name: str,
        query: Optional[str] = None,
        query_vector: Optional[Iterable[float]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.get_collection(collection_name).search(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            metric=metric,
            use_hybrid=use_hybrid,
        )

    def delete(self, collection_name: str, record_id: str) -> bool:
        return self.get_collection(collection_name).delete(record_id)

    def list_collections(self) -> List[str]:
        return sorted(self.collections)

    def save(self, directory: str | Path) -> None:
        PersistenceManager(Path(directory)).save(self)

    @classmethod
    def load(cls, directory: str | Path) -> "MiniVectorDB":
        return PersistenceManager(Path(directory)).load()

