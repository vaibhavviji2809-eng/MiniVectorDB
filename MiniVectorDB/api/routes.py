"""Route handlers for the MiniVectorDB HTTP API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..core import MiniVectorDB
from .schemas import CreateCollectionRequest, DeleteRequest, InsertRequest, SearchRequest


class MiniVectorDBRoutes:
    def __init__(self, db: MiniVectorDB | None = None) -> None:
        self.db = db or MiniVectorDB()

    def create_collection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = CreateCollectionRequest(**payload)
        collection = self.db.create_collection(
            request.name,
            dimension=request.dimension,
            page_capacity=request.page_capacity,
            pages_per_segment=request.pages_per_segment,
        )
        return {"ok": True, "collection": collection.name}

    def insert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = InsertRequest(**payload)
        record = self.db.insert(
            request.collection,
            vector=request.vector,
            metadata=request.metadata,
            text=request.text,
            record_id=request.id,
            wait=request.wait,
        )
        return {"ok": True, "record": record.to_dict()}

    def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = SearchRequest(**payload)
        results = self.db.search(
            request.collection,
            query=request.query,
            query_vector=request.query_vector,
            top_k=request.top_k,
            filters=request.filters,
            metric=request.metric,
            use_hybrid=request.use_hybrid,
            prefer_compression=request.prefer_compression,
        )
        return {"ok": True, "results": results}

    def delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = DeleteRequest(**payload)
        deleted = self.db.delete(request.collection, request.id)
        return {"ok": deleted}
