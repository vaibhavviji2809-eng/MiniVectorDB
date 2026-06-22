"""Segment-level storage primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .page import Page, VectorRecord


@dataclass
class Segment:
    name: str
    page_capacity: int = 128
    max_pages: int = 16
    pages: List[Page] = field(default_factory=list)

    def _active_page(self) -> Page:
        if not self.pages:
            self.pages.append(Page(capacity=self.page_capacity))
        page = self.pages[-1]
        if page.is_full():
            if len(self.pages) >= self.max_pages:
                raise ValueError("segment is full")
            page = Page(capacity=self.page_capacity)
            self.pages.append(page)
        return page

    def is_full(self) -> bool:
        return bool(self.pages) and len(self.pages) >= self.max_pages and self.pages[-1].is_full()

    def insert(self, record: VectorRecord) -> None:
        self._active_page().insert(record)

    def delete(self, record_id: str) -> bool:
        deleted = False
        for page in self.pages:
            deleted = page.delete(record_id) or deleted
        self.pages = [page for page in self.pages if page.records]
        return deleted

    def find(self, record_id: str) -> Optional[VectorRecord]:
        for page in self.pages:
            found = page.find(record_id)
            if found is not None:
                return found
        return None

    def all_records(self) -> List[VectorRecord]:
        records: List[VectorRecord] = []
        for page in self.pages:
            records.extend(page.all_records())
        return records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "page_capacity": self.page_capacity,
            "max_pages": self.max_pages,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Segment":
        return cls(
            name=payload["name"],
            page_capacity=payload.get("page_capacity", 128),
            max_pages=payload.get("max_pages", 16),
            pages=[Page.from_dict(page) for page in payload.get("pages", [])],
        )
