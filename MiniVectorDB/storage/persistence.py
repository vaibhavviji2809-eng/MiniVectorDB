"""Persistence helpers for saving and loading database state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import MiniVectorDB


class PersistenceManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, db: MiniVectorDB) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        vectors_payload = {name: collection.to_dict() for name, collection in db.collections.items()}
        metadata_payload = {
            name: collection.metadata_store.to_dict() for name, collection in db.collections.items()
        }
        index_payload = {
            name: {
                "strategy": collection.planner.choose_strategy(collection.count()),
                "count": collection.count(),
            }
            for name, collection in db.collections.items()
        }
        (self.directory / "vectors.bin").write_bytes(
            json.dumps(vectors_payload).encode("utf-8")
        )
        (self.directory / "metadata.bin").write_bytes(
            json.dumps(metadata_payload).encode("utf-8")
        )
        (self.directory / "index.bin").write_bytes(json.dumps(index_payload).encode("utf-8"))

    def load(self) -> MiniVectorDB:
        from ..core import Collection, MiniVectorDB
        from .metadata import MetadataStore

        path = self.directory / "vectors.bin"
        if not path.exists():
            return MiniVectorDB()
        payload = json.loads(path.read_text(encoding="utf-8"))
        db = MiniVectorDB()
        metadata_path = self.directory / "metadata.bin"
        metadata_payload = {}
        if metadata_path.exists():
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        collections = {}
        for name, collection_payload in payload.items():
            collection = Collection.from_dict(collection_payload)
            if name in metadata_payload:
                collection.metadata_store = MetadataStore.from_dict(metadata_payload[name])
            collections[name] = collection
        db.collections = collections
        return db
