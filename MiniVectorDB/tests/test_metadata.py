import unittest

from MiniVectorDB.storage.metadata import MetadataStore
from MiniVectorDB.storage.page import VectorRecord


class MetadataTests(unittest.TestCase):
    def test_range_and_prefix_filters(self):
        store = MetadataStore()
        store.add(VectorRecord(id="1", vector=[], metadata={"year": 2026, "title": "flashattention"}))
        store.add(VectorRecord(id="2", vector=[], metadata={"year": 2024, "title": "transformers"}))

        self.assertEqual(store.filter_ids({"year": {"gte": 2025}}), {"1"})
        self.assertEqual(store.filter_ids({"title": {"prefix": "flash"}}), {"1"})


if __name__ == "__main__":
    unittest.main()

