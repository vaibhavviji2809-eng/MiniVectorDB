import unittest

from MiniVectorDB.index.hnsw import HNSWIndex
from MiniVectorDB.storage.page import VectorRecord


class HNSWTests(unittest.TestCase):
    def test_incremental_insert_and_delete(self):
        index = HNSWIndex()
        records = [
            VectorRecord(id="1", vector=[1.0, 0.0], metadata={}),
            VectorRecord(id="2", vector=[0.9, 0.1], metadata={}),
            VectorRecord(id="3", vector=[0.0, 1.0], metadata={}),
        ]
        index.build(records)
        hits = index.search([1.0, 0.0], top_k=1)
        self.assertEqual(hits[0].id, "1")

        index.remove_record("1")
        hits = index.search([1.0, 0.0], top_k=1)
        self.assertNotEqual(hits[0].id, "1")


if __name__ == "__main__":
    unittest.main()

