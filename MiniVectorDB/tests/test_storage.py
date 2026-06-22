import unittest
import tempfile

from MiniVectorDB.core import MiniVectorDB


class StorageTests(unittest.TestCase):
    def test_insert_search_delete(self):
        db = MiniVectorDB()
        db.create_collection("docs", dimension=32)
        record = db.insert("docs", text="flashattention paper", metadata={"tag": "research"})
        self.assertEqual(record.metadata["tag"], "research")

        results = db.search("docs", query="flashattention", top_k=3)
        self.assertTrue(results)

        self.assertTrue(db.delete("docs", record.id))

    def test_persistence_round_trip(self):
        db = MiniVectorDB()
        db.create_collection("docs", dimension=32)
        db.insert("docs", text="persist me", metadata={"tag": "storage"})

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            loaded = MiniVectorDB.load(tmpdir)
            results = loaded.search("docs", query="persist", top_k=1)
            self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
