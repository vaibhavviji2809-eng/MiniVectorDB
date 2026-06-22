import asyncio
import unittest

from MiniVectorDB.distributed.router import Router
from MiniVectorDB.distributed.shard import Shard


class DistributedTests(unittest.TestCase):
    def test_async_search(self):
        router = Router()
        shard = Shard("shard-1")
        router.add_shard(shard)
        shard.db.create_collection("docs", dimension=8)
        shard.db.insert("docs", text="hello world")

        results = asyncio.run(router.search_async("docs", query="hello"))
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()

