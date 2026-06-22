import unittest

from MiniVectorDB.api.routes import MiniVectorDBRoutes


class ApiTests(unittest.TestCase):
    def test_routes(self):
        routes = MiniVectorDBRoutes()
        self.assertTrue(routes.create_collection({"name": "docs"})["ok"])
        self.assertTrue(routes.insert({"collection": "docs", "text": "hello"})["ok"])
        self.assertTrue(routes.search({"collection": "docs", "query": "hello"})["ok"])


if __name__ == "__main__":
    unittest.main()

