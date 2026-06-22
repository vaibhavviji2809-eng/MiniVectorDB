import unittest

from MiniVectorDB.search.cosine import cosine_similarity
from MiniVectorDB.search.l2 import l2_distance


class SearchMetricTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)

    def test_l2(self):
        self.assertAlmostEqual(l2_distance([0, 0], [3, 4]), 5.0)


if __name__ == "__main__":
    unittest.main()

