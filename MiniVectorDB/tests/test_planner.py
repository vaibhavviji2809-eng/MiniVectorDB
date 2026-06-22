import unittest

from MiniVectorDB.optimizer.planner import QueryPlanner


class PlannerTests(unittest.TestCase):
    def test_strategy_selection(self):
        planner = QueryPlanner()
        self.assertEqual(planner.choose_strategy(100), "brute_force")
        self.assertEqual(planner.choose_strategy(5_000), "ivf")
        self.assertEqual(planner.choose_strategy(20_000), "hnsw")
        self.assertEqual(planner.choose_strategy(2_000_000), "diskann")


if __name__ == "__main__":
    unittest.main()

