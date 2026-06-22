"""Query planner that chooses an index strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryPlanner:
    brute_force_threshold: int = 1_024
    ivf_threshold: int = 10_000

    def choose_strategy(self, collection_size: int) -> str:
        if collection_size < self.brute_force_threshold:
            return "brute_force"
        if collection_size < self.ivf_threshold:
            return "ivf"
        return "hnsw"

