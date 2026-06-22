"""Benchmark framework for retrieval quality and performance."""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from ..core import MiniVectorDB
from ..index.brute_force import BruteForceIndex


@dataclass
class BenchmarkResult:
    engine: str
    recall_at_10: float
    latency_ms: float
    memory_mb: float
    insert_per_sec: float
    search_per_sec: float
    index_build_time_ms: float


@dataclass
class BenchmarkSuite:
    collection_size: int = 2_000
    dimension: int = 64
    top_k: int = 10
    seed: int = 7
    external_engines: Dict[str, Callable[[], Optional[BenchmarkResult]]] = field(
        default_factory=dict
    )

    def _random_vector(self, rng: random.Random) -> List[float]:
        return [rng.random() for _ in range(self.dimension)]

    def _run_minivectordb(self) -> BenchmarkResult:
        rng = random.Random(self.seed)
        db = MiniVectorDB()
        collection = db.create_collection("benchmark", dimension=self.dimension)

        vectors = [self._random_vector(rng) for _ in range(self.collection_size)]
        queries = vectors[: min(50, len(vectors))]

        tracemalloc.start()
        start = time.perf_counter()
        for index, vector in enumerate(vectors):
            collection.insert(vector=vector, metadata={"bucket": index % 10})
        insert_time = time.perf_counter() - start

        start = time.perf_counter()
        collection.hnsw_index.build(collection._all_records())
        index_build_time = (time.perf_counter() - start) * 1000

        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        exact_index = BruteForceIndex()
        exact_hits = 0
        search_start = time.perf_counter()
        for query in queries:
            expected = [
                item.record.id
                for item in exact_index.search(query, collection._all_records(), top_k=self.top_k)
            ]
            actual = [item["id"] for item in db.search("benchmark", query_vector=query, top_k=self.top_k)]
            exact_hits += len(set(expected) & set(actual)) / max(len(expected), 1)
        search_time = time.perf_counter() - search_start
        recall_at_10 = exact_hits / max(len(queries), 1)

        total_queries = max(len(queries), 1)
        latency_ms = (search_time / total_queries) * 1000
        insert_per_sec = self.collection_size / max(insert_time, 1e-9)
        search_per_sec = total_queries / max(search_time, 1e-9)

        return BenchmarkResult(
            engine="MiniVectorDB",
            recall_at_10=recall_at_10,
            latency_ms=latency_ms,
            memory_mb=peak_memory,
            insert_per_sec=insert_per_sec,
            search_per_sec=search_per_sec,
            index_build_time_ms=index_build_time,
        )

    def run(self) -> Dict[str, BenchmarkResult]:
        results = {"MiniVectorDB": self._run_minivectordb()}
        for name, adapter in self.external_engines.items():
            result = adapter()
            if result is not None:
                results[name] = result
        return results


def run() -> Dict[str, BenchmarkResult]:
    return BenchmarkSuite().run()


if __name__ == "__main__":
    print({name: asdict(result) for name, result in run().items()})

