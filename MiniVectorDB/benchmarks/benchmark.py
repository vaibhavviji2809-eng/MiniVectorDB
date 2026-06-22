"""Benchmark harness for MiniVectorDB."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..core import MiniVectorDB


@dataclass
class BenchmarkResult:
    insert_sec: float
    search_sec: float


def run() -> BenchmarkResult:
    db = MiniVectorDB()
    collection = db.create_collection("demo")
    start = time.perf_counter()
    for index in range(100):
        collection.insert(text=f"document {index}", metadata={"group": "benchmark"})
    insert_sec = time.perf_counter() - start

    start = time.perf_counter()
    collection.search(query="document", top_k=10)
    search_sec = time.perf_counter() - start
    return BenchmarkResult(insert_sec=insert_sec, search_sec=search_sec)


if __name__ == "__main__":
    print(run())

