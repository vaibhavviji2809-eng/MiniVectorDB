"""Adaptive query planner for retrieval strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QueryPlan:
    strategy: str
    metric: str = "cosine"
    use_hybrid: bool = False
    use_compression: bool = False
    use_diskann: bool = False
    explanation: str = ""


@dataclass
class QueryPlanner:
    brute_force_threshold: int = 1_024
    ivf_threshold: int = 10_000
    hnsw_threshold: int = 100_000
    diskann_threshold: int = 1_000_000
    pq_threshold: int = 250_000

    def plan(
        self,
        collection_size: int,
        *,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        metric: str = "cosine",
        prefer_compression: bool = False,
    ) -> QueryPlan:
        filters = filters or {}
        has_keyword_signal = bool(query and query.strip())
        has_filter_signal = bool(filters)

        if collection_size < self.brute_force_threshold:
            return QueryPlan(
                strategy="brute_force",
                metric=metric,
                use_hybrid=has_keyword_signal and has_filter_signal,
                explanation="small collection",
            )

        if prefer_compression or collection_size >= self.pq_threshold:
            strategy = "pq"
            if collection_size >= self.diskann_threshold:
                strategy = "diskann"
            return QueryPlan(
                strategy=strategy,
                metric=metric,
                use_hybrid=has_keyword_signal and has_filter_signal,
                use_compression=True,
                use_diskann=strategy == "diskann",
                explanation="compression or very large collection",
            )

        if collection_size < self.ivf_threshold:
            return QueryPlan(
                strategy="ivf",
                metric=metric,
                use_hybrid=has_keyword_signal,
                explanation="mid-sized collection",
            )

        if collection_size < self.hnsw_threshold:
            return QueryPlan(
                strategy="hnsw",
                metric=metric,
                use_hybrid=has_keyword_signal,
                explanation="large collection",
            )

        return QueryPlan(
            strategy="diskann",
            metric=metric,
            use_hybrid=has_keyword_signal,
            use_diskann=True,
            use_compression=True,
            explanation="very large collection",
        )

    def choose_strategy(
        self,
        collection_size: int,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        metric: str = "cosine",
    ) -> str:
        return self.plan(
            collection_size,
            filters=filters,
            query=query,
            metric=metric,
        ).strategy

