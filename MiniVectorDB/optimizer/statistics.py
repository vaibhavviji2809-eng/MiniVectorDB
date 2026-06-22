"""Collection statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import List


@dataclass
class CollectionStatistics:
    inserts: int = 0
    searches: int = 0
    deletions: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record_latency(self, value: float) -> None:
        self.latencies_ms.append(value)

    @property
    def average_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

