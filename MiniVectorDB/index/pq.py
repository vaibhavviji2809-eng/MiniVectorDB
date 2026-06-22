"""Quantization helpers: PQ, OPQ, and residual quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, List, Sequence, Tuple


def _split(vector: Sequence[float], parts: int) -> List[List[float]]:
    if parts <= 1:
        return [list(vector)]
    length = len(vector)
    base, remainder = divmod(length, parts)
    chunks: List[List[float]] = []
    start = 0
    for index in range(parts):
        end = start + base + (1 if index < remainder else 0)
        chunks.append(list(vector[start:end]))
        start = end
    return chunks


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _mean_vector(vectors: List[Sequence[float]], dimension: int) -> List[float]:
    if not vectors:
        return [0.0] * dimension
    return [
        mean(values) if values else 0.0
        for values in zip(*[list(vector)[:dimension] for vector in vectors])
    ]


def _kmeans(vectors: List[Sequence[float]], k: int, iterations: int = 8) -> List[List[float]]:
    if not vectors:
        return []
    k = max(1, min(k, len(vectors)))
    centroids = [list(vectors[index]) for index in range(k)]
    for _ in range(iterations):
        clusters: List[List[Sequence[float]]] = [[] for _ in range(k)]
        for vector in vectors:
            index = min(range(k), key=lambda candidate: _distance(vector, centroids[candidate]))
            clusters[index].append(vector)
        new_centroids = []
        for index, cluster in enumerate(clusters):
            if cluster:
                new_centroids.append(_mean_vector(cluster, len(cluster[0])))
            else:
                new_centroids.append(centroids[index])
        centroids = new_centroids
    return centroids


@dataclass
class ScalarQuantizer:
    """Per-dimension scalar quantization."""

    bits: int = 8
    min_values: List[float] = field(default_factory=list)
    max_values: List[float] = field(default_factory=list)

    @property
    def levels(self) -> int:
        return max(2, 2**self.bits)

    def fit(self, vectors: List[Sequence[float]]) -> "ScalarQuantizer":
        if not vectors:
            self.min_values = []
            self.max_values = []
            return self
        dimension = len(vectors[0])
        self.min_values = [min(vector[index] for vector in vectors) for index in range(dimension)]
        self.max_values = [max(vector[index] for vector in vectors) for index in range(dimension)]
        return self

    def encode(self, vector: Sequence[float]) -> List[int]:
        codes: List[int] = []
        for index, value in enumerate(vector):
            min_v = self.min_values[index] if index < len(self.min_values) else value
            max_v = self.max_values[index] if index < len(self.max_values) else value
            span = max(max_v - min_v, 1e-9)
            normalized = (value - min_v) / span
            codes.append(int(round(normalized * (self.levels - 1))))
        return codes

    def decode(self, codes: Sequence[int]) -> List[float]:
        decoded: List[float] = []
        for index, code in enumerate(codes):
            min_v = self.min_values[index] if index < len(self.min_values) else 0.0
            max_v = self.max_values[index] if index < len(self.max_values) else 0.0
            span = max(max_v - min_v, 1e-9)
            decoded.append(min_v + (code / max(self.levels - 1, 1)) * span)
        return decoded


@dataclass
class OPQTransform:
    """A lightweight OPQ-style dimension reordering transform."""

    permutation: List[int] = field(default_factory=list)

    def fit(self, vectors: List[Sequence[float]]) -> "OPQTransform":
        if not vectors:
            self.permutation = []
            return self
        dimension = len(vectors[0])
        variances = []
        for index in range(dimension):
            values = [vector[index] for vector in vectors]
            avg = sum(values) / len(values)
            variance = sum((value - avg) ** 2 for value in values) / max(len(values), 1)
            variances.append((variance, index))
        self.permutation = [index for _, index in sorted(variances, reverse=True)]
        return self

    def transform(self, vector: Sequence[float]) -> List[float]:
        if not self.permutation:
            return list(vector)
        return [vector[index] for index in self.permutation]

    def inverse_transform(self, vector: Sequence[float]) -> List[float]:
        if not self.permutation:
            return list(vector)
        restored = [0.0] * len(self.permutation)
        for rotated_index, original_index in enumerate(self.permutation):
            if rotated_index < len(vector):
                restored[original_index] = vector[rotated_index]
        return restored


@dataclass
class ProductQuantizer:
    """Optimized product quantization with optional OPQ rotation."""

    parts: int = 4
    codebook_size: int = 16
    use_opq: bool = True
    transform: OPQTransform = field(default_factory=OPQTransform)
    codebooks: List[List[List[float]]] = field(default_factory=list)

    def fit(self, vectors: List[Sequence[float]]) -> "ProductQuantizer":
        if not vectors:
            self.codebooks = []
            self.transform = OPQTransform()
            return self

        if self.use_opq:
            self.transform.fit(vectors)
        transformed = [self.transform.transform(vector) for vector in vectors]
        parts = _split(transformed[0], self.parts)
        boundaries: List[Tuple[int, int]] = []
        start = 0
        for chunk in parts:
            end = start + len(chunk)
            boundaries.append((start, end))
            start = end

        self.codebooks = []
        for start, end in boundaries:
            sub_vectors = [vector[start:end] for vector in transformed]
            self.codebooks.append(_kmeans(sub_vectors, self.codebook_size))
        return self

    def encode(self, vector: Sequence[float]) -> List[int]:
        transformed = self.transform.transform(vector)
        parts = _split(transformed, self.parts)
        codes: List[int] = []
        for part_index, chunk in enumerate(parts):
            codebook = self.codebooks[part_index] if part_index < len(self.codebooks) else []
            if not codebook:
                codes.append(0)
                continue
            codes.append(min(range(len(codebook)), key=lambda idx: _distance(chunk, codebook[idx])))
        return codes

    def decode(self, codes: Sequence[int]) -> List[float]:
        reconstructed: List[float] = []
        for part_index, code in enumerate(codes):
            codebook = self.codebooks[part_index] if part_index < len(self.codebooks) else []
            if not codebook:
                continue
            reconstructed.extend(codebook[min(code, len(codebook) - 1)])
        return self.transform.inverse_transform(reconstructed)

    def reconstruction_error(self, vector: Sequence[float]) -> float:
        return _distance(self.decode(self.encode(vector)), vector)


@dataclass
class ResidualQuantizer:
    """Residual quantization across sequential codebooks."""

    stages: int = 2
    codebook_size: int = 16
    codebooks: List[List[List[float]]] = field(default_factory=list)

    def fit(self, vectors: List[Sequence[float]]) -> "ResidualQuantizer":
        if not vectors:
            self.codebooks = []
            return self

        residuals = [list(vector) for vector in vectors]
        self.codebooks = []
        for _ in range(self.stages):
            codebook = _kmeans(residuals, self.codebook_size)
            self.codebooks.append(codebook)
            next_residuals = []
            for residual in residuals:
                nearest = min(codebook, key=lambda centroid: _distance(residual, centroid))
                next_residuals.append([value - centroid for value, centroid in zip(residual, nearest)])
            residuals = next_residuals
        return self

    def encode(self, vector: Sequence[float]) -> List[int]:
        residual = list(vector)
        codes: List[int] = []
        for codebook in self.codebooks:
            if not codebook:
                codes.append(0)
                continue
            index = min(range(len(codebook)), key=lambda candidate: _distance(residual, codebook[candidate]))
            codes.append(index)
            residual = [value - centroid for value, centroid in zip(residual, codebook[index])]
        return codes

    def decode(self, codes: Sequence[int], dimension: int) -> List[float]:
        reconstruction = [0.0] * dimension
        for stage, code in enumerate(codes):
            if stage >= len(self.codebooks) or not self.codebooks[stage]:
                continue
            centroid = self.codebooks[stage][min(code, len(self.codebooks[stage]) - 1)]
            reconstruction = [value + centroid[index] for index, value in enumerate(reconstruction)]
        return reconstruction


@dataclass
class OptimizedProductQuantizer:
    """Composite helper that combines OPQ and residual quantization."""

    pq: ProductQuantizer = field(default_factory=ProductQuantizer)
    rq: ResidualQuantizer = field(default_factory=ResidualQuantizer)
    use_residual: bool = True

    def fit(self, vectors: List[Sequence[float]]) -> "OptimizedProductQuantizer":
        self.pq.fit(vectors)
        if self.use_residual:
            self.rq.fit(vectors)
        return self

    def encode(self, vector: Sequence[float]) -> dict:
        payload = {"pq": self.pq.encode(vector)}
        if self.use_residual:
            payload["rq"] = self.rq.encode(vector)
        return payload

    def decode(self, codes: dict, dimension: int | None = None) -> List[float]:
        reconstructed = self.pq.decode(codes.get("pq", []))
        if self.use_residual and dimension is not None:
            residual = self.rq.decode(codes.get("rq", []), dimension)
            reconstructed = [
                value + residual[index] if index < len(residual) else value
                for index, value in enumerate(reconstructed)
            ]
        return reconstructed
