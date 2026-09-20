"""Stable five-level linguistic partitions used by Cassandra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import LEVELS


@dataclass(frozen=True)
class FuzzyPartition:
    """A Ruspini-style partition with two shoulders and three triangles.

    Between consecutive centers, exactly two adjacent memberships interpolate;
    consequently the five memberships sum to one over the whole real line.
    """

    feature: str
    centers: tuple[float, float, float, float, float]
    mode: str = "quantile"

    def __post_init__(self) -> None:
        values = np.asarray(self.centers, dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("a fuzzy partition needs five finite centers")
        if not np.all(np.diff(values) > 0):
            raise ValueError("fuzzy centers must be strictly increasing")

    @classmethod
    def fit(cls, feature: str, values: Iterable[float]) -> "FuzzyPartition":
        x = np.asarray(tuple(values) if not isinstance(values, np.ndarray) else values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError(f"cannot fit fuzzy partition for empty feature {feature}")

        raw = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95]).astype(float)
        spread = float(np.quantile(x, 0.95) - np.quantile(x, 0.05))
        scale = max(float(np.std(x)), abs(float(np.mean(x))) * 1e-6, 1.0)
        min_gap = max(spread * 1e-6, scale * 1e-6, 1e-9)

        if spread <= min_gap * 10:
            center = float(np.median(x))
            step = max(scale * 0.5, 0.5)
            centers = center + step * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            mode = "expanded_constant"
        else:
            centers = raw.copy()
            for index in range(1, 5):
                if centers[index] <= centers[index - 1] + min_gap:
                    centers[index] = centers[index - 1] + min_gap
            mode = "quantile" if np.all(np.diff(raw) > 0) else "expanded_discrete"

        return cls(feature=feature, centers=tuple(float(v) for v in centers), mode=mode)

    def memberships(self, values: np.ndarray | float) -> np.ndarray:
        """Return shape ``(n, 5)`` (or ``(5,)`` for a scalar)."""

        scalar = np.isscalar(values)
        x = np.atleast_1d(np.asarray(values, dtype=float))
        c = np.asarray(self.centers, dtype=float)
        result = np.zeros((x.size, 5), dtype=float)

        result[:, 0] = np.where(
            x <= c[0],
            1.0,
            np.where(x < c[1], (c[1] - x) / (c[1] - c[0]), 0.0),
        )
        for level in range(1, 4):
            left = (x - c[level - 1]) / (c[level] - c[level - 1])
            right = (c[level + 1] - x) / (c[level + 1] - c[level])
            result[:, level] = np.maximum(0.0, np.minimum(left, right))
        result[:, 4] = np.where(
            x >= c[4],
            1.0,
            np.where(x > c[3], (x - c[3]) / (c[4] - c[3]), 0.0),
        )
        result[~np.isfinite(x), :] = 0.0
        np.clip(result, 0.0, 1.0, out=result)
        return result[0] if scalar else result

    def membership(self, level: str, values: np.ndarray | float) -> np.ndarray | float:
        try:
            index = LEVELS.index(level)
        except ValueError as exc:
            raise ValueError(f"unknown linguistic level: {level}") from exc
        result = self.memberships(values)[..., index]
        return float(result) if np.isscalar(values) else result

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "centers": list(self.centers), "mode": self.mode}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FuzzyPartition":
        return cls(
            feature=str(values["feature"]),
            centers=tuple(float(v) for v in values["centers"]),
            mode=str(values.get("mode", "quantile")),
        )


class FuzzyVocabulary:
    """Ordered set of feature partitions with vectorized atom activations."""

    def __init__(self, feature_names: Iterable[str], partitions: Iterable[FuzzyPartition]):
        self.feature_names = tuple(feature_names)
        partition_list = tuple(partitions)
        self.partitions = {partition.feature: partition for partition in partition_list}
        if set(self.feature_names) != set(self.partitions):
            raise ValueError("feature names and fuzzy partitions do not match")
        self._feature_index = {name: index for index, name in enumerate(self.feature_names)}

    @classmethod
    def fit(cls, feature_names: Iterable[str], x: np.ndarray) -> "FuzzyVocabulary":
        names = tuple(feature_names)
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("X shape does not match feature_names")
        return cls(names, [FuzzyPartition.fit(name, matrix[:, i]) for i, name in enumerate(names)])

    def atom_activation(self, x: np.ndarray, feature: str, level: str) -> np.ndarray:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X must be two-dimensional")
        index = self._feature_index[feature]
        return np.asarray(self.partitions[feature].membership(level, matrix[:, index]), dtype=float)

    def all_memberships(self, x: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
        matrix = np.asarray(x, dtype=float)
        result: dict[tuple[str, str], np.ndarray] = {}
        for column, feature in enumerate(self.feature_names):
            memberships = self.partitions[feature].memberships(matrix[:, column])
            for level_index, level in enumerate(LEVELS):
                result[(feature, level)] = memberships[:, level_index]
        return result

    def partition_diagnostics(self, x: np.ndarray, tolerance: float = 1e-10) -> dict[str, Any]:
        matrix = np.asarray(x, dtype=float)
        max_error = 0.0
        never_active = 0
        modes: dict[str, str] = {}
        for column, feature in enumerate(self.feature_names):
            memberships = self.partitions[feature].memberships(matrix[:, column])
            max_error = max(max_error, float(np.max(np.abs(np.sum(memberships, axis=1) - 1.0))))
            never_active += int(np.sum(np.max(memberships, axis=0) <= tolerance))
            modes[feature] = self.partitions[feature].mode
        return {
            "features": len(self.feature_names),
            "sets": len(self.feature_names) * len(LEVELS),
            "max_partition_sum_error": max_error,
            "partition_of_unity_ok": bool(max_error <= tolerance),
            "never_active_sets": never_active,
            "modes": modes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "partitions": [self.partitions[name].to_dict() for name in self.feature_names],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FuzzyVocabulary":
        return cls(values["feature_names"], [FuzzyPartition.from_dict(v) for v in values["partitions"]])
