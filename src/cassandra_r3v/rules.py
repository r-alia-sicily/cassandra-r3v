"""Human-readable semantic rules and their vectorized activations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from .config import LEVELS
from .fuzzy import FuzzyVocabulary


@dataclass(frozen=True, order=True)
class Antecedent:
    feature: str
    level: str

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown linguistic level: {self.level}")

    @property
    def label(self) -> str:
        return f"{self.feature} IS {self.level}"

    def to_dict(self) -> dict[str, str]:
        return {"feature": self.feature, "level": self.level}


@dataclass(frozen=True)
class SemanticRule:
    antecedents: tuple[Antecedent, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.antecedents))
        if not ordered:
            raise ValueError("a rule needs at least one antecedent")
        features = [antecedent.feature for antecedent in ordered]
        if len(set(features)) != len(features):
            raise ValueError("a semantic rule cannot repeat the same physical variable")
        object.__setattr__(self, "antecedents", ordered)

    @property
    def order(self) -> int:
        return len(self.antecedents)

    @property
    def key(self) -> str:
        return " AND ".join(antecedent.label for antecedent in self.antecedents)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(antecedent.feature for antecedent in self.antecedents)

    def activation(
        self,
        x: np.ndarray,
        vocabulary: FuzzyVocabulary,
        cache: dict[tuple[str, str], np.ndarray] | None = None,
    ) -> np.ndarray:
        atom_cache = cache if cache is not None else vocabulary.all_memberships(x)
        result = np.ones(np.asarray(x).shape[0], dtype=float)
        for antecedent in self.antecedents:
            result *= atom_cache[(antecedent.feature, antecedent.level)]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"antecedents": [antecedent.to_dict() for antecedent in self.antecedents]}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SemanticRule":
        return cls(tuple(Antecedent(**item) for item in values["antecedents"]))


def unary_backbone(features: Iterable[str]) -> list[SemanticRule]:
    return [SemanticRule((Antecedent(feature, level),)) for feature in features for level in LEVELS]


def activation_matrix(x: np.ndarray, vocabulary: FuzzyVocabulary, rules: Iterable[SemanticRule]) -> np.ndarray:
    matrix = np.asarray(x, dtype=float)
    rule_list = tuple(rules)
    cache = vocabulary.all_memberships(matrix)
    return np.column_stack([rule.activation(matrix, vocabulary, cache) for rule in rule_list])


def compose_rules(left: SemanticRule, right: SemanticRule) -> SemanticRule | None:
    """Combine readable concepts, rejecting repeated physical variables."""

    antecedents = tuple(sorted(set(left.antecedents + right.antecedents)))
    if len({item.feature for item in antecedents}) != len(antecedents):
        return None
    return SemanticRule(antecedents)


def pairwise_candidates(unary_rules: Iterable[SemanticRule]) -> list[SemanticRule]:
    rules = [rule for rule in unary_rules if rule.order == 1]
    result: dict[str, SemanticRule] = {}
    for left, right in combinations(rules, 2):
        candidate = compose_rules(left, right)
        if candidate is not None:
            result[candidate.key] = candidate
    return list(result.values())
