"""Validated configuration objects for training and structural search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


PROTOCOLS = ("uncapped", "cap125")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
LEVELS = ("VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH")
PSEUDO_ENDPOINT_FRACTIONS = (0.35, 0.50, 0.65, 0.80, 0.92)


@dataclass
class TrainingConfig:
    """Parameters of the manuscript-aligned parametric learner."""

    loss: str = "nasa"
    authority_mode: str = "learned"
    epochs: int = 250
    learning_rate_consequents: float = 0.025
    learning_rate_authorities: float = 0.025
    ridge_lambda: float = 0.9
    authority_l2: float = 0.0
    gradient_clip: float = 10.0
    patience: int = 50
    min_delta: float = 1e-8
    seed: int = 7
    pseudo_endpoint_fractions: tuple[float, ...] = PSEUDO_ENDPOINT_FRACTIONS
    denominator_epsilon: float = 1e-12

    def validate(self) -> "TrainingConfig":
        if self.loss not in {"nasa", "mse"}:
            raise ValueError("loss must be 'nasa' or 'mse'")
        if self.authority_mode not in {"learned", "uniform"}:
            raise ValueError("authority_mode must be 'learned' or 'uniform'")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        for name in ("learning_rate_consequents", "learning_rate_authorities"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ridge_lambda < 0 or self.authority_l2 < 0:
            raise ValueError("regularization strengths cannot be negative")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        fractions = tuple(float(v) for v in self.pseudo_endpoint_fractions)
        if not fractions or any(not 0.0 < v < 1.0 for v in fractions):
            raise ValueError("pseudo-endpoint fractions must lie strictly in (0, 1)")
        if tuple(sorted(set(fractions))) != fractions:
            raise ValueError("pseudo-endpoint fractions must be unique and increasing")
        self.pseudo_endpoint_fractions = fractions
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pseudo_endpoint_fractions"] = list(self.pseudo_endpoint_fractions)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        clean = dict(values)
        if "pseudo_endpoint_fractions" in clean:
            clean["pseudo_endpoint_fractions"] = tuple(clean["pseudo_endpoint_fractions"])
        return cls(**clean).validate()


@dataclass
class ComputeBudget:
    """Hard bounds that keep an AutoStructure run finite and explainable."""

    max_seconds: float = 3600.0
    max_candidates: int = 10_000
    max_validations_per_seed: int = 4
    max_rounds: int = 3

    def validate(self) -> "ComputeBudget":
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_candidates < 1 or self.max_validations_per_seed < 1:
            raise ValueError("candidate and validation limits must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        return self


@dataclass
class AutoStructureConfig:
    """Evidence thresholds and compute guards for AutoStructure."""

    min_support: float = 0.01
    activation_support_cutoff: float = 0.01
    min_novelty: float = 0.005
    min_relative_improvement: float = 0.01
    folds: int = 3
    seeds: tuple[int, ...] = (7, 42, 99)
    consensus_required: int = 2
    max_order: int = 3
    screen_top_k: int = 24
    prune_relative_authority: float = 0.002
    parent_relative_authority: float = 0.0002
    validation_epochs: int = 100
    budget: ComputeBudget = field(default_factory=ComputeBudget)

    def validate(self) -> "AutoStructureConfig":
        for name in (
            "min_support",
            "activation_support_cutoff",
            "min_novelty",
            "min_relative_improvement",
            "prune_relative_authority",
            "parent_relative_authority",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.min_support > 1 or self.activation_support_cutoff > 1:
            raise ValueError("support thresholds cannot exceed one")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        self.seeds = tuple(int(seed) for seed in self.seeds)
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not 1 <= self.consensus_required <= len(self.seeds):
            raise ValueError("consensus_required must be between one and the number of seeds")
        if self.max_order < 2:
            raise ValueError("max_order must be at least two")
        if self.screen_top_k < 1 or self.validation_epochs < 1:
            raise ValueError("screen_top_k and validation_epochs must be positive")
        if self.parent_relative_authority > self.prune_relative_authority:
            raise ValueError(
                "parent_relative_authority must not exceed prune_relative_authority; "
                "weak rules may remain eligible as parents before they are pruned"
            )
        self.budget.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AutoStructureConfig":
        clean = dict(values)
        if "seeds" in clean:
            clean["seeds"] = tuple(clean["seeds"])
        if isinstance(clean.get("budget"), dict):
            clean["budget"] = ComputeBudget(**clean["budget"])
        return cls(**clean).validate()


def normalize_protocols(values: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(dict.fromkeys(str(value).lower() for value in values))
    if not chosen:
        raise ValueError("select at least one target protocol")
    invalid = sorted(set(chosen) - set(PROTOCOLS))
    if invalid:
        raise ValueError(f"unknown protocols: {', '.join(invalid)}")
    return tuple(protocol for protocol in PROTOCOLS if protocol in chosen)


def normalize_subsets(values: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(dict.fromkeys(str(value).upper() for value in values))
    if not chosen:
        raise ValueError("select at least one C-MAPSS subset")
    invalid = sorted(set(chosen) - set(SUBSETS))
    if invalid:
        raise ValueError(f"unknown subsets: {', '.join(invalid)}")
    return tuple(subset for subset in SUBSETS if subset in chosen)
