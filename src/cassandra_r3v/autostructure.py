"""Evidence-controlled structural growth for readable R3v rules."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .cancellation import CancellationToken, NEVER_CANCEL
from .config import AutoStructureConfig, TrainingConfig
from .data import PreparedDataset, pseudo_endpoint_indices
from .fuzzy import FuzzyVocabulary
from .losses import task_loss_and_gradient
from .model import R3vModel
from .rules import SemanticRule, activation_matrix, compose_rules


class ComputeBudgetExceeded(RuntimeError):
    def __init__(self, message: str, recommendations: Iterable[str]):
        super().__init__(message)
        self.recommendations = tuple(recommendations)


@dataclass
class CandidateEvidence:
    rule: SemanticRule
    support: float
    novelty: float
    directional_score: float
    seed_improvements: dict[int, float]
    seed_favorable_folds: dict[int, int]
    consensus: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "rule_label": self.rule.key,
            "support": self.support,
            "novelty": self.novelty,
            "directional_score": self.directional_score,
            "seed_improvements": {str(k): v for k, v in self.seed_improvements.items()},
            "seed_favorable_folds": {str(k): v for k, v in self.seed_favorable_folds.items()},
            "consensus": self.consensus,
        }


@dataclass
class StructureResult:
    initial_rule_count: int
    final_rule_count: int
    accepted_rules: list[str]
    pruning_candidates: list[str]
    rounds: list[dict[str, Any]]
    stop_reason: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_pairwise_candidates(rule_count: int, feature_count: int, levels: int = 5) -> int:
    """Exact count for a complete unary backbone; conservative otherwise."""

    complete_unary = feature_count * levels
    if rule_count >= complete_unary:
        return feature_count * (feature_count - 1) // 2 * levels * levels
    return rule_count * max(0, rule_count - 1) // 2


def compute_recommendations(config: AutoStructureConfig) -> tuple[str, ...]:
    return (
        f"Increase min_support above {config.min_support:g}",
        f"Increase min_novelty above {config.min_novelty:g}",
        "Reduce screen_top_k or max_validations_per_seed",
        "Limit max_order to 2 for this run",
        "Use a more compact backbone while keeping structural selection separate",
    )


def _candidate_pool(model: R3vModel, config: AutoStructureConfig) -> list[SemanticRule]:
    shares = model.relative_authority_shares
    eligible = [rule for rule, share in zip(model.rules, shares) if share >= config.parent_relative_authority]
    current = {rule.key for rule in model.rules}
    maximum_order = max(rule.order for rule in model.rules)
    next_order = min(maximum_order + 1, config.max_order)

    if maximum_order == 1:
        left_pool = [rule for rule in eligible if rule.order == 1]
        right_pool = left_pool
    else:
        left_pool = [rule for rule in eligible if rule.order == maximum_order]
        right_pool = [rule for rule in eligible if rule.order == 1]

    candidates: dict[str, SemanticRule] = {}
    for left in left_pool:
        for right in right_pool:
            combined = compose_rules(left, right)
            if combined is None or combined.order != next_order or combined.key in current:
                continue
            candidates[combined.key] = combined
    return list(candidates.values())


def _screen_candidates(
    model: R3vModel,
    x: np.ndarray,
    y: np.ndarray,
    candidates: list[SemanticRule],
    config: AutoStructureConfig,
    started: float,
    token: CancellationToken,
    progress: Callable[[float, str], None] | None,
) -> list[CandidateEvidence]:
    current_mu = activation_matrix(x, model.vocabulary, model.rules)
    prediction = model.predict(x)
    _, direction = task_loss_and_gradient(model.config.loss, prediction - y)
    direction = -direction

    column_norms = np.linalg.norm(current_mu, axis=0)
    normalized_current = current_mu / np.where(column_norms > 1e-15, column_norms, 1.0)
    left_vectors, singular_values, _ = np.linalg.svd(normalized_current, full_matrices=False)
    tolerance = max(normalized_current.shape) * np.finfo(float).eps * max(float(singular_values[0]), 1.0)
    rank = int(np.sum(singular_values > tolerance))
    q = left_vectors[:, :rank]
    cache = model.vocabulary.all_memberships(x)
    screened: list[CandidateEvidence] = []
    total = max(1, len(candidates))
    for index, candidate in enumerate(candidates, start=1):
        token.checkpoint()
        if time.monotonic() - started > config.budget.max_seconds:
            raise ComputeBudgetExceeded(
                "AutoStructure reached its time limit during screening",
                compute_recommendations(config),
            )
        activation = candidate.activation(x, model.vocabulary, cache)
        support = float(np.mean(activation > config.activation_support_cutoff))
        if support < config.min_support:
            continue
        projection = q @ (q.T @ activation) if rank else np.zeros_like(activation)
        residual = activation - projection
        energy = float(np.dot(activation, activation))
        novelty = float(np.dot(residual, residual) / energy) if energy > 1e-15 else 0.0
        if novelty < config.min_novelty:
            continue
        denominator = float(np.linalg.norm(residual) * np.linalg.norm(direction))
        score = abs(float(np.dot(residual, direction))) / denominator if denominator > 1e-15 else 0.0
        screened.append(CandidateEvidence(candidate, support, novelty, score, {}, {}, 0))
        if progress and (index == 1 or index % 100 == 0 or index == total):
            progress(0.25 * index / total, f"Screening {index}/{total} · retained {len(screened)}")
    screened.sort(key=lambda item: item.directional_score, reverse=True)
    return screened[: config.screen_top_k]


def _group_folds(units: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(units).copy()
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    return [np.asarray(values) for values in np.array_split(unique, count)]


def _fold_data(
    dataset: PreparedDataset, validation_units: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FuzzyVocabulary, float]:
    columns = [dataset.feature_names.index(name) for name in feature_names]
    validation_mask = np.isin(dataset.train_units, validation_units)
    training_mask = ~validation_mask
    raw_training = dataset.raw_train_x[training_mask][:, columns]
    mean = np.mean(raw_training, axis=0)
    scale = np.std(raw_training, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    full_training_x = (raw_training - mean) / scale
    full_validation_x = (dataset.raw_train_x[validation_mask][:, columns] - mean) / scale
    training_units = dataset.train_units[training_mask]
    validation_group_units = dataset.train_units[validation_mask]
    training_cycles = dataset.train_cycles[training_mask]
    validation_cycles = dataset.train_cycles[validation_mask]
    train_indices = pseudo_endpoint_indices(training_units, training_cycles)
    validation_indices = pseudo_endpoint_indices(validation_group_units, validation_cycles)
    vocabulary = FuzzyVocabulary.fit(feature_names, full_training_x)
    return (
        full_training_x[train_indices],
        dataset.train_y[training_mask][train_indices],
        full_validation_x[validation_indices],
        dataset.train_y[validation_mask][validation_indices],
        vocabulary,
        125.0 if dataset.protocol == "cap125" else float(np.max(dataset.train_y[training_mask])),
    )


def _validation_config(base: TrainingConfig, auto: AutoStructureConfig, seed: int) -> TrainingConfig:
    values = base.to_dict()
    values.update({"epochs": auto.validation_epochs, "seed": seed, "patience": min(base.patience, auto.validation_epochs)})
    return TrainingConfig.from_dict(values)


def _validate(
    dataset: PreparedDataset,
    parent: R3vModel,
    candidates: list[CandidateEvidence],
    config: AutoStructureConfig,
    started: float,
    token: CancellationToken,
    progress: Callable[[float, str], None] | None,
) -> list[CandidateEvidence]:
    selected = candidates[: config.budget.max_validations_per_seed]
    if not selected:
        return []
    total_jobs = len(config.seeds) * config.folds * (1 + len(selected))
    completed = 0
    parent_rules = tuple(parent.rules)
    feature_names = tuple(parent.vocabulary.feature_names)

    for seed in config.seeds:
        parent_scores: list[float] = []
        child_scores: dict[str, list[float]] = {item.rule.key: [] for item in selected}
        folds = _group_folds(dataset.train_units, config.folds, seed)
        for fold_number, validation_units in enumerate(folds, start=1):
            token.checkpoint()
            if time.monotonic() - started > config.budget.max_seconds:
                raise ComputeBudgetExceeded(
                    "AutoStructure reached its time limit during validation",
                    compute_recommendations(config),
                )
            train_x, train_y, validation_x, validation_y, vocabulary, fold_upper_bound = _fold_data(
                dataset, validation_units, feature_names
            )
            train_config = _validation_config(parent.config, config, seed)
            fold_bounds = (0.0, fold_upper_bound)
            matched_parent = R3vModel(vocabulary, parent_rules, fold_bounds, train_config)
            matched_parent.fit(train_x, train_y, token=token)
            parent_score = float(matched_parent.evaluate(validation_x, validation_y)["nasa_score"])
            parent_scores.append(parent_score)
            completed += 1

            for item in selected:
                child = R3vModel(vocabulary, parent_rules + (item.rule,), fold_bounds, train_config)
                child.fit(train_x, train_y, token=token)
                child_score = float(child.evaluate(validation_x, validation_y)["nasa_score"])
                child_scores[item.rule.key].append(child_score)
                completed += 1
                if progress:
                    progress(
                        0.25 + 0.65 * completed / total_jobs,
                        f"Seed {seed} · fold {fold_number}/{config.folds} · {item.rule.key[:52]}",
                    )

        parent_total = float(np.sum(parent_scores))
        for item in selected:
            scores = child_scores[item.rule.key]
            child_total = float(np.sum(scores))
            improvement = (parent_total - child_total) / max(parent_total, 1e-12)
            favorable = sum(child < baseline for child, baseline in zip(scores, parent_scores))
            item.seed_improvements[seed] = float(improvement)
            item.seed_favorable_folds[seed] = int(favorable)

    for item in selected:
        item.consensus = sum(
            item.seed_improvements[seed] >= config.min_relative_improvement
            and item.seed_favorable_folds[seed] >= (config.folds // 2 + 1)
            for seed in config.seeds
        )
    selected.sort(
        key=lambda item: (item.consensus, np.mean(list(item.seed_improvements.values())), item.directional_score),
        reverse=True,
    )
    return selected


def grow_structure(
    dataset: PreparedDataset,
    initial_model: R3vModel,
    config: AutoStructureConfig,
    token: CancellationToken = NEVER_CANCEL,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[R3vModel, StructureResult]:
    """Grow at most one rule per round; stop immediately without evidence."""

    config.validate()
    started = time.monotonic()
    current = initial_model
    feature_names = tuple(current.vocabulary.feature_names)
    columns = [dataset.feature_names.index(name) for name in feature_names]
    initial_count = len(current.rules)
    accepted: list[str] = []
    rounds: list[dict[str, Any]] = []
    stop_reason = "maximum rounds reached"

    for round_number in range(1, config.budget.max_rounds + 1):
        token.checkpoint()
        candidates = _candidate_pool(current, config)
        if len(candidates) > config.budget.max_candidates:
            raise ComputeBudgetExceeded(
                f"The search would generate {len(candidates):,} candidates, above the "
                f"{config.budget.max_candidates:,} limit",
                compute_recommendations(config),
            )
        if not candidates:
            stop_reason = "no semantically admissible candidates"
            break

        fit_x = dataset.fit_x[:, columns]
        evidence = _screen_candidates(
            current, fit_x, dataset.fit_y, candidates, config, started, token, progress
        )
        round_record: dict[str, Any] = {
            "round": round_number,
            "order": max(candidate.order for candidate in candidates),
            "generated": len(candidates),
            "screened": len(evidence),
        }
        if not evidence:
            round_record["decision"] = "stop: no candidate passed support and novelty"
            rounds.append(round_record)
            stop_reason = "support/novelty evidence insufficient"
            break

        validated = _validate(dataset, current, evidence, config, started, token, progress)
        round_record["validated"] = [item.to_dict() for item in validated]
        winner = next((item for item in validated if item.consensus >= config.consensus_required), None)
        if winner is None:
            round_record["decision"] = "stop: validation consensus insufficient"
            rounds.append(round_record)
            stop_reason = "no validated shortlisted candidate reached the required consensus"
            break

        child_rules = tuple(current.rules) + (winner.rule,)
        child = R3vModel(current.vocabulary, child_rules, current.target_bounds, current.config)
        child.metadata = dict(current.metadata)
        child.metadata["structure_parent_rules"] = len(current.rules)
        child.fit(dataset.fit_x[:, columns], dataset.fit_y, token=token)
        current = child
        accepted.append(winner.rule.key)
        round_record["decision"] = f"accepted: {winner.rule.key}"
        rounds.append(round_record)
        if progress:
            progress(0.95, f"Accepted rule {winner.rule.key}")

        if winner.rule.order >= config.max_order:
            stop_reason = "maximum semantic order reached"
            break

    elapsed = time.monotonic() - started
    pruning_candidates = [
        rule.key
        for rule, share in zip(current.rules, current.relative_authority_shares)
        if share < config.prune_relative_authority
    ]
    return current, StructureResult(
        initial_rule_count=initial_count,
        final_rule_count=len(current.rules),
        accepted_rules=accepted,
        pruning_candidates=pruning_candidates,
        rounds=rounds,
        stop_reason=stop_reason,
        elapsed_seconds=float(elapsed),
    )
