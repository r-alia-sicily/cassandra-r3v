"""High-level, artifact-producing Cassandra operations."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .artifacts import ArtifactStore, atomic_json, utc_now
from .cancellation import CancellationToken, NEVER_CANCEL
from .config import TrainingConfig
from .data import PreparedDataset
from .fuzzy import FuzzyVocabulary
from .losses import nasa_penalty
from .model import R3vModel
from .rules import SemanticRule, unary_backbone


COMPARISON_BOOTSTRAP_SAMPLES = 20_000
COMPARISON_BOOTSTRAP_SEED = 20_260_916


def train_prepared(
    dataset: PreparedDataset,
    config: TrainingConfig,
    rules: Iterable[SemanticRule] | None = None,
    feature_names: Iterable[str] | None = None,
    token: CancellationToken = NEVER_CANCEL,
    progress: Callable[[float, str], None] | None = None,
) -> R3vModel:
    selected = tuple(feature_names or dataset.feature_names)
    indices = [dataset.feature_names.index(name) for name in selected]
    partition_x = dataset.train_x[:, indices]
    fit_x = dataset.fit_x[:, indices]
    vocabulary = FuzzyVocabulary.fit(selected, partition_x)
    rule_list = tuple(rules) if rules is not None else tuple(unary_backbone(selected))
    model = R3vModel(vocabulary, rule_list, (0.0, dataset.target_upper_bound), config)
    model.metadata = {
        "subset": dataset.subset,
        "protocol": dataset.protocol,
        "feature_names": list(selected),
        "partition_fit_scope": "complete TRAIN trajectories",
        "parameter_fit_scope": "TRAIN-only pseudo-endpoints",
        "pseudo_endpoint_fractions": dataset.metadata["pseudo_endpoint_fractions"],
        "created_at": utc_now(),
    }
    model.fit(fit_x, dataset.fit_y, token=token, progress=progress)
    return model


def save_trained_model(store: ArtifactStore, model: R3vModel) -> str:
    subset = str(model.metadata["subset"])
    protocol = str(model.metadata["protocol"])
    return str(model.save(store.model_path(subset, protocol)))


def evaluate_prepared(model: R3vModel, dataset: PreparedDataset) -> dict[str, Any]:
    names = tuple(model.vocabulary.feature_names)
    indices = [dataset.feature_names.index(name) for name in names]
    result = model.evaluate(dataset.test_endpoint_x[:, indices], dataset.test_endpoint_y)
    result.update(
        {
            "schema": 1,
            "subset": dataset.subset,
            "protocol": dataset.protocol,
            "scope": "official TEST endpoint, one row per engine",
            "model_rules": len(model.rules),
            "interaction_rules": sum(rule.order > 1 for rule in model.rules),
            "feature_names": list(names),
            "created_at": utc_now(),
        }
    )
    return result


def save_evaluation(store: ArtifactStore, result: dict[str, Any]) -> str:
    path = store.evaluation_path(str(result["subset"]), str(result["protocol"]))
    atomic_json(path, result)
    return str(path)


def _paired_bootstrap(
    backbone_penalties: np.ndarray,
    final_penalties: np.ndarray,
    *,
    samples: int,
    seed: int,
    token: CancellationToken,
) -> dict[str, Any]:
    """Bootstrap paired TEST engines without ever breaking the model pairing."""

    parent = np.asarray(backbone_penalties, dtype=float)
    final = np.asarray(final_penalties, dtype=float)
    if parent.shape != final.shape or parent.ndim != 1 or parent.size == 0:
        raise ValueError("paired penalties must be non-empty one-dimensional arrays of equal length")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")

    generator = np.random.default_rng(int(seed))
    absolute = np.empty(int(samples), dtype=float)
    relative = np.empty(int(samples), dtype=float)
    chunk_size = min(1_000, int(samples))
    for start in range(0, int(samples), chunk_size):
        token.checkpoint()
        stop = min(start + chunk_size, int(samples))
        indices = generator.integers(0, parent.size, size=(stop - start, parent.size))
        parent_totals = np.sum(parent[indices], axis=1)
        final_totals = np.sum(final[indices], axis=1)
        improvement = parent_totals - final_totals
        absolute[start:stop] = improvement
        relative[start:stop] = np.divide(
            100.0 * improvement,
            parent_totals,
            out=np.zeros_like(improvement),
            where=np.abs(parent_totals) > 1e-15,
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
        return {
            "median": float(median),
            "ci_95_low": float(low),
            "ci_95_high": float(high),
        }

    return {
        "method": "paired engine-level percentile bootstrap",
        "samples": int(samples),
        "seed": int(seed),
        "confidence_level": 0.95,
        "absolute_nasa_improvement": interval(absolute),
        "relative_nasa_improvement_percent": interval(relative),
        "favorable_resamples_percent": float(100.0 * np.mean(absolute > 0.0)),
    }


def compare_prepared_models(
    backbone: R3vModel,
    final_model: R3vModel,
    dataset: PreparedDataset,
    *,
    bootstrap_samples: int = COMPARISON_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = COMPARISON_BOOTSTRAP_SEED,
    token: CancellationToken = NEVER_CANCEL,
) -> dict[str, Any]:
    """Compare frozen backbone and final models on identical official TEST engines.

    The comparison is evaluative only: it never trains, modifies, accepts, or
    rejects a rule. Positive NASA improvement means that the final score is
    lower than the backbone score.
    """

    token.checkpoint()
    if backbone.metadata.get("subset") not in (None, dataset.subset):
        raise ValueError("backbone model identifies a different subset")
    if final_model.metadata.get("subset") not in (None, dataset.subset):
        raise ValueError("final model identifies a different subset")
    if backbone.metadata.get("protocol") not in (None, dataset.protocol):
        raise ValueError("backbone model identifies a different protocol")
    if final_model.metadata.get("protocol") not in (None, dataset.protocol):
        raise ValueError("final model identifies a different protocol")

    backbone_result = evaluate_prepared(backbone, dataset)
    final_result = evaluate_prepared(final_model, dataset)
    truth = np.asarray(dataset.test_endpoint_y, dtype=float)
    backbone_prediction = np.asarray(backbone_result["predictions"], dtype=float)
    final_prediction = np.asarray(final_result["predictions"], dtype=float)
    if not (truth.shape == backbone_prediction.shape == final_prediction.shape):
        raise ValueError("backbone, final, and truth endpoint counts differ")

    backbone_penalty = nasa_penalty(backbone_prediction - truth)
    final_penalty = nasa_penalty(final_prediction - truth)
    nasa_improvement = float(backbone_result["nasa_score"] - final_result["nasa_score"])
    backbone_nasa = float(backbone_result["nasa_score"])
    nasa_improvement_percent = (
        100.0 * nasa_improvement / backbone_nasa if abs(backbone_nasa) > 1e-15 else 0.0
    )
    bootstrap = _paired_bootstrap(
        backbone_penalty,
        final_penalty,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        token=token,
    )
    interval = bootstrap["relative_nasa_improvement_percent"]
    exact_no_change = bool(np.array_equal(backbone_prediction, final_prediction))
    if exact_no_change:
        conclusion = "no_change"
    elif interval["ci_95_low"] > 0.0:
        conclusion = "supported_improvement"
    elif interval["ci_95_high"] < 0.0:
        conclusion = "supported_degradation"
    else:
        conclusion = "inconclusive"

    backbone_keys = {rule.key for rule in backbone.rules}
    final_keys = {rule.key for rule in final_model.rules}
    tolerance = 1e-12
    return {
        "schema": 1,
        "subset": dataset.subset,
        "protocol": dataset.protocol,
        "scope": "paired official TEST endpoints, one row per engine",
        "selection_scope": "evaluation only; structural selection remains TRAIN-only",
        "created_at": utc_now(),
        "backbone": {
            "rules": len(backbone.rules),
            "interactions": sum(rule.order > 1 for rule in backbone.rules),
            **{key: backbone_result[key] for key in ("nasa_score", "mae", "rmse", "r2", "bias", "overestimation_percent")},
        },
        "final_model": {
            "rules": len(final_model.rules),
            "interactions": sum(rule.order > 1 for rule in final_model.rules),
            **{key: final_result[key] for key in ("nasa_score", "mae", "rmse", "r2", "bias", "overestimation_percent")},
        },
        "structure_difference": {
            "added_rules": sorted(final_keys - backbone_keys),
            "removed_rules": sorted(backbone_keys - final_keys),
        },
        "change": {
            "nasa_improvement": nasa_improvement,
            "nasa_improvement_percent": float(nasa_improvement_percent),
            "mae_final_minus_backbone": float(final_result["mae"] - backbone_result["mae"]),
            "rmse_final_minus_backbone": float(final_result["rmse"] - backbone_result["rmse"]),
            "r2_final_minus_backbone": float(final_result["r2"] - backbone_result["r2"]),
            "bias_final_minus_backbone": float(final_result["bias"] - backbone_result["bias"]),
            "bootstrap_conclusion": conclusion,
        },
        "paired_engine_counts": {
            "engines": int(truth.size),
            "lower_nasa_penalty": int(np.sum(final_penalty < backbone_penalty - tolerance)),
            "equal_nasa_penalty": int(np.sum(np.abs(final_penalty - backbone_penalty) <= tolerance)),
            "higher_nasa_penalty": int(np.sum(final_penalty > backbone_penalty + tolerance)),
            "lower_absolute_error": int(
                np.sum(np.abs(final_prediction - truth) < np.abs(backbone_prediction - truth) - tolerance)
            ),
            "equal_absolute_error": int(
                np.sum(np.abs(np.abs(final_prediction - truth) - np.abs(backbone_prediction - truth)) <= tolerance)
            ),
            "higher_absolute_error": int(
                np.sum(np.abs(final_prediction - truth) > np.abs(backbone_prediction - truth) + tolerance)
            ),
        },
        "bootstrap": bootstrap,
        "paired_endpoints": {
            "engine_ids": np.asarray(dataset.test_units, dtype=int).tolist(),
            "truth": truth.tolist(),
            "backbone_predictions": backbone_prediction.tolist(),
            "final_predictions": final_prediction.tolist(),
            "backbone_nasa_penalties": backbone_penalty.tolist(),
            "final_nasa_penalties": final_penalty.tolist(),
        },
    }


def save_comparison(store: ArtifactStore, result: dict[str, Any]) -> str:
    path = store.comparison_path(str(result["subset"]), str(result["protocol"]))
    atomic_json(path, result)
    return str(path)


def explain_test_engine(model: R3vModel, dataset: PreparedDataset, engine_id: int, top_k: int = 10) -> dict[str, Any]:
    matches = np.flatnonzero(dataset.test_units == int(engine_id))
    if not len(matches):
        raise ValueError(f"engine {engine_id} is not present in {dataset.subset}")
    index = int(matches[0])
    names = tuple(model.vocabulary.feature_names)
    columns = [dataset.feature_names.index(name) for name in names]
    result = model.explain_one(dataset.test_endpoint_x[index, columns], top_k=top_k)
    result.update(
        {
            "subset": dataset.subset,
            "protocol": dataset.protocol,
            "engine_id": int(engine_id),
            "true_rul": float(dataset.test_endpoint_y[index]),
            "note": "Faithful prediction decomposition; this is not a causal explanation.",
        }
    )
    return result
