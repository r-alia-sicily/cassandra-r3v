"""Predictive and rule-level diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .losses import nasa_score


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(truth, dtype=float)
    y_hat = np.asarray(prediction, dtype=float)
    if y.shape != y_hat.shape or y.size == 0:
        raise ValueError("truth and prediction must be non-empty arrays with equal shape")
    error = y_hat - y
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(y.size),
        "nasa_score": nasa_score(error),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "bias": float(np.mean(error)),
        "overestimation_percent": float(np.mean(error > 0.0) * 100.0),
    }


def responsibility_diagnostics(beta: np.ndarray) -> dict[str, Any]:
    responsibilities = np.asarray(beta, dtype=float)
    if responsibilities.ndim != 2 or responsibilities.shape[0] == 0:
        raise ValueError("beta must be a non-empty two-dimensional matrix")
    ordered = np.sort(responsibilities, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    k90 = np.argmax(cumulative >= 0.9, axis=1) + 1
    entropy = -np.sum(
        np.where(responsibilities > 0, responsibilities * np.log(responsibilities + 1e-300), 0.0),
        axis=1,
    )
    max_entropy = np.log(responsibilities.shape[1]) if responsibilities.shape[1] > 1 else 1.0
    return {
        "median_k90": float(np.median(k90)),
        "mean_k90": float(np.mean(k90)),
        "median_max_beta": float(np.median(np.max(responsibilities, axis=1))),
        "median_normalized_entropy": float(np.median(entropy / max_entropy)),
        "k90_per_sample": k90.tolist(),
    }
