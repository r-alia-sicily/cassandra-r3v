"""Task losses used by R3v.

The NASA score is the declared primary objective for the C-MAPSS instance.
MAE, RMSE, R2 and bias are diagnostics and intentionally live elsewhere.
"""

from __future__ import annotations

import numpy as np


_MAX_EXPONENT = 50.0


def nasa_penalty(errors: np.ndarray) -> np.ndarray:
    """Return the NASA asymmetric penalty for errors ``prediction - truth``.

    Clipping only prevents floating-point overflow. It is far beyond the useful
    optimization range and does not change ordinary C-MAPSS values.
    """

    error = np.asarray(errors, dtype=float)
    under = np.exp(np.clip(-error / 13.0, None, _MAX_EXPONENT)) - 1.0
    over = np.exp(np.clip(error / 10.0, None, _MAX_EXPONENT)) - 1.0
    return np.where(error < 0.0, under, over)


def nasa_score(errors: np.ndarray) -> float:
    """Standard endpoint NASA score: sum of per-engine penalties."""

    return float(np.sum(nasa_penalty(errors)))


def nasa_mean_loss_and_gradient(errors: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean NASA loss and derivative with respect to each prediction."""

    error = np.asarray(errors, dtype=float)
    n = max(1, error.size)
    under_exp = np.exp(np.clip(-error / 13.0, None, _MAX_EXPONENT))
    over_exp = np.exp(np.clip(error / 10.0, None, _MAX_EXPONENT))
    penalty = np.where(error < 0.0, under_exp - 1.0, over_exp - 1.0)
    derivative = np.where(error < 0.0, -under_exp / 13.0, over_exp / 10.0)
    derivative = np.where(error == 0.0, 0.0, derivative)
    return float(np.mean(penalty)), derivative / n


def mse_mean_loss_and_gradient(errors: np.ndarray) -> tuple[float, np.ndarray]:
    error = np.asarray(errors, dtype=float)
    n = max(1, error.size)
    return 0.5 * float(np.mean(error**2)), error / n


def task_loss_and_gradient(name: str, errors: np.ndarray) -> tuple[float, np.ndarray]:
    if name == "nasa":
        return nasa_mean_loss_and_gradient(errors)
    if name == "mse":
        return mse_mean_loss_and_gradient(errors)
    raise ValueError(f"unknown task loss: {name}")
