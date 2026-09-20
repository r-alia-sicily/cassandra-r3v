"""Zero-order normalized Takagi--Sugeno model with explicit R3v semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ._version import __version__
from .cancellation import CancellationToken, NEVER_CANCEL
from .config import TrainingConfig
from .fuzzy import FuzzyVocabulary
from .losses import task_loss_and_gradient
from .metrics import regression_metrics, responsibility_diagnostics
from .rules import SemanticRule, activation_matrix


@dataclass
class FitResult:
    epochs_completed: int
    stopped_early: bool
    best_epoch: int
    best_objective: float
    history: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stopped_early,
            "best_epoch": self.best_epoch,
            "best_objective": self.best_objective,
            "history": self.history,
        }


class _Adam:
    def __init__(self, shape: tuple[int, ...], learning_rate: float):
        self.learning_rate = float(learning_rate)
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)
        self.t = 0

    def step(self, values: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * (gradient**2)
        m_hat = self.m / (1.0 - 0.9**self.t)
        v_hat = self.v / (1.0 - 0.999**self.t)
        return values - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def _clip_norm(gradient: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(gradient))
    return gradient * (maximum / norm) if norm > maximum else gradient


class R3vModel:
    """R3v-compatible normalized Sugeno model.

    For rule ``i`` the public quantities are:

    * antecedent activation ``mu_i(x)``;
    * zero-order consequent ``c_i``;
    * positive global relative authority ``a_i``;
    * normalized local responsibility ``beta_i(x)``.

    Raw authorities are represented by centered log-authorities. They are not
    constrained to sum to one; their common positive scale is immaterial.
    """

    SCHEMA = 2

    def __init__(
        self,
        vocabulary: FuzzyVocabulary,
        rules: Iterable[SemanticRule],
        target_bounds: tuple[float, float],
        config: TrainingConfig | None = None,
    ):
        self.vocabulary = vocabulary
        self.rules = tuple(rules)
        if not self.rules:
            raise ValueError("R3vModel needs at least one rule")
        self.target_bounds = (float(target_bounds[0]), float(target_bounds[1]))
        if not self.target_bounds[0] < self.target_bounds[1]:
            raise ValueError("target bounds must be increasing")
        self.config = (config or TrainingConfig()).validate()
        self.theta = np.zeros(len(self.rules), dtype=float)
        midpoint = 0.5 * (self.target_bounds[0] + self.target_bounds[1])
        self.consequents = np.full(len(self.rules), midpoint, dtype=float)
        self.fit_result: FitResult | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def authorities(self) -> np.ndarray:
        centered = self.theta - float(np.mean(self.theta))
        return np.exp(np.clip(centered, -40.0, 40.0))

    @property
    def relative_authority_shares(self) -> np.ndarray:
        authority = self.authorities
        return authority / np.sum(authority)

    def _mu(self, x: np.ndarray) -> np.ndarray:
        return activation_matrix(np.asarray(x, dtype=float), self.vocabulary, self.rules)

    def _from_mu(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weighted = np.asarray(mu, dtype=float) * self.authorities.reshape(1, -1)
        denominator = np.sum(weighted, axis=1)
        if np.any(denominator <= self.config.denominator_epsilon):
            count = int(np.sum(denominator <= self.config.denominator_epsilon))
            raise ValueError(f"{count} samples are not covered by any active rule")
        beta = weighted / denominator.reshape(-1, 1)
        return beta @ self.consequents, beta

    def predict(self, x: np.ndarray, return_beta: bool = False):
        prediction, beta = self._from_mu(self._mu(x))
        return (prediction, beta) if return_beta else prediction

    def _centered_ridge_initialization(self, beta: np.ndarray, y: np.ndarray) -> np.ndarray:
        center = float(np.mean(y))
        target = np.asarray(y, dtype=float) - center
        lhs = beta.T @ beta + self.config.ridge_lambda * np.eye(beta.shape[1])
        rhs = beta.T @ target
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(lhs) @ rhs
        return np.clip(center + delta, *self.target_bounds)

    def _objective_and_gradients(
        self, mu: np.ndarray, y: np.ndarray, target_center: float
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        prediction, beta = self._from_mu(mu)
        task_loss, d_prediction = task_loss_and_gradient(self.config.loss, prediction - y)
        centered_c = self.consequents - target_center
        omega_c = 0.5 * self.config.ridge_lambda * float(np.mean(centered_c**2))
        centered_theta = self.theta - float(np.mean(self.theta))
        omega_a = 0.5 * self.config.authority_l2 * float(np.mean(centered_theta**2))

        grad_c = beta.T @ d_prediction
        grad_c += self.config.ridge_lambda * centered_c / len(centered_c)
        sensitivity = beta * (self.consequents.reshape(1, -1) - prediction.reshape(-1, 1))
        grad_theta = sensitivity.T @ d_prediction
        grad_theta += self.config.authority_l2 * centered_theta / len(centered_theta)
        grad_theta -= float(np.mean(grad_theta))
        return task_loss + omega_c + omega_a, task_loss, grad_c, grad_theta, beta

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        token: CancellationToken = NEVER_CANCEL,
        progress: Callable[[float, str], None] | None = None,
    ) -> FitResult:
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != target.size or target.size == 0:
            raise ValueError("X and y must contain the same non-zero number of samples")
        if matrix.shape[1] != len(self.vocabulary.feature_names):
            raise ValueError("X does not match the fuzzy vocabulary")
        if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(target)):
            raise ValueError("training data contain non-finite values")
        if np.any(target < self.target_bounds[0]) or np.any(target > self.target_bounds[1]):
            raise ValueError("training targets violate the declared semantic bounds")

        token.checkpoint()
        mu = self._mu(matrix)
        _, initial_beta = self._from_mu(mu)
        self.consequents = self._centered_ridge_initialization(initial_beta, target)
        target_center = float(np.mean(target))

        adam_c = _Adam(self.consequents.shape, self.config.learning_rate_consequents)
        adam_theta = _Adam(self.theta.shape, self.config.learning_rate_authorities)
        history: list[dict[str, float | int]] = []
        best_objective = float("inf")
        best_epoch = 0
        best_c = self.consequents.copy()
        best_theta = self.theta.copy()
        stale = 0
        stopped_early = False

        for epoch in range(1, self.config.epochs + 1):
            token.checkpoint()
            objective, task_loss, grad_c, _, _ = self._objective_and_gradients(mu, target, target_center)
            grad_c = _clip_norm(grad_c, self.config.gradient_clip)
            self.consequents = np.clip(adam_c.step(self.consequents, grad_c), *self.target_bounds)

            objective, task_loss, _, grad_theta, beta = self._objective_and_gradients(mu, target, target_center)
            if self.config.authority_mode == "learned":
                grad_theta = _clip_norm(grad_theta, self.config.gradient_clip)
                self.theta = adam_theta.step(self.theta, grad_theta)
                self.theta -= float(np.mean(self.theta))
            else:
                self.theta.fill(0.0)

            objective, task_loss, grad_c, grad_theta, beta = self._objective_and_gradients(mu, target, target_center)
            prediction = beta @ self.consequents
            error = prediction - target
            record: dict[str, float | int] = {
                "epoch": epoch,
                "objective": float(objective),
                "task_loss": float(task_loss),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "consequent_gradient_norm": float(np.linalg.norm(grad_c)),
                "authority_gradient_norm": float(np.linalg.norm(grad_theta)),
                "min_consequent": float(np.min(self.consequents)),
                "max_consequent": float(np.max(self.consequents)),
            }
            history.append(record)

            if objective < best_objective - self.config.min_delta:
                best_objective = float(objective)
                best_epoch = epoch
                best_c = self.consequents.copy()
                best_theta = self.theta.copy()
                stale = 0
            else:
                stale += 1

            if progress and (epoch == 1 or epoch % 5 == 0 or epoch == self.config.epochs):
                progress(
                    epoch / self.config.epochs,
                    f"Epoch {epoch}/{self.config.epochs} · loss={task_loss:.5g} · MAE={record['mae']:.3f}",
                )
            if stale >= self.config.patience:
                stopped_early = True
                break

        self.consequents = best_c
        self.theta = best_theta
        self.fit_result = FitResult(
            epochs_completed=len(history),
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            best_objective=best_objective,
            history=history,
        )
        return self.fit_result

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        prediction, beta = self.predict(x, return_beta=True)
        result: dict[str, Any] = regression_metrics(y, prediction)
        result["responsibility"] = responsibility_diagnostics(beta)
        result["predictions"] = prediction.tolist()
        result["truth"] = np.asarray(y, dtype=float).tolist()
        return result

    def explain_one(self, x: np.ndarray, top_k: int = 10) -> dict[str, Any]:
        row = np.asarray(x, dtype=float).reshape(1, -1)
        mu = self._mu(row)[0]
        prediction, beta_matrix = self._from_mu(mu.reshape(1, -1))
        beta = beta_matrix[0]
        authority = self.authorities
        details = []
        for index, rule in enumerate(self.rules):
            details.append(
                {
                    "rule": rule.key,
                    "order": rule.order,
                    "activation_mu": float(mu[index]),
                    "relative_authority_a": float(authority[index]),
                    "responsibility_beta": float(beta[index]),
                    "consequent_c": float(self.consequents[index]),
                    "contribution_beta_c": float(beta[index] * self.consequents[index]),
                }
            )
        details.sort(key=lambda item: item["responsibility_beta"], reverse=True)
        return {
            "prediction": float(prediction[0]),
            "responsibility_sum": float(np.sum(beta)),
            "reconstruction": float(np.sum(beta * self.consequents)),
            "top_rules": details[: max(1, int(top_k))],
            "causal_explanation": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "application": "Cassandra R3v",
            "version": __version__,
            "vocabulary": self.vocabulary.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "target_bounds": list(self.target_bounds),
            "config": self.config.to_dict(),
            "theta": self.theta.tolist(),
            "consequents": self.consequents.tolist(),
            "fit_result": self.fit_result.to_dict() if self.fit_result else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "R3vModel":
        if int(values.get("schema", 0)) != cls.SCHEMA:
            raise ValueError("unsupported Cassandra model schema")
        model = cls(
            vocabulary=FuzzyVocabulary.from_dict(values["vocabulary"]),
            rules=[SemanticRule.from_dict(rule) for rule in values["rules"]],
            target_bounds=tuple(values["target_bounds"]),
            config=TrainingConfig.from_dict(values["config"]),
        )
        model.theta = np.asarray(values["theta"], dtype=float)
        model.consequents = np.asarray(values["consequents"], dtype=float)
        fit_values = values.get("fit_result")
        if fit_values:
            model.fit_result = FitResult(**fit_values)
        model.metadata = dict(values.get("metadata", {}))
        return model

    def save(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: Path) -> "R3vModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
