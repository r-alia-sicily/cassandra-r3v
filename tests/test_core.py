from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cassandra_r3v.cancellation import CancellationToken, CancelledError
from cassandra_r3v.config import AutoStructureConfig, TrainingConfig, normalize_protocols, normalize_subsets
from cassandra_r3v.fuzzy import FuzzyPartition, FuzzyVocabulary
from cassandra_r3v.losses import nasa_mean_loss_and_gradient, nasa_penalty, nasa_score
from cassandra_r3v.metrics import regression_metrics, responsibility_diagnostics
from cassandra_r3v.model import R3vModel
from cassandra_r3v.rules import Antecedent, SemanticRule, activation_matrix, compose_rules, unary_backbone


class ConfigurationTests(unittest.TestCase):
    def test_protocol_order_is_canonical(self):
        self.assertEqual(normalize_protocols(["cap125", "uncapped"]), ("uncapped", "cap125"))

    def test_subset_normalization(self):
        self.assertEqual(normalize_subsets(["fd004", "FD001"]), ("FD001", "FD004"))

    def test_empty_protocol_rejected(self):
        with self.assertRaises(ValueError):
            normalize_protocols([])

    def test_parent_threshold_must_be_lower(self):
        with self.assertRaises(ValueError):
            AutoStructureConfig(parent_relative_authority=0.1, prune_relative_authority=0.01).validate()

    def test_bad_loss_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(loss="unknown").validate()


class FuzzyTests(unittest.TestCase):
    def test_partition_of_unity(self):
        partition = FuzzyPartition.fit("x", np.linspace(-2.0, 3.0, 501))
        membership = partition.memberships(np.linspace(-20.0, 20.0, 2001))
        np.testing.assert_allclose(np.sum(membership, axis=1), 1.0, atol=1e-12)

    def test_constant_feature_is_non_degenerate(self):
        partition = FuzzyPartition.fit("constant", np.ones(100) * 4.0)
        self.assertEqual(partition.mode, "expanded_constant")
        self.assertTrue(np.all(np.diff(partition.centers) > 0))
        self.assertAlmostEqual(float(np.sum(partition.memberships(4.0))), 1.0)

    def test_extreme_shoulders_cover_domain(self):
        partition = FuzzyPartition.fit("x", np.arange(10.0))
        self.assertAlmostEqual(partition.membership("VERY_LOW", -1e6), 1.0)
        self.assertAlmostEqual(partition.membership("VERY_HIGH", 1e6), 1.0)

    def test_non_finite_has_zero_membership(self):
        partition = FuzzyPartition.fit("x", np.arange(10.0))
        self.assertEqual(float(np.sum(partition.memberships(np.nan))), 0.0)

    def test_vocabulary_round_trip(self):
        x = np.column_stack([np.linspace(0, 1, 20), np.linspace(2, 5, 20)])
        vocabulary = FuzzyVocabulary.fit(("a", "b"), x)
        restored = FuzzyVocabulary.from_dict(vocabulary.to_dict())
        np.testing.assert_allclose(
            vocabulary.atom_activation(x, "a", "LOW"), restored.atom_activation(x, "a", "LOW")
        )


class LossTests(unittest.TestCase):
    def test_zero_error_has_zero_penalty(self):
        self.assertEqual(nasa_score(np.array([0.0])), 0.0)

    def test_overestimation_is_penalized_more(self):
        penalties = nasa_penalty(np.array([-20.0, 20.0]))
        self.assertGreater(penalties[1], penalties[0])

    def test_gradient_direction(self):
        _, gradient = nasa_mean_loss_and_gradient(np.array([-5.0, 0.0, 5.0]))
        self.assertLess(gradient[0], 0.0)
        self.assertEqual(gradient[1], 0.0)
        self.assertGreater(gradient[2], 0.0)

    def test_nasa_gradient_matches_finite_difference(self):
        error = np.array([-3.0, 4.0])
        _, gradient = nasa_mean_loss_and_gradient(error)
        epsilon = 1e-6
        for index in range(2):
            plus = error.copy(); plus[index] += epsilon
            minus = error.copy(); minus[index] -= epsilon
            numerical = (np.mean(nasa_penalty(plus)) - np.mean(nasa_penalty(minus))) / (2 * epsilon)
            self.assertAlmostEqual(gradient[index], numerical, places=7)


class RuleAndModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.x = rng.normal(size=(120, 2))
        self.y = np.clip(5.0 + 2.0 * self.x[:, 0] - self.x[:, 1], 0.0, 10.0)
        self.vocabulary = FuzzyVocabulary.fit(("a", "b"), self.x)
        self.rules = unary_backbone(("a", "b"))

    def test_rule_rejects_repeated_feature(self):
        with self.assertRaises(ValueError):
            SemanticRule((Antecedent("a", "LOW"), Antecedent("a", "HIGH")))

    def test_composition_is_canonical(self):
        left = SemanticRule((Antecedent("b", "HIGH"),))
        right = SemanticRule((Antecedent("a", "LOW"),))
        self.assertEqual(compose_rules(left, right).key, "a IS LOW AND b IS HIGH")

    def test_unary_activations_cover_each_sample(self):
        mu = activation_matrix(self.x, self.vocabulary, self.rules)
        self.assertTrue(np.all(np.sum(mu, axis=1) > 0.0))

    def test_responsibilities_sum_to_one(self):
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), TrainingConfig(epochs=2))
        _, beta = model.predict(self.x, return_beta=True)
        np.testing.assert_allclose(np.sum(beta, axis=1), 1.0, atol=1e-12)

    def test_authority_common_shift_is_invariant(self):
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), TrainingConfig(epochs=2))
        model.consequents = np.linspace(0, 10, len(self.rules))
        first = model.predict(self.x)
        model.theta += 123.0
        second = model.predict(self.x)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_prediction_respects_consequent_bounds(self):
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), TrainingConfig(epochs=2))
        model.consequents = np.linspace(1, 9, len(self.rules))
        prediction = model.predict(self.x)
        self.assertGreaterEqual(float(np.min(prediction)), 1.0)
        self.assertLessEqual(float(np.max(prediction)), 9.0)

    def test_fit_and_model_round_trip(self):
        config = TrainingConfig(epochs=12, patience=6, ridge_lambda=0.1, learning_rate_consequents=0.02, learning_rate_authorities=0.02)
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), config)
        result = model.fit(self.x, self.y)
        self.assertGreaterEqual(result.epochs_completed, 1)
        self.assertTrue(np.all(model.consequents >= 0.0))
        self.assertTrue(np.all(model.consequents <= 10.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = R3vModel.load(path)
            np.testing.assert_allclose(model.predict(self.x), restored.predict(self.x), atol=1e-12)

    def test_explanation_reconstructs_prediction(self):
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), TrainingConfig(epochs=2))
        explanation = model.explain_one(self.x[0], top_k=len(self.rules))
        self.assertAlmostEqual(explanation["prediction"], explanation["reconstruction"], places=12)
        self.assertAlmostEqual(explanation["responsibility_sum"], 1.0, places=12)

    def test_cancelled_fit_stops_at_checkpoint(self):
        token = CancellationToken(); token.cancel()
        model = R3vModel(self.vocabulary, self.rules, (0.0, 10.0), TrainingConfig(epochs=2))
        with self.assertRaises(CancelledError):
            model.fit(self.x, self.y, token=token)


class MetricTests(unittest.TestCase):
    def test_regression_metrics(self):
        result = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertGreater(result["nasa_score"], 0.0)

    def test_k90_one_hot(self):
        result = responsibility_diagnostics(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(result["median_k90"], 1.0)


if __name__ == "__main__":
    unittest.main()
