from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cassandra_r3v.artifacts import ArtifactStore
from cassandra_r3v.cancellation import CancellationToken, CancelledError
from cassandra_r3v.config import TrainingConfig
from cassandra_r3v.data import FEATURE_NAMES, DatasetRepository, PreparedDataset
from cassandra_r3v.diagnostics import run_full_test
from cassandra_r3v.experiment import (
    compare_prepared_models,
    evaluate_prepared,
    save_comparison,
    save_evaluation,
)
from cassandra_r3v.fuzzy import FuzzyVocabulary
from cassandra_r3v.model import R3vModel
from cassandra_r3v.rules import unary_backbone


class PairedComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        train_x = np.linspace(-2.0, 2.0, 20).reshape(-1, 1)
        test_x = np.linspace(-1.5, 1.5, 10).reshape(-1, 1)
        self.dataset = PreparedDataset(
            subset="FD003",
            protocol="cap125",
            feature_names=("sensor",),
            raw_train_x=train_x.copy(),
            train_x=train_x,
            train_y=np.linspace(0.0, 10.0, 20),
            train_units=np.repeat([1, 2], 10),
            train_cycles=np.tile(np.arange(1, 11), 2),
            fit_indices=np.arange(20),
            raw_test_endpoint_x=test_x.copy(),
            test_endpoint_x=test_x,
            test_endpoint_y=np.linspace(4.0, 6.0, 10),
            test_units=np.arange(1, 11),
            mean=np.array([0.0]),
            scale=np.array([1.0]),
            metadata={"pseudo_endpoint_fractions": [0.35, 0.5, 0.65, 0.8, 0.92]},
        )
        self.vocabulary = FuzzyVocabulary.fit(("sensor",), train_x)

    def _model(self, consequent: float) -> R3vModel:
        model = R3vModel(
            self.vocabulary,
            unary_backbone(("sensor",)),
            (0.0, 10.0),
            TrainingConfig(epochs=2),
        )
        model.consequents.fill(consequent)
        model.metadata = {"subset": "FD003", "protocol": "cap125"}
        return model

    def _full_dataset_and_model(self, consequent: float) -> tuple[PreparedDataset, R3vModel]:
        train_axis = np.linspace(-2.0, 2.0, 20)
        test_axis = np.linspace(-1.5, 1.5, 10)
        train_x = np.column_stack([train_axis + index * 0.01 for index in range(len(FEATURE_NAMES))])
        test_x = np.column_stack([test_axis + index * 0.01 for index in range(len(FEATURE_NAMES))])
        dataset = PreparedDataset(
            subset="FD003",
            protocol="cap125",
            feature_names=FEATURE_NAMES,
            raw_train_x=train_x.copy(),
            train_x=train_x,
            train_y=np.linspace(0.0, 10.0, 20),
            train_units=np.repeat([1, 2], 10),
            train_cycles=np.tile(np.arange(1, 11), 2),
            fit_indices=np.arange(20),
            raw_test_endpoint_x=test_x.copy(),
            test_endpoint_x=test_x,
            test_endpoint_y=np.linspace(4.0, 6.0, 10),
            test_units=np.arange(1, 11),
            mean=np.zeros(len(FEATURE_NAMES)),
            scale=np.ones(len(FEATURE_NAMES)),
            metadata={"pseudo_endpoint_fractions": [0.35, 0.5, 0.65, 0.8, 0.92]},
        )
        vocabulary = FuzzyVocabulary.fit(FEATURE_NAMES, train_x)
        model = R3vModel(
            vocabulary,
            unary_backbone(FEATURE_NAMES),
            (0.0, 125.0),
            TrainingConfig(epochs=2),
        )
        model.consequents.fill(consequent)
        model.metadata = {"subset": "FD003", "protocol": "cap125"}
        return dataset, model

    def test_identical_models_report_exact_no_change(self):
        result = compare_prepared_models(
            self._model(5.0),
            self._model(5.0),
            self.dataset,
            bootstrap_samples=200,
            bootstrap_seed=17,
        )
        interval = result["bootstrap"]["relative_nasa_improvement_percent"]
        self.assertEqual(result["change"]["bootstrap_conclusion"], "no_change")
        self.assertEqual(result["change"]["nasa_improvement"], 0.0)
        self.assertEqual(interval["ci_95_low"], 0.0)
        self.assertEqual(interval["ci_95_high"], 0.0)

    def test_uniformly_better_final_model_is_supported(self):
        result = compare_prepared_models(
            self._model(9.0),
            self._model(5.0),
            self.dataset,
            bootstrap_samples=400,
            bootstrap_seed=23,
        )
        interval = result["bootstrap"]["relative_nasa_improvement_percent"]
        self.assertEqual(result["change"]["bootstrap_conclusion"], "supported_improvement")
        self.assertGreater(result["change"]["nasa_improvement_percent"], 0.0)
        self.assertGreater(interval["ci_95_low"], 0.0)
        self.assertEqual(result["paired_engine_counts"]["lower_nasa_penalty"], 10)

    def test_bootstrap_is_reproducible(self):
        arguments = {
            "bootstrap_samples": 250,
            "bootstrap_seed": 20260916,
        }
        first = compare_prepared_models(self._model(9.0), self._model(5.0), self.dataset, **arguments)
        second = compare_prepared_models(self._model(9.0), self._model(5.0), self.dataset, **arguments)
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_cancelled_comparison_stops_before_work(self):
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            compare_prepared_models(self._model(9.0), self._model(5.0), self.dataset, token=token)

    def test_full_test_reports_the_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            store = ArtifactStore(workspace)
            dataset, backbone = self._full_dataset_and_model(9.0)
            _, final_model = self._full_dataset_and_model(5.0)
            store.save_prepared(dataset)
            backbone.save(store.backbone_model_path("FD003", "cap125"))
            final_model.save(store.model_path("FD003", "cap125"))
            save_evaluation(store, evaluate_prepared(final_model, dataset))
            save_comparison(
                store,
                compare_prepared_models(
                    backbone,
                    final_model,
                    dataset,
                    bootstrap_samples=200,
                    bootstrap_seed=29,
                ),
            )
            result = run_full_test(
                project_root=Path(directory),
                repository=DatasetRepository(workspace),
                store=store,
                selected_subsets=("FD003",),
                selected_protocols=("cap125",),
                include_source_tests=False,
                include_pipeline_smoke=False,
            )
            self.assertNotEqual(result.overall_status, "FAIL", result.report_text)
            self.assertIn("PAIRED BACKBONE–AUTOSTRUCTURE TEST COMPARISON", result.report_text)
            self.assertIn("[PASS] Paired comparison FD003/cap125", result.report_text)


if __name__ == "__main__":
    unittest.main()
