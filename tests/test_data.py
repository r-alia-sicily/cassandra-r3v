from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra_r3v.artifacts import ArtifactStore, atomic_json
from cassandra_r3v.data import (
    DatasetRepository,
    PreparedDataset,
    add_test_rul,
    add_train_rul,
    apply_protocol,
    pseudo_endpoint_indices,
)


class TargetTests(unittest.TestCase):
    def test_train_rul_ends_at_zero(self):
        frame = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
        result = add_train_rul(frame)
        self.assertEqual(result["RUL"].tolist(), [2, 1, 0, 1, 0])

    def test_test_rul_expands_backwards_from_endpoint(self):
        frame = pd.DataFrame({"unit": [1, 1, 2, 2, 2], "cycle": [1, 2, 1, 2, 3]})
        result = add_test_rul(frame, np.array([5.0, 8.0]))
        self.assertEqual(result["RUL"].tolist(), [6.0, 5.0, 10.0, 9.0, 8.0])

    def test_cap125(self):
        np.testing.assert_array_equal(apply_protocol(np.array([2, 125, 200]), "cap125"), [2, 125, 125])

    def test_uncapped_is_unchanged(self):
        np.testing.assert_array_equal(apply_protocol(np.array([2, 200]), "uncapped"), [2, 200])

    def test_five_pseudo_endpoints_per_engine(self):
        units = np.repeat([1, 2], 10)
        cycles = np.tile(np.arange(1, 11), 2)
        indices = pseudo_endpoint_indices(units, cycles)
        self.assertEqual(len(indices), 10)
        self.assertEqual(set(units[indices]), {1, 2})


class RepositoryTests(unittest.TestCase):
    def _nested_archive(self, path: Path) -> None:
        inner_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_buffer, "w") as inner:
            for prefix in ("train", "test", "RUL"):
                inner.writestr(f"{prefix}_FD001.txt", "1 1 0\n")
        with zipfile.ZipFile(path, "w") as outer:
            outer.writestr("folder/CMAPSSData.zip", inner_buffer.getvalue())

    def test_nested_nasa_archive_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "outer.zip"
            self._nested_archive(archive)
            repository = DatasetRepository(root / "workspace")
            result = repository.import_archive(archive)
            self.assertTrue(result["subsets"]["FD001"])
            self.assertFalse(result["complete"])

    def test_unrecognized_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "empty.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("hello.txt", "nothing")
            repository = DatasetRepository(root / "workspace")
            with self.assertRaises(ValueError):
                repository.import_archive(archive)


class PreparedRoundTripTests(unittest.TestCase):
    def _dataset(self) -> PreparedDataset:
        return PreparedDataset(
            subset="FD001",
            protocol="cap125",
            feature_names=("a", "b"),
            raw_train_x=np.arange(20, dtype=float).reshape(10, 2),
            train_x=np.arange(20, dtype=float).reshape(10, 2) / 10,
            train_y=np.arange(10, dtype=float),
            train_units=np.repeat([1, 2], 5),
            train_cycles=np.tile(np.arange(1, 6), 2),
            fit_indices=np.array([1, 3, 6, 8]),
            raw_test_endpoint_x=np.ones((2, 2)),
            test_endpoint_x=np.zeros((2, 2)),
            test_endpoint_y=np.array([4.0, 6.0]),
            test_units=np.array([1, 2]),
            mean=np.array([1.0, 2.0]),
            scale=np.array([3.0, 4.0]),
            metadata={"pseudo_endpoint_fractions": [0.35, 0.5, 0.65, 0.8, 0.92]},
        )

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.npz"
            source = self._dataset(); source.save(path)
            restored = PreparedDataset.load(path)
            self.assertEqual(restored.feature_names, source.feature_names)
            np.testing.assert_allclose(restored.fit_x, source.fit_x)

    def test_downstream_invalidation_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            atomic_json(store.model_path("FD001", "cap125"), {"model": 1})
            atomic_json(store.backbone_model_path("FD001", "cap125"), {"backbone": 1})
            atomic_json(store.evaluation_path("FD001", "cap125"), {"evaluation": 1})
            atomic_json(store.comparison_path("FD001", "cap125"), {"comparison": 1})
            archived = store.invalidate_after("prepared", "FD001", "cap125")
            self.assertEqual(len(archived), 4)
            self.assertFalse(store.model_path("FD001", "cap125").exists())
            self.assertFalse(store.backbone_model_path("FD001", "cap125").exists())
            self.assertFalse(store.comparison_path("FD001", "cap125").exists())
            self.assertTrue(all(Path(path).exists() for path in archived))


if __name__ == "__main__":
    unittest.main()
