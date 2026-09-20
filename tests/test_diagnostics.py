from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cassandra_r3v._version import __version__
from cassandra_r3v.artifacts import ArtifactStore, atomic_json
from cassandra_r3v.cancellation import CancellationToken, CancelledError
from cassandra_r3v.data import DatasetRepository
from cassandra_r3v.diagnostics import _ReportRecorder, _version_tuple, run_full_test
from cassandra_r3v.gui.app import launch
from cassandra_r3v.gui.tabs import (
    AutoStructureTab,
    DataTab,
    EvaluateTab,
    FullTestTab,
    PrepareTab,
    TrainTab,
)
from cassandra_r3v.gui.widgets import StageStrip


class DiagnosticUtilityTests(unittest.TestCase):
    def test_canonical_cassandra_banner_is_stable_and_readable(self):
        stream = io.StringIO()
        workspace = Path("/tmp/cassandra-banner-test")
        with patch("cassandra_r3v.gui.app.CassandraApp") as application:
            with redirect_stdout(stream):
                launch(workspace)
        application.assert_called_once_with(workspace=workspace)
        application.return_value.run.assert_called_once_with()
        expected = rf"""
   ___   _   ___ ___   _   _  _ ___  ___    _
  / __| /_\ / __/ __| /_\ | \| |   \| _ \  /_\
 | (__ / _ \\__ \__ \/ _ \| .` | |) |   / / _ \
  \___/_/ \_\___/___/_/ \_\_|\_|___/|_|_\/_/ \_\

        Rul3volution Neuro-Fuzzy Prognostic System
          guarded six-stage publication workflow
============================================================

                 CASSANDRA SUGENO / R3v {__version__}

"""
        self.assertEqual(stream.getvalue(), expected)

    def test_version_tuple_ignores_non_numeric_suffixes(self):
        self.assertEqual(_version_tuple("2.1.0rc1"), (2, 1, 0))

    def test_report_status_reflects_failures(self):
        recorder = _ReportRecorder()
        recorder.add("PASS", "first")
        recorder.add("FAIL", "second")
        self.assertEqual(recorder.default_overall_status(), "FAIL")

    def test_full_test_saves_report_without_nasa_data(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            store = ArtifactStore(workspace)
            repository = DatasetRepository(workspace)
            atomic_json(
                store.structure_path("FD001", "uncapped"),
                {
                    "schema": 1,
                    "subset": "FD001",
                    "protocol": "uncapped",
                    "result": {
                        "initial_rule_count": 120,
                        "final_rule_count": 120,
                        "accepted_rules": [],
                        "stop_reason": "test fixture",
                    },
                },
            )
            result = run_full_test(
                project_root=Path(directory),
                repository=repository,
                store=store,
                selected_subsets=("FD001",),
                selected_protocols=("uncapped", "cap125"),
                include_source_tests=False,
                include_pipeline_smoke=False,
            )
            report = Path(result.report_path)
            self.assertTrue(report.is_file())
            text = report.read_text(encoding="utf-8")
            self.assertIn("CASSANDRA R3V — FULL TEST REPORT", text)
            self.assertIn("OVERALL STATUS:", text)
            self.assertIn("FD001/uncapped", text)
            self.assertIn("[PASS] Structure artifact FD001/uncapped", text)
            self.assertIn("Selected raw-data inventory: 0/3 expected files present", text)
            self.assertNotIn("FD002/", text)
            self.assertNotEqual(result.overall_status, "FAIL")

    def test_cancelled_full_test_saves_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            store = ArtifactStore(workspace)
            repository = DatasetRepository(workspace)
            token = CancellationToken()
            token.cancel()
            with self.assertRaises(CancelledError):
                run_full_test(
                    project_root=Path(directory),
                    repository=repository,
                    store=store,
                    selected_subsets=("FD001",),
                    selected_protocols=("uncapped",),
                    token=token,
                    include_source_tests=False,
                    include_pipeline_smoke=False,
                )
            report = store.latest_full_test_report()
            self.assertIsNotNone(report)
            self.assertIn("OVERALL STATUS: CANCELLED", report.read_text(encoding="utf-8"))

    def test_gui_exposes_sixth_full_test_tab_in_english(self):
        titles = tuple(
            tab.title
            for tab in (DataTab, PrepareTab, TrainTab, AutoStructureTab, EvaluateTab, FullTestTab)
        )
        self.assertEqual(
            titles,
            (
                "1 · Data",
                "2 · Preparation",
                "3 · Model",
                "4 · AutoStructure",
                "5 · Evaluation",
                "6 · Full Test",
            ),
        )
        self.assertEqual(FullTestTab.title, "6 · Full Test")
        self.assertEqual(StageStrip.STAGES[-1], ("full_test", "6  Full Test"))
        self.assertEqual(len(StageStrip.STAGES), 6)


if __name__ == "__main__":
    unittest.main()
