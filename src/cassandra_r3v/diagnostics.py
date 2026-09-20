"""Comprehensive, non-destructive diagnostics for Cassandra releases."""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import platform
import re
import sys
import tempfile
import traceback
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ._version import __version__
from .artifacts import ArtifactStore
from .autostructure import grow_structure
from .cancellation import CancellationToken, CancelledError, NEVER_CANCEL
from .config import (
    AutoStructureConfig,
    ComputeBudget,
    PROTOCOLS,
    SUBSETS,
    TrainingConfig,
    normalize_protocols,
    normalize_subsets,
)
from .data import (
    FEATURE_NAMES,
    DatasetRepository,
    PreparedDataset,
    expected_files,
    prepare_subset,
    read_subset,
    sha256_file,
)
from .experiment import compare_prepared_models, evaluate_prepared, train_prepared
from .fuzzy import FuzzyPartition, FuzzyVocabulary
from .losses import nasa_penalty
from .model import R3vModel
from .rules import unary_backbone


StatusLog = Callable[[str, str], None]
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class FullTestResult:
    """Result returned to the GUI and the command-line interface."""

    report_path: str
    overall_status: str
    counts: dict[str, int]
    report_text: str


class _ReportRecorder:
    def __init__(self, log: StatusLog | None = None):
        self.lines: list[str] = []
        self.counts = {"PASS": 0, "WARN": 0, "SKIP": 0, "FAIL": 0}
        self.log = log

    def section(self, title: str) -> None:
        self.lines.extend(("", f"==== {title} ===="))
        if self.log:
            self.log(f"\n{title}", "muted")

    def add(self, status: str, name: str, detail: str = "") -> None:
        normalized = status.upper()
        if normalized not in self.counts:
            raise ValueError(f"unsupported diagnostic status: {status}")
        self.counts[normalized] += 1
        suffix = f": {detail}" if detail else ""
        line = f"[{normalized}] {name}{suffix}"
        self.lines.append(line)
        if self.log:
            tag = {"PASS": "success", "WARN": "warning", "SKIP": "muted", "FAIL": "error"}[normalized]
            self.log(line, tag)

    def detail(self, text: str) -> None:
        for line in str(text).rstrip().splitlines():
            rendered = f"    {line}"
            self.lines.append(rendered)
            if self.log:
                self.log(rendered, "muted")

    def default_overall_status(self) -> str:
        if self.counts["FAIL"]:
            return "FAIL"
        if self.counts["WARN"]:
            return "PASS WITH WARNINGS"
        if self.counts["SKIP"]:
            return "PASS WITH SKIPS"
        return "PASS"

    def render(
        self,
        *,
        started_at: str,
        finished_at: str,
        overall_status: str,
        selected_subsets: Iterable[str],
        selected_protocols: Iterable[str],
        workspace_name: str,
    ) -> str:
        counts = " · ".join(f"{key}={self.counts[key]}" for key in ("PASS", "WARN", "SKIP", "FAIL"))
        header = [
            "CASSANDRA R3V — FULL TEST REPORT",
            "=" * 39,
            f"Application version: {__version__}",
            f"Started (UTC): {started_at}",
            f"Finished (UTC): {finished_at}",
            f"Workspace: {workspace_name}",
            f"Selected subsets: {', '.join(selected_subsets)}",
            f"Selected protocols: {', '.join(selected_protocols)}",
            "Protocol rule: uncapped and cap125 are tested and reported separately.",
            "Smoke-test metrics are software diagnostics, not publication results.",
            "",
            f"OVERALL STATUS: {overall_status}",
            f"SUMMARY: {counts}",
        ]
        footer = [
            "",
            "==== END OF REPORT ====",
            "Send this complete TXT file when requesting development diagnostics.",
        ]
        return "\n".join(header + self.lines + footer) + "\n"


def _utc_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unique_report_path(store: ArtifactStore) -> Path:
    stamp = _utc_stamp()
    candidate = store.full_test_report_path(stamp)
    counter = 1
    while candidate.exists():
        candidate = store.full_test_report_path(f"{stamp}_{counter:02d}")
        counter += 1
    return candidate


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)
    return path


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(number) for number in numbers[:3])


def _check_dependencies(recorder: _ReportRecorder, token: CancellationToken) -> None:
    recorder.section("RUNTIME AND DEPENDENCIES")
    token.checkpoint()
    python_ok = sys.version_info >= (3, 10)
    recorder.add(
        "PASS" if python_ok else "FAIL",
        "Python runtime",
        f"{platform.python_version()} · {platform.python_implementation()} · {platform.machine() or 'unknown architecture'}",
    )
    recorder.add("PASS", "Operating system", platform.platform())

    for module_name, distribution_name, minimum in (
        ("numpy", "numpy", (1, 24)),
        ("pandas", "pandas", (2, 0)),
    ):
        token.checkpoint()
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution_name)
            ok = _version_tuple(version) >= minimum
            recorder.add(
                "PASS" if ok else "FAIL",
                f"{distribution_name} dependency",
                f"version {version}; required >= {'.'.join(map(str, minimum))}",
            )
        except Exception as exc:
            recorder.add("FAIL", f"{distribution_name} dependency", str(exc))

    try:
        tkinter = importlib.import_module("tkinter")
        recorder.add("PASS", "Tkinter dependency", f"Tk {tkinter.TkVersion}")
    except Exception as exc:
        recorder.add("FAIL", "Tkinter dependency", str(exc))


def _check_source_integrity(
    recorder: _ReportRecorder,
    project_root: Path,
    token: CancellationToken,
) -> None:
    recorder.section("APPLICATION AND SOURCE INTEGRITY")
    package_root = Path(__file__).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    failures: list[str] = []
    for path in source_files:
        token.checkpoint()
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append(f"{path.name}: {exc}")
    if failures:
        recorder.add("FAIL", "Python source compilation", f"{len(failures)} file(s) failed")
        recorder.detail("\n".join(failures))
    else:
        recorder.add("PASS", "Python source compilation", f"{len(source_files)} package modules compiled")

    resources_root = package_root / "resources"
    expected_resources = ("GUI_GUIDE.md", "METHOD_ALIGNMENT.md")
    missing_resources = [name for name in expected_resources if not (resources_root / name).is_file()]
    recorder.add(
        "FAIL" if missing_resources else "PASS",
        "Packaged help resources",
        f"missing: {', '.join(missing_resources)}" if missing_resources else "GUI guide and method alignment are present",
    )

    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        declared = match.group(1) if match else "not found"
        citation = project_root / "CITATION.cff"
        citation_text = citation.read_text(encoding="utf-8") if citation.is_file() else ""
        citation_match = re.search(r"^version:\s*([^\s]+)", citation_text, flags=re.MULTILINE)
        citation_version = citation_match.group(1).strip('"\'') if citation_match else "not found"
        recorder.add(
            "PASS" if declared == citation_version == __version__ else "FAIL",
            "Version consistency",
            f"package={__version__}; pyproject={declared}; citation={citation_version}",
        )
    else:
        recorder.add("SKIP", "Source-tree version check", "pyproject.toml is not installed with the wheel")


def _check_builtin_numerics(recorder: _ReportRecorder, token: CancellationToken) -> None:
    recorder.section("BUILT-IN NUMERICAL SELF-TEST")
    try:
        token.checkpoint()
        partition = FuzzyPartition.fit("x", np.linspace(-2.0, 3.0, 501))
        membership = partition.memberships(np.linspace(-20.0, 20.0, 2001))
        partition_error = float(np.max(np.abs(np.sum(membership, axis=1) - 1.0)))
        if partition_error > 1e-12:
            raise AssertionError(f"partition-of-unity error {partition_error:.3e}")

        penalties = nasa_penalty(np.array([-20.0, 20.0]))
        if not penalties[1] > penalties[0] > 0.0:
            raise AssertionError("NASA asymmetric penalty direction is incorrect")

        x = np.column_stack((np.linspace(-1.0, 1.0, 60), np.linspace(1.0, -1.0, 60)))
        y = np.linspace(1.0, 9.0, 60)
        vocabulary = FuzzyVocabulary.fit(("x1", "x2"), x)
        config = TrainingConfig(loss="mse", epochs=3, patience=3, ridge_lambda=0.1)
        model = R3vModel(vocabulary, unary_backbone(vocabulary.feature_names), (0.0, 10.0), config)
        model.fit(x, y, token=token)
        prediction, beta = model.predict(x[:8], return_beta=True)
        if not np.all(np.isfinite(prediction)):
            raise AssertionError("model predictions contain non-finite values")
        if not np.allclose(np.sum(beta, axis=1), 1.0, atol=1e-12):
            raise AssertionError("local responsibilities do not sum to one")
        explanation = model.explain_one(x[0], top_k=10)
        if not np.isclose(explanation["prediction"], explanation["reconstruction"], atol=1e-12):
            raise AssertionError("rule contributions do not reconstruct the prediction")

        with tempfile.TemporaryDirectory(prefix="cassandra-self-test-") as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = R3vModel.load(path)
            if not np.allclose(model.predict(x[:8]), restored.predict(x[:8]), atol=1e-12):
                raise AssertionError("model serialization changed predictions")
        recorder.add(
            "PASS",
            "Core numerical contracts",
            f"partition error={partition_error:.3e}; responsibilities normalized; model round trip exact",
        )
    except CancelledError:
        raise
    except Exception as exc:
        recorder.add("FAIL", "Core numerical contracts", str(exc))
        recorder.detail(traceback.format_exc())


def _run_source_tests(
    recorder: _ReportRecorder,
    project_root: Path,
    token: CancellationToken,
) -> None:
    recorder.section("AUTOMATED SOURCE TEST SUITE")
    tests_dir = project_root / "tests"
    if not tests_dir.is_dir():
        recorder.add("SKIP", "Source test suite", "tests are not installed with the wheel")
        return
    token.checkpoint()
    stream = io.StringIO()
    try:
        suite = unittest.defaultTestLoader.discover(str(tests_dir))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        token.checkpoint()
        recorder.detail(stream.getvalue())
        status = "PASS" if result.wasSuccessful() else "FAIL"
        recorder.add(
            status,
            "Source test suite",
            f"{result.testsRun} run; {len(result.failures)} failures; {len(result.errors)} errors; "
            f"{len(result.skipped)} skipped",
        )
    except CancelledError:
        raise
    except Exception as exc:
        recorder.detail(stream.getvalue())
        recorder.add("FAIL", "Source test suite", str(exc))
        recorder.detail(traceback.format_exc())


def _validate_prepared(dataset: PreparedDataset, subset: str, protocol: str) -> str:
    if dataset.subset != subset or dataset.protocol != protocol:
        raise ValueError(f"artifact identifies {dataset.subset}/{dataset.protocol}")
    if tuple(dataset.feature_names) != tuple(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} canonical features")
    matrices = (dataset.raw_train_x, dataset.train_x, dataset.train_y, dataset.test_endpoint_x, dataset.test_endpoint_y)
    if any(not np.all(np.isfinite(values)) for values in matrices):
        raise ValueError("prepared arrays contain non-finite values")
    if dataset.train_x.shape[1] != len(FEATURE_NAMES) or dataset.test_endpoint_x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("prepared feature dimensions are inconsistent")
    if dataset.fit_x.shape[0] != dataset.fit_y.shape[0]:
        raise ValueError("pseudo-endpoint feature and target counts differ")
    if np.any(dataset.scale <= 0.0):
        raise ValueError("standardization scale must be positive")
    if protocol == "cap125" and float(np.max(dataset.train_y)) > 125.0 + 1e-12:
        raise ValueError("cap125 targets exceed 125")
    return (
        f"TRAIN rows={dataset.train_x.shape[0]:,}; pseudo-endpoints={dataset.fit_x.shape[0]:,}; "
        f"TEST engines={dataset.test_endpoint_x.shape[0]:,}; features={dataset.train_x.shape[1]}"
    )


def _audit_raw_data(
    recorder: _ReportRecorder,
    repository: DatasetRepository,
    subsets: tuple[str, ...],
    token: CancellationToken,
    progress: Progress | None,
) -> set[str]:
    recorder.section("NASA C-MAPSS RAW DATA AUDIT")
    available: set[str] = set()
    raw_status = repository.status()["subsets"]
    for index, subset in enumerate(subsets, start=1):
        token.checkpoint()
        paths = [repository.raw_dir / f"{prefix}_{subset}.txt" for prefix in ("train", "test", "RUL")]
        if not raw_status[subset]:
            missing = [path.name for path in paths if not path.is_file() or path.stat().st_size == 0]
            recorder.add("SKIP", f"{subset} raw data", f"not available ({', '.join(missing)})")
            continue
        try:
            train, test, endpoint_rul = read_subset(repository.raw_dir, subset)
            train_engines = int(train["unit"].nunique())
            test_engines = int(test["unit"].nunique())
            if len(endpoint_rul) != test_engines:
                raise ValueError("RUL endpoint count does not match TEST engine count")
            train_values = train.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
            test_values = test.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
            if not np.all(np.isfinite(train_values)) or not np.all(np.isfinite(test_values)):
                raise ValueError("physical inputs contain non-finite values")
            if np.any(~np.isfinite(endpoint_rul)):
                raise ValueError("RUL endpoints contain non-finite values")
            hashes = {path.name: sha256_file(path) for path in paths}
            available.add(subset)
            recorder.add(
                "PASS",
                f"{subset} raw data",
                f"TRAIN rows={len(train):,}; TEST rows={len(test):,}; "
                f"TRAIN engines={train_engines}; TEST engines={test_engines}",
            )
            for name, digest in hashes.items():
                recorder.detail(f"SHA-256 {name}: {digest}")
        except Exception as exc:
            recorder.add("FAIL", f"{subset} raw data", str(exc))
            recorder.detail(traceback.format_exc())
        if progress:
            progress(0.28 + 0.12 * index / len(subsets), f"Audited raw data for {subset}")
    selected_files = tuple(
        name
        for name in expected_files()
        if any(name.endswith(f"_{subset}.txt") for subset in subsets)
    )
    found = sum(1 for name in selected_files if (repository.raw_dir / name).is_file())
    recorder.add(
        "PASS" if found == len(selected_files) else "WARN",
        "Selected raw-data inventory",
        f"{found}/{len(selected_files)} expected files present",
    )
    return available


def _prepare_selected_jobs(
    recorder: _ReportRecorder,
    repository: DatasetRepository,
    jobs: tuple[tuple[str, str], ...],
    raw_available: set[str],
    token: CancellationToken,
    progress: Progress | None,
) -> dict[tuple[str, str], PreparedDataset]:
    recorder.section("NON-DESTRUCTIVE PREPARATION CHECK")
    datasets: dict[tuple[str, str], PreparedDataset] = {}
    total = max(1, len(jobs))
    for index, (subset, protocol) in enumerate(jobs, start=1):
        token.checkpoint()
        if subset not in raw_available:
            recorder.add("SKIP", f"Prepare {subset}/{protocol}", "raw data are unavailable")
            continue
        try:
            dataset = prepare_subset(repository.raw_dir, subset, protocol, token=token)
            detail = _validate_prepared(dataset, subset, protocol)
            datasets[(subset, protocol)] = dataset
            recorder.add("PASS", f"Prepare {subset}/{protocol}", detail)
        except CancelledError:
            raise
        except Exception as exc:
            recorder.add("FAIL", f"Prepare {subset}/{protocol}", str(exc))
            recorder.detail(traceback.format_exc())
        if progress:
            progress(0.40 + 0.15 * index / total, f"Checked preparation for {subset}/{protocol}")
    return datasets


def _audit_artifacts(
    recorder: _ReportRecorder,
    store: ArtifactStore,
    datasets: dict[tuple[str, str], PreparedDataset],
    jobs: tuple[tuple[str, str], ...],
    token: CancellationToken,
    progress: Progress | None,
) -> None:
    recorder.section("SAVED ARTIFACT AUDIT")
    for index, (subset, protocol) in enumerate(jobs, start=1):
        token.checkpoint()
        job = (subset, protocol)
        paths = {
            "prepared": store.prepared_path(*job),
            "model": store.model_path(*job),
            "backbone": store.backbone_model_path(*job),
            "structure": store.structure_path(*job),
            "evaluation": store.evaluation_path(*job),
            "comparison": store.comparison_path(*job),
        }
        present = [name for name, path in paths.items() if path.is_file()]
        if not present:
            recorder.add("SKIP", f"Stored artifacts {subset}/{protocol}", "none found")
            continue

        if paths["prepared"].is_file():
            try:
                prepared = store.load_prepared(*job)
                detail = _validate_prepared(prepared, subset, protocol)
                datasets.setdefault(job, prepared)
                recorder.add("PASS", f"Prepared artifact {subset}/{protocol}", detail)
            except Exception as exc:
                recorder.add("FAIL", f"Prepared artifact {subset}/{protocol}", str(exc))

        if paths["model"].is_file():
            try:
                model = R3vModel.load(paths["model"])
                if len(model.rules) != len(model.theta) or len(model.rules) != len(model.consequents):
                    raise ValueError("rule, authority, and consequent counts differ")
                if not np.all(np.isfinite(model.theta)) or not np.all(np.isfinite(model.consequents)):
                    raise ValueError("model parameters contain non-finite values")
                if np.any(model.consequents < model.target_bounds[0]) or np.any(model.consequents > model.target_bounds[1]):
                    raise ValueError("model consequents violate their declared bounds")
                dataset = datasets.get(job)
                if dataset is not None:
                    columns = [dataset.feature_names.index(name) for name in model.vocabulary.feature_names]
                    _, beta = model.predict(dataset.test_endpoint_x[: min(5, len(dataset.test_endpoint_x)), columns], return_beta=True)
                    if not np.allclose(np.sum(beta, axis=1), 1.0, atol=1e-10):
                        raise ValueError("saved model responsibilities do not sum to one")
                recorder.add(
                    "PASS",
                    f"Model artifact {subset}/{protocol}",
                    f"rules={len(model.rules)}; interactions={sum(rule.order > 1 for rule in model.rules)}; "
                    f"loss={model.config.loss}; seed={model.config.seed}; configured epochs={model.config.epochs}; "
                    f"ridge={model.config.ridge_lambda:g}",
                )
                if model.fit_result is not None:
                    recorder.detail(
                        f"Training: completed epochs={model.fit_result.epochs_completed}; "
                        f"best epoch={model.fit_result.best_epoch}; "
                        f"best objective={model.fit_result.best_objective:.9g}; "
                        f"stopped early={model.fit_result.stopped_early}."
                    )
            except Exception as exc:
                recorder.add("FAIL", f"Model artifact {subset}/{protocol}", str(exc))

        if paths["backbone"].is_file():
            try:
                backbone = R3vModel.load(paths["backbone"])
                if len(backbone.rules) != len(backbone.theta) or len(backbone.rules) != len(backbone.consequents):
                    raise ValueError("rule, authority, and consequent counts differ")
                if not np.all(np.isfinite(backbone.theta)) or not np.all(np.isfinite(backbone.consequents)):
                    raise ValueError("model parameters contain non-finite values")
                recorder.add(
                    "PASS",
                    f"Backbone artifact {subset}/{protocol}",
                    f"rules={len(backbone.rules)}; "
                    f"interactions={sum(rule.order > 1 for rule in backbone.rules)}; "
                    f"seed={backbone.config.seed}",
                )
                if backbone.fit_result is not None:
                    recorder.detail(
                        f"Backbone training: completed epochs={backbone.fit_result.epochs_completed}; "
                        f"best epoch={backbone.fit_result.best_epoch}; "
                        f"best objective={backbone.fit_result.best_objective:.9g}."
                    )
            except Exception as exc:
                recorder.add("FAIL", f"Backbone artifact {subset}/{protocol}", str(exc))

        if paths["structure"].is_file():
            try:
                values = json.loads(paths["structure"].read_text(encoding="utf-8"))
                if values.get("subset") != subset or values.get("protocol") != protocol:
                    raise ValueError("structure report identifies a different experiment")
                result = values["result"]
                for key in ("initial_rule_count", "final_rule_count", "accepted_rules", "stop_reason"):
                    if key not in result:
                        raise ValueError(f"structure report is missing {key}")
                rounds = result.get("rounds", [])
                generated = sum(int(round_values.get("generated", 0)) for round_values in rounds)
                shortlisted = sum(int(round_values.get("screened", 0)) for round_values in rounds)
                validated = sum(len(round_values.get("validated", [])) for round_values in rounds)
                pruning = result.get("pruning_candidates", [])
                elapsed = float(result.get("elapsed_seconds", 0.0))
                recorder.add(
                    "PASS",
                    f"Structure artifact {subset}/{protocol}",
                    f"rules={result['initial_rule_count']}->{result['final_rule_count']}; "
                    f"accepted={len(result['accepted_rules'])}; generated={generated}; "
                    f"shortlisted={shortlisted}; validated={validated}; pruning proposals={len(pruning)}; "
                    f"elapsed={elapsed:.3f}s; stop={result['stop_reason']}",
                )
                for accepted_rule in result["accepted_rules"]:
                    recorder.detail(f"Accepted rule: {accepted_rule}")
                for round_values in rounds:
                    recorder.detail(
                        f"Round {round_values.get('round', '?')}: order={round_values.get('order', '?')}; "
                        f"generated={round_values.get('generated', 0)}; "
                        f"shortlisted={round_values.get('screened', 0)}; "
                        f"validated={len(round_values.get('validated', []))}; "
                        f"decision={round_values.get('decision', 'not recorded')}."
                    )
            except Exception as exc:
                recorder.add("FAIL", f"Structure artifact {subset}/{protocol}", str(exc))

        if paths["evaluation"].is_file():
            try:
                values = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
                if values.get("subset") != subset or values.get("protocol") != protocol:
                    raise ValueError("evaluation identifies a different experiment")
                metrics = ("nasa_score", "mae", "rmse", "r2", "bias")
                if any(not np.isfinite(float(values[key])) for key in metrics):
                    raise ValueError("evaluation metrics contain non-finite values")
                if len(values.get("predictions", [])) != len(values.get("truth", [])):
                    raise ValueError("evaluation prediction and truth counts differ")
                recorder.add(
                    "PASS",
                    f"Evaluation artifact {subset}/{protocol}",
                    f"n={int(values.get('n', len(values.get('truth', []))))}; "
                    f"rules={int(values.get('model_rules', 0))}; "
                    f"interactions={int(values.get('interaction_rules', 0))}; "
                    f"NASA={values['nasa_score']:.3f}; MAE={values['mae']:.3f}; "
                    f"RMSE={values['rmse']:.3f}; R2={values['r2']:.6f}; bias={values['bias']:.3f}; "
                    f"overestimation={float(values.get('overestimation_percent', 0.0)):.2f}%",
                )
                responsibility = values.get("responsibility", {})
                if responsibility:
                    recorder.detail(
                        f"Responsibility: median k90={float(responsibility['median_k90']):.3f}; "
                        f"mean k90={float(responsibility['mean_k90']):.3f}; "
                        f"median max beta={float(responsibility['median_max_beta']):.6f}; "
                        f"median normalized entropy="
                        f"{float(responsibility['median_normalized_entropy']):.6f}."
                    )
            except Exception as exc:
                recorder.add("FAIL", f"Evaluation artifact {subset}/{protocol}", str(exc))

        if paths["comparison"].is_file():
            try:
                values = json.loads(paths["comparison"].read_text(encoding="utf-8"))
                if values.get("subset") != subset or values.get("protocol") != protocol:
                    raise ValueError("comparison identifies a different experiment")
                if values.get("selection_scope") != "evaluation only; structural selection remains TRAIN-only":
                    raise ValueError("comparison does not declare the TRAIN-only selection boundary")
                backbone_values = values["backbone"]
                final_values = values["final_model"]
                change = values["change"]
                bootstrap = values["bootstrap"]
                interval = bootstrap["relative_nasa_improvement_percent"]
                finite_values = (
                    backbone_values["nasa_score"],
                    final_values["nasa_score"],
                    change["nasa_improvement_percent"],
                    interval["ci_95_low"],
                    interval["ci_95_high"],
                )
                if any(not np.isfinite(float(value)) for value in finite_values):
                    raise ValueError("comparison metrics contain non-finite values")
                counts = values["paired_engine_counts"]
                compared = (
                    int(counts["lower_nasa_penalty"])
                    + int(counts["equal_nasa_penalty"])
                    + int(counts["higher_nasa_penalty"])
                )
                if compared != int(counts["engines"]):
                    raise ValueError("paired engine counts do not add up")
                recorder.add(
                    "PASS",
                    f"Comparison artifact {subset}/{protocol}",
                    f"rules={int(backbone_values['rules'])}->{int(final_values['rules'])}; "
                    f"NASA={float(backbone_values['nasa_score']):.3f}->"
                    f"{float(final_values['nasa_score']):.3f}; "
                    f"gain={float(change['nasa_improvement_percent']):+.3f}%; "
                    f"95% interval=[{float(interval['ci_95_low']):+.3f}%, "
                    f"{float(interval['ci_95_high']):+.3f}%]; "
                    f"conclusion={change['bootstrap_conclusion']}",
                )
            except Exception as exc:
                recorder.add("FAIL", f"Comparison artifact {subset}/{protocol}", str(exc))

        if progress:
            progress(0.55 + 0.10 * index / len(jobs), f"Audited artifacts for {subset}/{protocol}")


def _compare_saved_models(
    recorder: _ReportRecorder,
    store: ArtifactStore,
    datasets: dict[tuple[str, str], PreparedDataset],
    jobs: tuple[tuple[str, str], ...],
    token: CancellationToken,
    progress: Progress | None,
) -> None:
    recorder.section("PAIRED BACKBONE–AUTOSTRUCTURE TEST COMPARISON")
    recorder.detail(
        "PASS in this section means that the paired comparison ran correctly; it does not by itself "
        "mean that AutoStructure improved predictive performance. Rule selection remains TRAIN-only."
    )
    for index, (subset, protocol) in enumerate(jobs, start=1):
        token.checkpoint()
        job = (subset, protocol)
        missing: list[str] = []
        if job not in datasets:
            missing.append("prepared data")
        backbone_path = store.backbone_model_path(*job)
        final_path = store.model_path(*job)
        if not backbone_path.is_file():
            missing.append("model_backbone.json")
        if not final_path.is_file():
            missing.append("model.json")
        if missing:
            recorder.add(
                "SKIP",
                f"Paired comparison {subset}/{protocol}",
                f"missing {', '.join(missing)}",
            )
            continue

        try:
            result = compare_prepared_models(
                R3vModel.load(backbone_path),
                R3vModel.load(final_path),
                datasets[job],
                token=token,
            )
            backbone_values = result["backbone"]
            final_values = result["final_model"]
            change = result["change"]
            interval = result["bootstrap"]["relative_nasa_improvement_percent"]
            counts = result["paired_engine_counts"]
            recorder.add(
                "PASS",
                f"Paired comparison {subset}/{protocol}",
                f"rules={int(backbone_values['rules'])}->{int(final_values['rules'])}; "
                f"NASA={float(backbone_values['nasa_score']):.3f}->"
                f"{float(final_values['nasa_score']):.3f}; "
                f"gain={float(change['nasa_improvement_percent']):+.3f}%; "
                f"95% interval=[{float(interval['ci_95_low']):+.3f}%, "
                f"{float(interval['ci_95_high']):+.3f}%]; "
                f"conclusion={change['bootstrap_conclusion']}",
            )
            recorder.detail(
                f"Paired engines: {int(counts['lower_nasa_penalty'])} lower, "
                f"{int(counts['equal_nasa_penalty'])} equal, "
                f"{int(counts['higher_nasa_penalty'])} higher NASA penalty; "
                f"favorable bootstrap resamples="
                f"{float(result['bootstrap']['favorable_resamples_percent']):.2f}%."
            )
            added_rules = result["structure_difference"]["added_rules"]
            removed_rules = result["structure_difference"]["removed_rules"]
            recorder.detail(
                f"Structure difference: added={len(added_rules)}; removed={len(removed_rules)}."
            )
            for rule in added_rules:
                recorder.detail(f"Added rule: {rule}")
            for rule in removed_rules:
                recorder.detail(f"Removed rule: {rule}")
        except CancelledError:
            raise
        except Exception as exc:
            recorder.add("FAIL", f"Paired comparison {subset}/{protocol}", str(exc))
            recorder.detail(traceback.format_exc())
        if progress:
            progress(0.65 + 0.08 * index / len(jobs), f"Compared models for {subset}/{protocol}")


def _run_pipeline_smoke_tests(
    recorder: _ReportRecorder,
    datasets: dict[tuple[str, str], PreparedDataset],
    jobs: tuple[tuple[str, str], ...],
    token: CancellationToken,
    progress: Progress | None,
) -> None:
    recorder.section("REAL-DATA PIPELINE SMOKE TEST")
    runnable = [job for job in jobs if job in datasets]
    if not runnable:
        recorder.add("SKIP", "Real-data pipeline", "no selected prepared or raw dataset is available")
        return
    total = len(runnable)
    for index, (subset, protocol) in enumerate(runnable, start=1):
        token.checkpoint()
        dataset = datasets[(subset, protocol)]
        try:
            config = TrainingConfig(
                loss="nasa",
                epochs=2,
                patience=2,
                learning_rate_consequents=0.025,
                learning_rate_authorities=0.025,
                ridge_lambda=0.9,
                seed=7,
            ).validate()

            def local_progress(value: float, message: str) -> None:
                if progress:
                    progress(0.73 + 0.19 * (index - 1 + value) / total, f"{subset}/{protocol} · {message}")

            model = train_prepared(dataset, config, token=token, progress=local_progress)
            result = evaluate_prepared(model, dataset)
            for key in ("nasa_score", "mae", "rmse", "r2", "bias"):
                if not np.isfinite(float(result[key])):
                    raise ValueError(f"non-finite {key}")
            recorder.add(
                "PASS",
                f"Pipeline {subset}/{protocol}",
                f"24 features; {len(model.rules)} rules; 2 diagnostic epochs; "
                f"NASA={result['nasa_score']:.3f}; MAE={result['mae']:.3f}; RMSE={result['rmse']:.3f}",
            )
        except CancelledError:
            raise
        except Exception as exc:
            recorder.add("FAIL", f"Pipeline {subset}/{protocol}", str(exc))
            recorder.detail(traceback.format_exc())

    token.checkpoint()
    first_job = runnable[0]
    dataset = datasets[first_job]
    try:
        names = tuple(dataset.feature_names[:2])
        compact_config = TrainingConfig(loss="nasa", epochs=2, patience=2, ridge_lambda=0.9, seed=7)
        compact_model = train_prepared(dataset, compact_config, feature_names=names, token=token)
        auto_config = AutoStructureConfig(
            min_support=0.0,
            min_novelty=0.0,
            min_relative_improvement=0.0,
            folds=2,
            seeds=(7,),
            consensus_required=1,
            max_order=2,
            screen_top_k=1,
            validation_epochs=1,
            budget=ComputeBudget(
                max_seconds=120.0,
                max_candidates=100,
                max_validations_per_seed=1,
                max_rounds=1,
            ),
        ).validate()

        def structure_progress(value: float, message: str) -> None:
            if progress:
                progress(0.92 + 0.06 * value, f"AutoStructure smoke test · {message}")

        _, structure = grow_structure(dataset, compact_model, auto_config, token=token, progress=structure_progress)
        if structure.initial_rule_count != 10 or not structure.rounds:
            raise ValueError("AutoStructure did not exercise its screening path")
        recorder.add(
            "PASS",
            "AutoStructure smoke path",
            f"{first_job[0]}/{first_job[1]}; generated={structure.rounds[0]['generated']}; "
            f"stop={structure.stop_reason}",
        )
    except CancelledError:
        raise
    except Exception as exc:
        recorder.add("FAIL", "AutoStructure smoke path", str(exc))
        recorder.detail(traceback.format_exc())


def run_full_test(
    *,
    project_root: Path,
    repository: DatasetRepository,
    store: ArtifactStore,
    selected_subsets: Iterable[str] = SUBSETS,
    selected_protocols: Iterable[str] = PROTOCOLS,
    token: CancellationToken = NEVER_CANCEL,
    progress: Progress | None = None,
    log: StatusLog | None = None,
    include_source_tests: bool = True,
    include_pipeline_smoke: bool = True,
) -> FullTestResult:
    """Run all available diagnostics without changing scientific artifacts.

    Missing NASA data or saved artifacts are reported as skipped because they are
    intentionally not bundled. Any material that is present is parsed and
    validated. The report is saved atomically even when checks fail or the user
    requests a cooperative stop.
    """

    subsets = normalize_subsets(selected_subsets)
    protocols = normalize_protocols(selected_protocols)
    jobs = tuple((subset, protocol) for subset in subsets for protocol in protocols)
    project_root = Path(project_root).resolve()
    recorder = _ReportRecorder(log)
    started_at = _utc_text()
    report_path = _unique_report_path(store)

    def finalize(status: str | None = None) -> FullTestResult:
        overall = status or recorder.default_overall_status()
        report_text = recorder.render(
            started_at=started_at,
            finished_at=_utc_text(),
            overall_status=overall,
            selected_subsets=subsets,
            selected_protocols=protocols,
            workspace_name=store.root.name,
        )
        _atomic_text(report_path, report_text)
        return FullTestResult(str(report_path), overall, dict(recorder.counts), report_text)

    try:
        if progress:
            progress(0.01, "Starting Full Test")
        _check_dependencies(recorder, token)
        if progress:
            progress(0.08, "Runtime and dependencies checked")
        _check_source_integrity(recorder, project_root, token)
        if progress:
            progress(0.14, "Application integrity checked")
        _check_builtin_numerics(recorder, token)
        if progress:
            progress(0.20, "Built-in numerical checks completed")
        if include_source_tests:
            _run_source_tests(recorder, project_root, token)
        else:
            recorder.section("AUTOMATED SOURCE TEST SUITE")
            recorder.add("SKIP", "Source test suite", "disabled by caller")
        if progress:
            progress(0.28, "Automated source tests completed")
        raw_available = _audit_raw_data(recorder, repository, subsets, token, progress)
        datasets = _prepare_selected_jobs(recorder, repository, jobs, raw_available, token, progress)
        _audit_artifacts(recorder, store, datasets, jobs, token, progress)
        _compare_saved_models(recorder, store, datasets, jobs, token, progress)
        if include_pipeline_smoke:
            _run_pipeline_smoke_tests(recorder, datasets, jobs, token, progress)
        else:
            recorder.section("REAL-DATA PIPELINE SMOKE TEST")
            recorder.add("SKIP", "Real-data pipeline", "disabled by caller")
        result = finalize()
        if log:
            log(f"Full Test report saved: {result.report_path}", "success" if not recorder.counts["FAIL"] else "warning")
        if progress:
            progress(1.0, f"Full Test completed: {result.overall_status}")
        return result
    except CancelledError:
        recorder.section("RUN CONTROL")
        recorder.add("WARN", "Full Test interrupted", "the user requested a cooperative stop")
        result = finalize("CANCELLED")
        if log:
            log(f"Partial Full Test report saved: {result.report_path}", "warning")
        raise CancelledError(f"Full Test stopped by the user. Partial report saved: {report_path.name}")
    except Exception as exc:
        recorder.section("UNEXPECTED DIAGNOSTIC ERROR")
        recorder.add("FAIL", "Full Test runner", str(exc))
        recorder.detail(traceback.format_exc())
        result = finalize("FAIL")
        if log:
            log(f"Failed Full Test report saved: {result.report_path}", "error")
        if progress:
            progress(1.0, "Full Test completed with an internal error")
        return result
