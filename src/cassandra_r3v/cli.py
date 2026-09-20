"""Headless entry point for reproducible Cassandra operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._version import __version__
from .artifacts import ArtifactStore, atomic_json, utc_now
from .autostructure import ComputeBudgetExceeded, grow_structure
from .config import (
    PROTOCOLS,
    SUBSETS,
    AutoStructureConfig,
    ComputeBudget,
    TrainingConfig,
    normalize_protocols,
    normalize_subsets,
)
from .data import DatasetRepository, prepare_subset
from .experiment import compare_prepared_models, evaluate_prepared, save_comparison, train_prepared
from .model import R3vModel


def _emit(values: Any) -> None:
    print(json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False))


def _jobs(args) -> list[tuple[str, str]]:
    subsets = normalize_subsets(args.subsets)
    protocols = normalize_protocols(args.protocols)
    return [(subset, protocol) for subset in subsets for protocol in protocols]


def _progress(value: float, message: str) -> None:
    print(f"[{value * 100:6.2f}%] {message}")


def _diagnostic_log(message: str, _tag: str = "") -> None:
    print(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cassandra-r3v", description="Cassandra R3v for NASA C-MAPSS")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--workspace", type=Path, default=Path("cassandra_workspace"), help="artifact workspace")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="show raw data and artifact status")
    import_parser = commands.add_parser("import-data", help="import NASA's outer or inner ZIP archive")
    import_parser.add_argument("archive", type=Path)

    for name in ("prepare", "train", "autostructure", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
        command.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
        if name == "train":
            command.add_argument("--epochs", type=int, default=250)
            command.add_argument("--learning-rate", type=float, default=0.025)
            command.add_argument("--ridge", type=float, default=0.9)
            command.add_argument("--patience", type=int, default=50)
        elif name == "autostructure":
            command.add_argument("--min-support", type=float, default=0.01)
            command.add_argument("--activation-support-cutoff", type=float, default=0.01)
            command.add_argument("--min-novelty", type=float, default=0.005)
            command.add_argument("--min-relative-improvement", type=float, default=0.01)
            command.add_argument("--folds", type=int, default=3)
            command.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 99])
            command.add_argument("--consensus-required", type=int, default=2)
            command.add_argument("--max-order", type=int, default=3)
            command.add_argument("--screen-top-k", type=int, default=24)
            command.add_argument("--pruning-threshold", type=float, default=0.002)
            command.add_argument("--parent-threshold", type=float, default=0.0002)
            command.add_argument("--validation-epochs", type=int, default=100)
            command.add_argument("--max-minutes", type=float, default=60.0)
            command.add_argument("--max-candidates", type=int, default=10_000)
            command.add_argument("--validations-per-seed", type=int, default=4)
            command.add_argument("--max-rounds", type=int, default=3)
            command.add_argument(
                "--resume-current",
                action="store_true",
                help="continue from model.json instead of restarting from model_backbone.json",
            )

    gui = commands.add_parser("gui", help="launch the desktop interface")
    gui.set_defaults(gui=True)

    full_test = commands.add_parser("full-test", help="run comprehensive diagnostics and save a TXT report")
    full_test.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    full_test.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    full_test.add_argument("--skip-source-tests", action="store_true")
    full_test.add_argument("--skip-pipeline-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    store = ArtifactStore(workspace)
    repository = DatasetRepository(workspace)

    if args.command == "gui":
        from .gui.app import launch

        launch(workspace)
        return 0
    if args.command == "status":
        _emit({"dataset": repository.status(), "artifacts": store.summary()})
        return 0
    if args.command == "full-test":
        from .diagnostics import run_full_test

        result = run_full_test(
            project_root=Path(__file__).resolve().parents[2],
            repository=repository,
            store=store,
            selected_subsets=normalize_subsets(args.subsets),
            selected_protocols=normalize_protocols(args.protocols),
            progress=_progress,
            log=_diagnostic_log,
            include_source_tests=not args.skip_source_tests,
            include_pipeline_smoke=not args.skip_pipeline_smoke,
        )
        _emit(
            {
                "full_test_report": result.report_path,
                "overall_status": result.overall_status,
                "counts": result.counts,
            }
        )
        return 1 if result.overall_status == "FAIL" else 0
    if args.command == "import-data":
        _emit(repository.import_archive(args.archive, progress=_progress))
        return 0
    if args.command == "prepare":
        results = []
        for subset, protocol in _jobs(args):
            dataset = prepare_subset(repository.raw_dir, subset, protocol)
            store.invalidate_after("prepared", subset, protocol)
            results.append(str(store.save_prepared(dataset)))
        _emit({"prepared": results})
        return 0
    if args.command == "train":
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate_consequents=args.learning_rate,
            learning_rate_authorities=args.learning_rate,
            ridge_lambda=args.ridge,
            patience=args.patience,
        ).validate()
        results = []
        for subset, protocol in _jobs(args):
            dataset = store.load_prepared(subset, protocol)
            model = train_prepared(dataset, config, progress=_progress)
            store.invalidate_after("model", subset, protocol)
            model.save(store.model_path(subset, protocol))
            model.save(store.backbone_model_path(subset, protocol))
            results.append(str(store.model_path(subset, protocol)))
        _emit({"models": results})
        return 0
    if args.command == "autostructure":
        config = AutoStructureConfig(
            min_support=args.min_support,
            activation_support_cutoff=args.activation_support_cutoff,
            min_novelty=args.min_novelty,
            min_relative_improvement=args.min_relative_improvement,
            folds=args.folds,
            seeds=tuple(args.seeds),
            consensus_required=args.consensus_required,
            max_order=args.max_order,
            screen_top_k=args.screen_top_k,
            prune_relative_authority=args.pruning_threshold,
            parent_relative_authority=args.parent_threshold,
            validation_epochs=args.validation_epochs,
            budget=ComputeBudget(
                max_seconds=args.max_minutes * 60.0,
                max_candidates=args.max_candidates,
                max_validations_per_seed=args.validations_per_seed,
                max_rounds=args.max_rounds,
            ),
        ).validate()
        results = []
        for subset, protocol in _jobs(args):
            dataset = store.load_prepared(subset, protocol)
            source = (
                store.model_path(subset, protocol)
                if args.resume_current
                else store.backbone_model_path(subset, protocol)
            )
            model = R3vModel.load(source)
            try:
                child, result = grow_structure(dataset, model, config, progress=_progress)
            except ComputeBudgetExceeded as exc:
                _emit(
                    {
                        "error": str(exc),
                        "subset": subset,
                        "protocol": protocol,
                        "recommendations": list(exc.recommendations),
                    }
                )
                return 2
            store.invalidate_after("structure", subset, protocol)
            child.save(store.model_path(subset, protocol))
            report = {
                "schema": 1,
                "subset": subset,
                "protocol": protocol,
                "created_at": utc_now(),
                "source_model": "current" if args.resume_current else "backbone",
                "config": config.to_dict(),
                "result": result.to_dict(),
            }
            report_path = atomic_json(store.structure_path(subset, protocol), report)
            results.append(
                {
                    "subset": subset,
                    "protocol": protocol,
                    "model": str(store.model_path(subset, protocol)),
                    "report": str(report_path),
                    "initial_rules": result.initial_rule_count,
                    "final_rules": result.final_rule_count,
                    "accepted_rules": result.accepted_rules,
                    "stop_reason": result.stop_reason,
                }
            )
        _emit({"structures": results})
        return 0
    if args.command == "evaluate":
        results = []
        for subset, protocol in _jobs(args):
            dataset = store.load_prepared(subset, protocol)
            final_model = R3vModel.load(store.model_path(subset, protocol))
            backbone = R3vModel.load(store.backbone_model_path(subset, protocol))
            evaluation = evaluate_prepared(final_model, dataset)
            comparison = compare_prepared_models(backbone, final_model, dataset)
            evaluation_path = atomic_json(store.evaluation_path(subset, protocol), evaluation)
            comparison_path = save_comparison(store, comparison)
            interval = comparison["bootstrap"]["relative_nasa_improvement_percent"]
            results.append(
                {
                    "subset": subset,
                    "protocol": protocol,
                    "evaluation": str(evaluation_path),
                    "comparison": comparison_path,
                    "rules": {
                        "backbone": comparison["backbone"]["rules"],
                        "final": comparison["final_model"]["rules"],
                    },
                    "nasa_score": {
                        "backbone": comparison["backbone"]["nasa_score"],
                        "final": comparison["final_model"]["nasa_score"],
                    },
                    "gain_percent": comparison["change"]["nasa_improvement_percent"],
                    "gain_95_percent_interval": [interval["ci_95_low"], interval["ci_95_high"]],
                    "conclusion": comparison["change"]["bootstrap_conclusion"],
                }
            )
        _emit({"evaluations": results})
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
