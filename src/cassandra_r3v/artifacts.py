"""Stable paths and atomic metadata for Cassandra runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import PROTOCOLS, SUBSETS
from .data import PreparedDataset


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, values: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(values, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


class ArtifactStore:
    def __init__(self, workspace: Path):
        self.root = Path(workspace).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def prepared_path(self, subset: str, protocol: str) -> Path:
        return self.root / "prepared" / subset / f"{protocol}.npz"

    def model_path(self, subset: str, protocol: str) -> Path:
        return self.root / "models" / subset / protocol / "model.json"

    def backbone_model_path(self, subset: str, protocol: str) -> Path:
        return self.root / "models" / subset / protocol / "model_backbone.json"

    def structure_path(self, subset: str, protocol: str) -> Path:
        return self.root / "structures" / subset / protocol / "autostructure.json"

    def evaluation_path(self, subset: str, protocol: str) -> Path:
        return self.root / "evaluations" / subset / f"{protocol}.json"

    def comparison_path(self, subset: str, protocol: str) -> Path:
        return self.root / "evaluations" / subset / f"{protocol}_backbone_vs_autostructure.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def full_test_report_path(self, stamp: str) -> Path:
        return self.reports_dir / f"cassandra_full_test_{stamp}.txt"

    def latest_full_test_report(self) -> Path | None:
        reports = sorted(self.reports_dir.glob("cassandra_full_test_*.txt"))
        return reports[-1] if reports else None

    def save_prepared(self, dataset: PreparedDataset) -> Path:
        return dataset.save(self.prepared_path(dataset.subset, dataset.protocol))

    def load_prepared(self, subset: str, protocol: str) -> PreparedDataset:
        return PreparedDataset.load(self.prepared_path(subset, protocol))

    def stage_status(self, subset: str, protocol: str) -> dict[str, bool]:
        return {
            "prepared": self.prepared_path(subset, protocol).is_file(),
            "model": self.model_path(subset, protocol).is_file(),
            "backbone": self.backbone_model_path(subset, protocol).is_file(),
            "structure": self.structure_path(subset, protocol).is_file(),
            "evaluation": self.evaluation_path(subset, protocol).is_file(),
            "comparison": self.comparison_path(subset, protocol).is_file(),
        }

    def invalidate_after(self, stage: str, subset: str, protocol: str) -> list[str]:
        """Archive downstream artifacts instead of silently deleting them."""

        order = ("prepared", "model", "structure", "evaluation")
        if stage not in order:
            raise ValueError(f"unknown stage: {stage}")
        paths = {
            "prepared": (self.prepared_path(subset, protocol),),
            "model": (
                self.model_path(subset, protocol),
                self.backbone_model_path(subset, protocol),
            ),
            "structure": (self.structure_path(subset, protocol),),
            "evaluation": (
                self.evaluation_path(subset, protocol),
                self.comparison_path(subset, protocol),
            ),
        }
        archived: list[str] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_root = self.root / "archive" / stamp / subset / protocol
        for downstream in order[order.index(stage) + 1 :]:
            for path in paths[downstream]:
                if not path.is_file():
                    continue
                archive_root.mkdir(parents=True, exist_ok=True)
                destination = archive_root / path.name
                counter = 1
                while destination.exists():
                    destination = archive_root / f"{path.stem}_{counter}{path.suffix}"
                    counter += 1
                shutil.move(str(path), str(destination))
                archived.append(str(destination))
        return archived

    def manifest(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": 1,
            "application": "Cassandra R3v",
            "version": __version__,
            "operation": operation,
            "created_at": utc_now(),
            "parameters": parameters,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "process_id": os.getpid(),
            },
        }

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for subset in SUBSETS:
            result[subset] = {protocol: self.stage_status(subset, protocol) for protocol in PROTOCOLS}
        return result
