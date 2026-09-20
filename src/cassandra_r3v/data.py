"""Leakage-safe NASA C-MAPSS loading and endpoint preparation."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ._version import __version__
from .cancellation import CancellationToken, NEVER_CANCEL
from .config import PSEUDO_ENDPOINT_FRACTIONS, PROTOCOLS, SUBSETS


NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

RAW_COLUMNS = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{index}" for index in range(1, 22)
]

CANONICAL_MAP = {
    "op_setting_1": "OS1_Altitude",
    "op_setting_2": "OS2_Mach",
    "op_setting_3": "OS3_TRA",
    "sensor_1": "S01_T2_FanInletTemperature",
    "sensor_2": "S02_T24_LPCOutletTemperature",
    "sensor_3": "S03_T30_HPCOutletTemperature",
    "sensor_4": "S04_T50_LPTOutletTemperature",
    "sensor_5": "S05_P2_FanInletPressure",
    "sensor_6": "S06_P15_BypassDuctPressure",
    "sensor_7": "S07_P30_HPCOutletPressure",
    "sensor_8": "S08_Nf_FanSpeed",
    "sensor_9": "S09_Nc_CoreSpeed",
    "sensor_10": "S10_epr_EnginePressureRatio",
    "sensor_11": "S11_Ps30_HPCOutletStaticPressure",
    "sensor_12": "S12_phi_FuelFlowPs30Ratio",
    "sensor_13": "S13_NRf_CorrectedFanSpeed",
    "sensor_14": "S14_NRc_CorrectedCoreSpeed",
    "sensor_15": "S15_BPR_BypassRatio",
    "sensor_16": "S16_farB_BurnerFuelAirRatio",
    "sensor_17": "S17_htBleed_BleedEnthalpy",
    "sensor_18": "S18_Nf_dmd_DemandedFanSpeed",
    "sensor_19": "S19_PCNfR_dmd_DemandedCorrectedFanSpeed",
    "sensor_20": "S20_W31_HPTCoolantBleed",
    "sensor_21": "S21_W32_LPTCoolantBleed",
}
FEATURE_NAMES = tuple(CANONICAL_MAP[column] for column in RAW_COLUMNS[2:])


def expected_files() -> tuple[str, ...]:
    return tuple(
        filename
        for subset in SUBSETS
        for filename in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return candidate.name


class DatasetRepository:
    """Own the local raw-data directory without bundling NASA data in the code."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.raw_dir = self.workspace / "raw" / "extracted"
        self.download_dir = self.workspace / "raw" / "downloads"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        available = {}
        for subset in SUBSETS:
            files = [
                self.raw_dir / f"train_{subset}.txt",
                self.raw_dir / f"test_{subset}.txt",
                self.raw_dir / f"RUL_{subset}.txt",
            ]
            available[subset] = all(path.is_file() and path.stat().st_size > 0 for path in files)
        return {
            "raw_dir": str(self.raw_dir),
            "subsets": available,
            "complete": all(available.values()),
            "found_files": sum(3 for value in available.values() if value),
            "expected_files": 12,
        }

    def import_archive(
        self,
        archive: Path,
        token: CancellationToken = NEVER_CANCEL,
        progress: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Import either NASA's outer archive or its inner ``CMAPSSData.zip``."""

        source = Path(archive).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        token.checkpoint()
        wanted = set(expected_files()) | {"readme.txt", "Damage Propagation Modeling.pdf"}

        def extract_zip(handle: zipfile.ZipFile) -> int:
            members = [item for item in handle.infolist() if not item.is_dir()]
            extracted = 0
            for index, member in enumerate(members, start=1):
                token.checkpoint()
                basename = _safe_member_name(member.filename)
                if basename not in wanted:
                    continue
                destination = self.raw_dir / basename
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with handle.open(member, "r") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                temporary.replace(destination)
                extracted += 1
                if progress:
                    progress(index / max(1, len(members)), f"Imported {basename}")
            return extracted

        extracted_count = 0
        with zipfile.ZipFile(source) as outer:
            inner_members = [item for item in outer.infolist() if Path(item.filename).name == "CMAPSSData.zip"]
            if inner_members:
                token.checkpoint()
                with outer.open(inner_members[0]) as inner_stream:
                    inner_bytes = inner_stream.read()
                token.checkpoint()
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    extracted_count = extract_zip(inner)
            else:
                extracted_count = extract_zip(outer)

        result = self.status()
        result.update({"source": str(source), "archive_sha256": sha256_file(source), "extracted": extracted_count})
        if not any(result["subsets"].values()):
            raise ValueError("the archive does not contain recognizable C-MAPSS data files")
        return result

    def download_official(
        self,
        token: CancellationToken = NEVER_CANCEL,
        progress: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        destination = self.download_dir / "nasa_cmapss.zip"
        partial = destination.with_suffix(".zip.partial")
        request = urllib.request.Request(NASA_URL, headers={"User-Agent": f"Cassandra-R3v/{__version__}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
                total = int(response.headers.get("Content-Length", "0") or 0)
                copied = 0
                while True:
                    token.checkpoint()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    copied += len(chunk)
                    if progress:
                        ratio = copied / total if total else 0.0
                        progress(ratio, f"Downloaded {copied / 1024 / 1024:.1f} MiB")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return self.import_archive(destination, token=token, progress=progress)


def read_subset(raw_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset: {subset}")
    root = Path(raw_dir)
    paths = (
        root / f"train_{subset}.txt",
        root / f"test_{subset}.txt",
        root / f"RUL_{subset}.txt",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    train = pd.read_csv(paths[0], sep=r"\s+", header=None, names=RAW_COLUMNS)
    test = pd.read_csv(paths[1], sep=r"\s+", header=None, names=RAW_COLUMNS)
    endpoint_rul = pd.read_csv(paths[2], sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    train = train.rename(columns=CANONICAL_MAP)
    test = test.rename(columns=CANONICAL_MAP)
    return train, test, endpoint_rul


def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    maximum = result.groupby("unit", sort=False)["cycle"].transform("max")
    result["RUL"] = maximum - result["cycle"]
    return result


def add_test_rul(frame: pd.DataFrame, endpoint_rul: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    units = np.sort(result["unit"].unique())
    truth = np.asarray(endpoint_rul, dtype=float)
    if len(units) != len(truth):
        raise ValueError("the RUL file does not match the number of test engines")
    map_truth = dict(zip(units, truth))
    final_cycle = result.groupby("unit", sort=False)["cycle"].transform("max")
    result["RUL"] = result["unit"].map(map_truth).astype(float) + final_cycle - result["cycle"]
    return result


def apply_protocol(target: np.ndarray, protocol: str) -> np.ndarray:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    values = np.asarray(target, dtype=float)
    return np.minimum(values, 125.0) if protocol == "cap125" else values.copy()


def pseudo_endpoint_indices(
    units: np.ndarray,
    cycles: np.ndarray,
    fractions: Iterable[float] = PSEUDO_ENDPOINT_FRACTIONS,
) -> np.ndarray:
    unit_values = np.asarray(units)
    cycle_values = np.asarray(cycles)
    chosen: list[int] = []
    for unit in np.unique(unit_values):
        indices = np.flatnonzero(unit_values == unit)
        indices = indices[np.argsort(cycle_values[indices], kind="stable")]
        if len(indices) == 1:
            chosen.append(int(indices[0]))
            continue
        for fraction in fractions:
            position = int(round(float(fraction) * (len(indices) - 1)))
            chosen.append(int(indices[min(max(position, 0), len(indices) - 1)]))
    return np.asarray(chosen, dtype=int)


@dataclass
class PreparedDataset:
    subset: str
    protocol: str
    feature_names: tuple[str, ...]
    raw_train_x: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    train_units: np.ndarray
    train_cycles: np.ndarray
    fit_indices: np.ndarray
    raw_test_endpoint_x: np.ndarray
    test_endpoint_x: np.ndarray
    test_endpoint_y: np.ndarray
    test_units: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]

    @property
    def fit_x(self) -> np.ndarray:
        return self.train_x[self.fit_indices]

    @property
    def fit_y(self) -> np.ndarray:
        return self.train_y[self.fit_indices]

    @property
    def fit_units(self) -> np.ndarray:
        return self.train_units[self.fit_indices]

    @property
    def target_upper_bound(self) -> float:
        return 125.0 if self.protocol == "cap125" else float(np.max(self.train_y))

    def save(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                subset=np.array(self.subset),
                protocol=np.array(self.protocol),
                feature_names=np.asarray(self.feature_names),
                raw_train_x=self.raw_train_x,
                train_x=self.train_x,
                train_y=self.train_y,
                train_units=self.train_units,
                train_cycles=self.train_cycles,
                fit_indices=self.fit_indices,
                raw_test_endpoint_x=self.raw_test_endpoint_x,
                test_endpoint_x=self.test_endpoint_x,
                test_endpoint_y=self.test_endpoint_y,
                test_units=self.test_units,
                mean=self.mean,
                scale=self.scale,
                metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
            )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: Path) -> "PreparedDataset":
        with np.load(Path(path), allow_pickle=False) as values:
            return cls(
                subset=str(values["subset"].item()),
                protocol=str(values["protocol"].item()),
                feature_names=tuple(str(v) for v in values["feature_names"].tolist()),
                raw_train_x=values["raw_train_x"],
                train_x=values["train_x"],
                train_y=values["train_y"],
                train_units=values["train_units"],
                train_cycles=values["train_cycles"],
                fit_indices=values["fit_indices"],
                raw_test_endpoint_x=values["raw_test_endpoint_x"],
                test_endpoint_x=values["test_endpoint_x"],
                test_endpoint_y=values["test_endpoint_y"],
                test_units=values["test_units"],
                mean=values["mean"],
                scale=values["scale"],
                metadata=json.loads(str(values["metadata"].item())),
            )


def prepare_subset(
    raw_dir: Path,
    subset: str,
    protocol: str,
    fractions: Iterable[float] = PSEUDO_ENDPOINT_FRACTIONS,
    token: CancellationToken = NEVER_CANCEL,
) -> PreparedDataset:
    token.checkpoint()
    train, test, endpoint_truth = read_subset(raw_dir, subset)
    token.checkpoint()
    train = add_train_rul(train)
    test = add_test_rul(test, endpoint_truth)
    train_y = apply_protocol(train["RUL"].to_numpy(dtype=float), protocol)
    test["RUL"] = apply_protocol(test["RUL"].to_numpy(dtype=float), protocol)
    raw_train_x = train.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    mean = np.mean(raw_train_x, axis=0)
    scale = np.std(raw_train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_x = (raw_train_x - mean) / scale

    endpoint_rows = test.sort_values(["unit", "cycle"]).groupby("unit", sort=True).tail(1)
    raw_test_endpoint_x = endpoint_rows.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    test_endpoint_x = (raw_test_endpoint_x - mean) / scale
    fit_indices = pseudo_endpoint_indices(
        train["unit"].to_numpy(), train["cycle"].to_numpy(), fractions
    )
    token.checkpoint()

    raw_paths = [Path(raw_dir) / f"{prefix}_{subset}.txt" for prefix in ("train", "test", "RUL")]
    metadata = {
        "schema": 1,
        "subset": subset,
        "protocol": protocol,
        "n_train_rows": int(len(train)),
        "n_train_engines": int(train["unit"].nunique()),
        "n_fit_pseudo_endpoints": int(len(fit_indices)),
        "n_test_engines": int(len(endpoint_rows)),
        "pseudo_endpoint_fractions": [float(value) for value in fractions],
        "scaler_fit_scope": "TRAIN complete trajectories only",
        "target_upper_bound": 125.0 if protocol == "cap125" else float(np.max(train_y)),
        "raw_sha256": {path.name: sha256_file(path) for path in raw_paths},
    }
    return PreparedDataset(
        subset=subset,
        protocol=protocol,
        feature_names=FEATURE_NAMES,
        raw_train_x=raw_train_x,
        train_x=train_x,
        train_y=train_y,
        train_units=train["unit"].to_numpy(dtype=int),
        train_cycles=train["cycle"].to_numpy(dtype=int),
        fit_indices=fit_indices,
        raw_test_endpoint_x=raw_test_endpoint_x,
        test_endpoint_x=test_endpoint_x,
        test_endpoint_y=endpoint_rows["RUL"].to_numpy(dtype=float),
        test_units=endpoint_rows["unit"].to_numpy(dtype=int),
        mean=mean,
        scale=scale,
        metadata=metadata,
    )
