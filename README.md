# Cassandra R3v 1.0.0

**Public release date:** 2026-09-20

**Repository:** https://github.com/r-alia-sicily/cassandra-r3v

**DOI for version 1.0.0:** https://doi.org/10.5281/zenodo.22852441

**DOI for all versions:** https://doi.org/10.5281/zenodo.22852440

Cassandra is a transparent implementation of **Rul3volution (R3v)** for Remaining Useful Life (RUL) estimation on NASA C-MAPSS. It combines a normalized zero-order Takagi–Sugeno engine, readable fuzzy rules, evidence-controlled structural growth, faithful local decompositions, and a desktop workflow for the four C-MAPSS subsets.

**Cassandra R3v 1.0.0 is a public reference implementation of the R3v method.** Results attributed specifically to this release should be generated with the tagged `v1.0.0` source, the declared experiment profile, and archived machine-readable outputs.

The software is licensed under GNU GPL v3 or later. NASA data are not redistributed.

## Main capabilities

- five stable linguistic levels for each of the 24 physical C-MAPSS channels;
- distinct antecedent activation `mu`, consequent `c`, positive relative authority `a`, and normalized local responsibility `beta`;
- centered log-authorities without a `sum(a)=1` constraint;
- bounded Ridge initialization followed by alternating optimization;
- NASA asymmetric loss as the canonical objective;
- TRAIN-only standardization, fuzzy partitions, and pseudo-endpoints;
- grouped validation by complete engine;
- evidence-controlled `AND` rule growth using support, novelty, matched parent–child budgets, and multi-seed consensus;
- independent `uncapped` and `cap125` artifacts throughout the workflow;
- recoverable invalidation of downstream artifacts;
- cooperative cancellation that keeps Cassandra open and preserves valid work;
- a six-stage English desktop interface, including a comprehensive Full Test;
- a headless CLI covering data import, preparation, training, AutoStructure, evaluation, and release diagnostics;
- an automatic paired backbone–AutoStructure TEST comparison with a deterministic 20,000-resample engine bootstrap;
- deterministic JSON/NPZ/TXT artifacts suitable for audit and article replication.

Rules below the configured pruning threshold are reported as proposals only. Cassandra 1.0.0 does not delete them automatically and does not implement disjunctive `OR` consolidation.

## Requirements

- Python 3.10 or newer;
- NumPy 1.24 or newer;
- pandas 2.0 or newer;
- Tkinter for the desktop interface.

On Debian, Ubuntu, or Linux Mint:

```bash
sudo apt install python3 python3-venv python3-tk
```

No GPU is required. x86-64 and ARM64 are supported when compatible Python dependencies are available.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
cassandra-r3v --version
```

On Windows, activate the environment with `.venv\Scripts\activate` and replace `python3` with `py -3` where appropriate.

## Start the desktop application

From an unpacked source release:

```bash
./run_cassandra.sh
```

or:

```bash
python3 run_cassandra.py
```

On Windows, double-click `run_cassandra.bat`. An installed copy can be started with:

```bash
cassandra-r3v gui
```

The source and dedicated GUI launchers print Cassandra's canonical ASCII banner before opening the window. Headless commands remain banner-free.

## Guided workflow

1. **Data** — import NASA's outer Turbofan ZIP or the inner `CMAPSSData.zip`.
2. **Preparation** — create a leakage-safe artifact for each selected subset/protocol pair.
3. **Model** — train the 120-rule Full24 unary backbone.
4. **AutoStructure** — optionally grow validated conjunctions.
5. **Evaluation** — compare the frozen backbone and final model on identical official TEST endpoints, then inspect rule decompositions.
6. **Full Test** — audit the environment, data, artifacts, paired comparisons, source tests, and bounded pipeline paths.

The selections for FD001–FD004 and `uncapped`/`cap125` are shared across tabs. The workflow strip exposes missing prerequisites before an operation starts.

## Headless workflow

The defaults below are the canonical values recorded in [`examples/article_profile.json`](examples/article_profile.json):

```bash
cassandra-r3v --workspace cassandra_workspace import-data /path/to/nasa_cmapss.zip
cassandra-r3v --workspace cassandra_workspace prepare
cassandra-r3v --workspace cassandra_workspace train
cassandra-r3v --workspace cassandra_workspace autostructure
cassandra-r3v --workspace cassandra_workspace evaluate
cassandra-r3v --workspace cassandra_workspace full-test
```

AutoStructure restarts from `model_backbone.json` by default. This makes a repeated canonical run independent of any previously derived current model. `--resume-current` is available only when an explicitly continued structural search is intended.

Use `--subsets` and `--protocols` to run a smaller explicit scope. Every experiment remains separate; Cassandra never averages the two target protocols.

## Full Test

Full Test is a non-destructive software acceptance check. It covers:

- Python, NumPy, pandas, Tkinter, source syntax, packaged resources, and version consistency;
- numerical invariants and the complete automated source suite;
- every available FD001–FD004 raw dataset, including SHA-256 file hashes;
- each saved preparation, retained Full24 backbone, current model, AutoStructure report, and evaluation;
- each paired backbone–AutoStructure TEST comparison, including its deterministic 20,000-resample engine bootstrap;
- short training/evaluation smoke paths for all selected experiments;
- a bounded AutoStructure screening and matched-validation path.

Reports are written to `cassandra_workspace/reports/`. Missing NASA data are marked `SKIP` because data are intentionally excluded from the repository. Smoke-test metrics demonstrate executable software paths; they are not article results.

## Artifact layout

```text
cassandra_workspace/
├── raw/extracted/
├── prepared/FDxxx/{uncapped,cap125}.npz
├── models/FDxxx/{uncapped,cap125}/
│   ├── model_backbone.json
│   └── model.json
├── structures/FDxxx/{uncapped,cap125}/autostructure.json
├── evaluations/FDxxx/
│   ├── {uncapped,cap125}.json
│   └── {uncapped,cap125}_backbone_vs_autostructure.json
├── reports/cassandra_full_test_<timestamp>.txt
└── archive/<timestamp>/
```

Model files include the fuzzy vocabulary, semantic rules, log-authorities, bounded consequents, training configuration, and optimization history. Evaluation files contain endpoint truth, predictions, metrics, responsibility diagnostics, paired engine outcomes, and bootstrap intervals. TEST data are used only after structural selection has finished.

## Scientific reproducibility

Before using numbers in the article:

1. check out tag `v1.0.0` and record the commit SHA;
2. use the profile in `examples/article_profile.json` without undocumented changes;
3. run each of the eight subset/protocol experiments independently;
4. preserve the generated preparation, model, structure, evaluation, paired-comparison, and Full Test artifacts;
5. report experimental metrics from the evaluation artifacts, not from the diagnostic smoke section;
6. record the data hashes, Python environment, machine, and command line.

The complete procedure and evidence checklist are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Method-to-code mappings and declared limitations are in [`docs/METHOD_ALIGNMENT.md`](docs/METHOD_ALIGNMENT.md).

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src
```

The release acceptance run used all four C-MAPSS subsets and both protocols. Its bundled report records 88 PASS, 0 WARN, 0 SKIP, and 0 FAIL; the source suite contains 55 tests.

## Documentation

- [`docs/Cassandra_User_Manual_v1.0.0.pdf`](docs/Cassandra_User_Manual_v1.0.0.pdf) — concise illustrated user manual;
- [`docs/Cassandra_User_Manual_v1.0.0.tex`](docs/Cassandra_User_Manual_v1.0.0.tex) — LaTeX source of the user manual;
- [`docs/GUI_GUIDE.md`](docs/GUI_GUIDE.md) — desktop workflow and controls;
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — canonical experiment procedure;
- [`docs/METHOD_ALIGNMENT.md`](docs/METHOD_ALIGNMENT.md) — mathematical contract and implementation map;
- [`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) — release acceptance evidence;
- [`docs/PUBLISHING.md`](docs/PUBLISHING.md) — GitHub release checklist.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). After archival, add the release DOI to the GitHub release and the article's software-availability statement without changing the tagged source.
