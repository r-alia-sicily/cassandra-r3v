# Reproducing the Cassandra R3v 1.0.0 experiments

This is the canonical procedure for results attributed to Cassandra in the associated article. Run it from a clean checkout of tag `v1.0.0`.

## 1. Record the source identity

```bash
git checkout v1.0.0
git rev-parse HEAD
git status --short
```

The status output must be empty before the campaign begins. Record the tag and commit SHA in the article evidence bundle.

## 2. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip freeze > environment.txt
cassandra-r3v --version
```

The version command must print `cassandra-r3v 1.0.0`. Also record the operating system, CPU architecture, available memory, and whether numerical libraries use multiple threads. Small floating-point differences between platforms are possible; structural decisions near a threshold must therefore be checked against the saved evidence rather than assumed to be bitwise portable.

## 3. Obtain NASA C-MAPSS

Download the C-MAPSS Turbofan Engine Degradation Simulation data from NASA's repository under its applicable terms. Cassandra accepts the outer archive or the inner `CMAPSSData.zip`.

The archive used for the release acceptance run had SHA-256:

```text
c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2
```

This identifies the tested archive, not every possible repackaging of the same NASA files. Full Test records hashes for the 12 extracted input files.

## 4. Import and prepare all eight experiments

```bash
cassandra-r3v --workspace cassandra_workspace import-data /path/to/nasa_cmapss.zip
cassandra-r3v --workspace cassandra_workspace prepare \
  --subsets FD001 FD002 FD003 FD004 \
  --protocols uncapped cap125
```

Preparation fits standardization and fuzzy partitions on TRAIN only. The two target protocols produce independent artifacts.

## 5. Train the Full24 backbones

```bash
cassandra-r3v --workspace cassandra_workspace train \
  --subsets FD001 FD002 FD003 FD004 \
  --protocols uncapped cap125 \
  --epochs 250 --learning-rate 0.025 --ridge 0.9 --patience 50
```

Training writes both `model_backbone.json` and the initial `model.json`. All non-exposed settings are the canonical values in `examples/article_profile.json`.

## 6. Run AutoStructure

```bash
cassandra-r3v --workspace cassandra_workspace autostructure \
  --subsets FD001 FD002 FD003 FD004 \
  --protocols uncapped cap125 \
  --min-support 0.01 \
  --activation-support-cutoff 0.01 \
  --min-novelty 0.005 \
  --min-relative-improvement 0.01 \
  --folds 3 --seeds 7 42 99 --consensus-required 2 \
  --max-order 3 --screen-top-k 24 \
  --pruning-threshold 0.002 --parent-threshold 0.0002 \
  --validation-epochs 100 --max-minutes 60 \
  --max-candidates 10000 --validations-per-seed 4 --max-rounds 3
```

By default this command always restarts from the retained Full24 backbone. The 60-minute guard applies separately to each experiment; running all eight pairs serially can therefore take several hours. For recoverability, each pair may be run as an individual command with the same settings.

AutoStructure writes a report whether or not it accepts a rule. A reported pruning candidate is not removed.

## 7. Evaluate official TEST endpoints

```bash
cassandra-r3v --workspace cassandra_workspace evaluate \
  --subsets FD001 FD002 FD003 FD004 \
  --protocols uncapped cap125
```

This command writes two files per experiment:

- `{protocol}.json`, containing final-model endpoint metrics and faithful rule-responsibility diagnostics;
- `{protocol}_backbone_vs_autostructure.json`, containing the paired backbone–final comparison.

The comparison uses identical official TEST engines and a deterministic 20,000-resample paired percentile bootstrap with seed `20260916`. A positive gain means that the final NASA score is lower. TEST data are used only for final evaluation and never for structural selection.

Article tables must be generated from these files under `evaluations/FDxxx/`. Keep uncapped and cap125 columns separate and report `no_change`, `supported_improvement`, `supported_degradation`, or `inconclusive` exactly as recorded.

## 8. Run acceptance diagnostics

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src
cassandra-r3v --workspace cassandra_workspace full-test
```

A scientific campaign is not accepted if Full Test reports a failure. Full Test smoke metrics are not article metrics.

## 9. Preserve the evidence bundle

Archive, without modifying them:

- the `v1.0.0` commit SHA and clean-status record;
- `examples/article_profile.json`;
- `environment.txt` and machine description;
- all prepared NPZ files;
- all `model_backbone.json` and final `model.json` files;
- all `autostructure.json` files;
- all evaluation JSON files;
- all `*_backbone_vs_autostructure.json` files;
- the complete Full Test TXT report;
- the command log and SHA-256 checksums of every archived artifact.

NASA raw data should remain outside the public software repository. Preserve their hashes so an authorized copy can be verified.

## 10. Article consistency checks

Before submission, confirm that:

- every numerical table can be traced to a saved evaluation field;
- every claimed interaction appears in both the structure report and final model;
- every claimed AutoStructure effect is traceable to a paired-comparison JSON and includes its uncertainty interval;
- the methods section matches the profile values exactly;
- no smoke-test number is presented as an experimental result;
- the software-availability statement identifies Cassandra R3v 1.0.0, tag `v1.0.0`, the commit SHA, repository URL, license, and archival DOI.
