# Validation report — Cassandra R3v 1.0.0

Real-data acceptance run: 2026-09-16 (UTC)
Final documentation and package audit: 2026-09-18 (UTC)
Public release publication date: 2026-09-20

## Release result

The canonical real-data Full Test completed with **PASS=88, WARN=0, SKIP=0, FAIL=0**. It covered FD001–FD004 under both `uncapped` and `cap125`, keeping all eight experiments separate.

The NASA outer ZIP used for the run had SHA-256:

`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`

The run checked:

- Python, NumPy, pandas, Tkinter, operating-system, resource, and version information;
- syntax for all 23 package modules and the packaged English help documents;
- built-in fuzzy, NASA-loss, responsibility, explanation, and serialization contracts;
- all 55 automated source tests;
- hashes, dimensions, finiteness, and engine counts for all four C-MAPSS subsets;
- eight independent preparations and their saved NPZ artifacts;
- eight retained Full24 backbones, eight final models, and eight AutoStructure reports;
- eight final-model evaluations and eight paired comparison artifacts;
- fresh paired backbone–AutoStructure comparisons with 20,000 bootstrap resamples each;
- eight two-epoch, non-destructive training/evaluation smoke paths;
- a bounded AutoStructure screening and matched-validation smoke path.

The complete diagnostic narrative is `docs/FULL_TEST_RELEASE_VALIDATION.txt`.

## Automated source suite

Command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Outcome: **55 tests passed; 0 failures; 0 errors; 0 skipped**.

The suite includes dedicated checks for exact no-change detection, supported improvement, deterministic paired bootstrap output, cancellation, and recoverable invalidation of both final and backbone artifacts.

## AutoStructure outcome

Seven experiments accepted no new rule, so their final model is exactly the frozen 120-rule backbone and the paired comparison is `no_change`.

FD003/cap125 accepted one TRAIN-only validated conjunction:

`OS1_Altitude IS LOW AND S20_W31_HPTCoolantBleed IS LOW`

The final FD003/cap125 model therefore contains 121 rules. Its official TEST comparison is:

- backbone NASA score: **855.452**;
- final NASA score: **823.922**;
- point improvement: **+3.686%**;
- paired 95% bootstrap interval: **[-7.676%, +9.488%]**;
- favorable bootstrap resamples: **77.76%**;
- engine penalties: **39 lower, 0 equal, 61 higher**;
- conclusion: **`inconclusive`**.

The point estimate is favorable, but the interval includes zero. Cassandra therefore does not present this result as demonstrated TEST improvement. TEST data did not participate in rule selection.

| Subset | Protocol | Rules | Backbone NASA | Final NASA | Gain | 95% interval | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FD001 | uncapped | 120→120 | 747.763 | 747.763 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD001 | cap125 | 120→120 | 510.572 | 510.572 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD002 | uncapped | 120→120 | 316,899.106 | 316,899.106 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD002 | cap125 | 120→120 | 6,699.852 | 6,699.852 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD003 | uncapped | 120→120 | 335,706.067 | 335,706.067 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD003 | cap125 | 120→121 | 855.452 | 823.922 | +3.686% | [-7.676%, +9.488%] | `inconclusive` |
| FD004 | uncapped | 120→120 | 161,796.562 | 161,796.562 | +0.000% | [0.000%, 0.000%] | `no_change` |
| FD004 | cap125 | 120→120 | 7,388.818 | 7,388.818 | +0.000% | [0.000%, 0.000%] | `no_change` |

## NASA data inventory

| Subset | TRAIN rows | TEST rows | TRAIN engines | TEST engines |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 20,631 | 13,096 | 100 | 100 |
| FD002 | 53,759 | 33,991 | 260 | 259 |
| FD003 | 24,720 | 16,596 | 100 | 100 |
| FD004 | 61,249 | 41,214 | 249 | 248 |

## End-to-end diagnostic smoke metrics

These fresh two-epoch models verify executable paths and numerical finiteness. They are **not publication results**.

| Subset | Protocol | Features | Rules | NASA score | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | uncapped | 24 | 120 | 1,253.230 | 23.374 | 26.522 |
| FD001 | cap125 | 24 | 120 | 1,070.508 | 23.630 | 26.338 |
| FD002 | uncapped | 24 | 120 | 94,759.238 | 41.158 | 49.174 |
| FD002 | cap125 | 24 | 120 | 15,099.243 | 35.912 | 40.020 |
| FD003 | uncapped | 24 | 120 | 5,151.766 | 23.235 | 28.162 |
| FD003 | cap125 | 24 | 120 | 1,207.433 | 21.240 | 24.595 |
| FD004 | uncapped | 24 | 120 | 88,866.572 | 42.191 | 49.845 |
| FD004 | cap125 | 24 | 120 | 20,910.172 | 35.852 | 40.070 |

## Package and interface boundary

The source tree compiles, the wheel installs in an isolated environment, and the version, general help, AutoStructure help, and Full Test help commands execute successfully. The GUI modules compile and automated checks confirm six English stages ending in Full Test. Pixel-level rendering remains a target-desktop check because the build environment has no graphical display server.

The 12-page English user manual compiles cleanly from its included LaTeX source. The final PDF was reopened programmatically, its text and metadata were checked, all fonts were confirmed embedded, and every rendered page was visually inspected for clipping, overlap, and legibility.
