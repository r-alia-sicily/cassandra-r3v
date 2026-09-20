# Cassandra R3v 1.0.0

**Release date:** 2026-09-20

Cassandra R3v 1.0.0 is a public implementation of the R3v method, prepared for reproducible research use and future alignment with the associated article.

## Highlights

- transparent zero-order Takagi–Sugeno RUL modelling on NASA C-MAPSS;
- explicit activations, consequents, relative authorities, and local responsibilities;
- NASA-loss optimization with TRAIN-only preparation and grouped validation;
- evidence-controlled structural growth for readable conjunctions;
- strict separation of uncapped and cap125 protocols;
- English desktop and headless workflows;
- cooperative cancellation and atomic, auditable artifacts;
- comprehensive Full Test diagnostics;
- automatic paired backbone–AutoStructure evaluation with a deterministic 20,000-resample engine bootstrap;
- canonical experiment profile and end-to-end reproducibility guide;
- concise illustrated user manual supplied as both LaTeX source and compiled PDF.

## Acceptance status

The release source suite, source compilation, wheel build, isolated-environment installation, CLI checks, and public-reference scan pass. The real-data Full Test covers FD001–FD004 under both protocols and records 88 PASS, 0 WARN, 0 SKIP, and 0 FAIL. The source suite contains 55 tests.

Seven experiments retain the unchanged 120-rule backbone. FD003/cap125 accepts one conjunction and reaches 121 rules. Its NASA point score improves from 855.452 to 823.922 (+3.686%), while the paired 95% bootstrap interval is [-7.676%, +9.488%]; the release therefore reports this TEST effect as `inconclusive`.

NASA data are not included. See `docs/REPRODUCIBILITY.md` for the exact campaign procedure and `docs/VALIDATION_REPORT.md` for release-level evidence.
