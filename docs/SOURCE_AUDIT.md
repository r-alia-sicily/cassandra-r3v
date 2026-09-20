# Public source audit — Cassandra R3v 1.0.0

This audit defines the contents of the public repository and the material intentionally excluded from it.

## Included

- the `cassandra_r3v` numerical engine and desktop application;
- the headless data, training, AutoStructure, paired evaluation, and Full Test commands;
- automated unit and release-contract tests;
- English user, method, validation, reproducibility, and publishing documentation, including the illustrated LaTeX/PDF manual;
- the canonical experiment profile;
- GPL-3.0-or-later licensing and citation metadata;
- GitHub continuous integration and issue templates;
- platform launchers and a deterministic source-export utility.

## Excluded

- NASA C-MAPSS data and any repackaging of those data;
- generated workspaces, trained parameters, evaluations, and local reports;
- Python caches, virtual environments, build directories, and editor metadata;
- credentials, access tokens, machine-specific paths, and personal configuration;
- article result tables that have not been regenerated and verified with tag `v1.0.0`.

## Release checks

The publication package is accepted only after source tests, source compilation, package build, isolated-environment installation, command-line checks, forbidden-reference scanning, and a Full Test over all eight subset/protocol experiments. The Full Test must include the paired backbone–AutoStructure comparison for every experiment.

Generated scientific artifacts belong in a separately archived evidence bundle. They are not source code and must not be committed to the repository.
