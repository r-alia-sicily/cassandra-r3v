# Contributing

Thank you for helping improve Cassandra R3v.

## Before opening a change

- Open an issue describing the problem, expected behavior, and scientific impact.
- Keep the public API, artifact schema, and `uncapped`/`cap125` separation explicit.
- Do not commit NASA data, generated workspaces, trained models, or article evidence.
- Do not change a scientific default without updating the profile, documentation, tests, and rationale.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src
```

## Pull requests

A pull request should be focused, written in English, and include:

- a concise explanation of the change;
- tests for new behavior or a reproducible defect;
- documentation updates when commands, artifacts, or scientific semantics change;
- confirmation that no restricted data or secrets are included.

Changes to the mathematical contract require explicit review. Diagnostic smoke metrics must never be presented as scientific benchmark results.
