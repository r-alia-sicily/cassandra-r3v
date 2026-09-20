#!/usr/bin/env python3
"""Create a chat-friendly deterministic source dump without data or artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".cff", ".sh", ".bat", ".in"}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "cassandra_workspace", "validation_workspace", "build", "dist"}


def files(root: Path, output: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root)
        if any(
            part in EXCLUDED_PARTS or part.startswith(".") or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "Makefile"}:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    selected = list(files(root, output))
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("*** PROJECT STRUCTURE ***\n")
        for path in selected:
            handle.write(f"{path.relative_to(root)}\n")
        handle.write("\n*** FILE CONTENTS ***\n")
        for path in selected:
            relative = path.relative_to(root)
            handle.write(f"\n+++ START ./{relative} +++\n")
            handle.write(path.read_text(encoding="utf-8", errors="replace"))
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                handle.write("\n")
            handle.write(f"=== END ./{relative} ===\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
