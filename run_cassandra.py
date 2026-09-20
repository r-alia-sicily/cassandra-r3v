#!/usr/bin/env python3
"""Launch Cassandra directly from an unpacked source archive."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cassandra_r3v.gui.app import launch


if __name__ == "__main__":
    launch()
