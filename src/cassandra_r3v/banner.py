"""Canonical Cassandra console banner."""

from __future__ import annotations

from ._version import __version__


def print_banner() -> None:
    """Print the stable ASCII banner used by Cassandra desktop launchers."""

    print(
        r"""
   ___   _   ___ ___   _   _  _ ___  ___    _
  / __| /_\ / __/ __| /_\ | \| |   \| _ \  /_\
 | (__ / _ \\__ \__ \/ _ \| .` | |) |   / / _ \
  \___/_/ \_\___/___/_/ \_\_|\_|___/|_|_\/_/ \_\

        Rul3volution Neuro-Fuzzy Prognostic System
          guarded six-stage publication workflow
============================================================
"""
    )
    print(f"                 CASSANDRA SUGENO / R3v {__version__}\n")
