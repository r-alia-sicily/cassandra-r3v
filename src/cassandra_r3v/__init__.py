"""Cassandra: a transparent Rul3volution implementation for NASA C-MAPSS."""

from ._version import __version__
from .config import AutoStructureConfig, TrainingConfig
from .model import R3vModel

__all__ = ["AutoStructureConfig", "TrainingConfig", "R3vModel", "__version__"]
