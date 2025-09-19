"""Backwards compatibility shim exposing the packaged transform framework."""

from infa_to_pyspark import transform_framework as _tf
from infa_to_pyspark.transform_framework import *  # noqa: F401,F403

__all__ = getattr(_tf, "__all__", [name for name in dir(_tf) if not name.startswith("_")])
del _tf
