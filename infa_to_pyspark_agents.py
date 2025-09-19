"""Backwards compatibility shim for legacy imports."""

from infa_to_pyspark import agents as _agents
from infa_to_pyspark.agents import *  # noqa: F401,F403

__all__ = getattr(_agents, "__all__", [name for name in dir(_agents) if not name.startswith("_")])
del _agents
