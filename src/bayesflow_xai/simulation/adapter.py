"""Backward-compatibility shim.

Canonical location: bayesflow_xai.simulation.sir.adapter
Old code/notebooks importing `bayesflow_xai.simulation.adapter` keep working.
"""

from bayesflow_xai.simulation.sir.adapter import build_simulator, build_adapter  # noqa: F401
