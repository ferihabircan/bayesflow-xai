"""Backward-compatibility shim.

Canonical location: xai.simulation.sir.adapter
Old code/notebooks importing `xai.simulation.adapter` keep working.
"""

from xai.simulation.sir.adapter import build_simulator, build_adapter  # noqa: F401
