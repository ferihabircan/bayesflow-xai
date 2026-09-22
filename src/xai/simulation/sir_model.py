"""Backward-compatibility shim.

Canonical location: xai.simulation.sir.sir_model
Old code/notebooks importing `xai.simulation.sir_model` keep working.
"""

from xai.simulation.sir.sir_model import prior, stationary_SIR, RNG  # noqa: F401
