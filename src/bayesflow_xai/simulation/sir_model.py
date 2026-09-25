"""Backward-compatibility shim.

Canonical location: bayesflow_xai.simulation.sir.sir_model
Old code/notebooks importing `bayesflow_xai.simulation.sir_model` keep working.
"""

from bayesflow_xai.simulation.sir.sir_model import prior, stationary_SIR, RNG  # noqa: F401
