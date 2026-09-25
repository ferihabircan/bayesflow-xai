"""Backward-compatibility shim.

Canonical location: bayesflow_xai.simulation.grf.grf_generative_model
Old code/notebooks importing `bayesflow_xai.simulation.grf_generative_model` keep
working.
"""

from bayesflow_xai.simulation.grf.grf_generative_model import (  # noqa: F401
    distribution,
    prior,
    likelihood,
    build_grf_generative_simulator,
)
