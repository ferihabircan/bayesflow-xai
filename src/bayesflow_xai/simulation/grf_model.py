"""Backward-compatibility shim.

Canonical location: bayesflow_xai.simulation.grf.grf_model
Old code/notebooks importing `bayesflow_xai.simulation.grf_model` keep working.
"""

from bayesflow_xai.simulation.grf.grf_model import (  # noqa: F401
    FIELD_SHAPE,
    PARAM_NAMES,
    generate_power_spectrum,
    distribution,
    prior,
    likelihood,
    build_grf_simulator,
)
