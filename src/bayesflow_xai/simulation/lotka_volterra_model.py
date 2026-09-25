"""Backward-compatibility shim.

Canonical location: bayesflow_xai.simulation.lotka_volterra.lotka_volterra_model
Old code/notebooks importing `bayesflow_xai.simulation.lotka_volterra_model` keep
working.
"""

from bayesflow_xai.simulation.lotka_volterra.lotka_volterra_model import (  # noqa: F401
    N_TIMEPOINTS,
    N_CHANNELS,
    DIM_THETA,
    PARAM_NAMES,
    CHANNEL_NAMES,
    sample_fn,
    build_lv_simulator,
)
