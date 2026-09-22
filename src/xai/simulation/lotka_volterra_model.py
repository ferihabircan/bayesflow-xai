"""Backward-compatibility shim.

Canonical location: xai.simulation.lotka_volterra.lotka_volterra_model
Old code/notebooks importing `xai.simulation.lotka_volterra_model` keep
working.
"""

from xai.simulation.lotka_volterra.lotka_volterra_model import (  # noqa: F401
    N_TIMEPOINTS,
    N_CHANNELS,
    DIM_THETA,
    PARAM_NAMES,
    CHANNEL_NAMES,
    sample_fn,
    build_lv_simulator,
)
