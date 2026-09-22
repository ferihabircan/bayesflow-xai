"""Backward-compatibility shim.

Canonical location: xai.simulation.grf.grf_generative_model
Old code/notebooks importing `xai.simulation.grf_generative_model` keep
working.
"""

from xai.simulation.grf.grf_generative_model import (  # noqa: F401
    distribution,
    prior,
    likelihood,
    build_grf_generative_simulator,
)
