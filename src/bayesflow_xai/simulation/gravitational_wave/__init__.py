"""Gravitational-wave (compact binary, PyCBC IMRPhenomPv2) simulator
subpackage. See README.md -- in particular: this is a PyCBC-based
alternative, NOT the notebook's GravitationalWaveBenchmarkSimulator.

Only numpy is needed at import time; pycbc/lal/scipy/torch are imported
lazily inside the functions that use them.
"""

from bayesflow_xai.simulation.gravitational_wave.simulator import (  # noqa: F401
    simulate,
    simulate_one,
    sample_prior,
    build_gw_tensor_dataset,
    PRIORS,
    PARAM_NAMES,
    CHANNEL_NAMES,
)

__all__ = [
    "simulate",
    "simulate_one",
    "sample_prior",
    "build_gw_tensor_dataset",
    "PRIORS",
    "PARAM_NAMES",
    "CHANNEL_NAMES",
]
