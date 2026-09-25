"""Gravitational-wave (compact binary, PyCBC IMRPhenomPv2) simulator
subpackage: the notebook's original `GravitationalWaveBenchmarkSimulator`
from sbi-dev/sbi-practical-guide, vendored in `sbi_practical_guide/`, plus
the notebook's PaperEmbedding summary network in `embedding.py`. See
README.md.

pycbc/lal/torch are imported lazily inside the functions that use them.
"""

from bayesflow_xai.simulation.gravitational_wave.sbi_practical_guide import (  # noqa: F401
    simulate,
    build_gw_tensor_dataset,
    PARAM_NAMES,
    CHANNEL_NAMES,
    SAMPLING_RATE,
    SECONDS_BEFORE_EVENT,
    SECONDS_AFTER_EVENT,
    THETA_LOW,
    THETA_HIGH,
)

__all__ = [
    "simulate",
    "build_gw_tensor_dataset",
    "PARAM_NAMES",
    "CHANNEL_NAMES",
    "SAMPLING_RATE",
    "SECONDS_BEFORE_EVENT",
    "SECONDS_AFTER_EVENT",
    "THETA_LOW",
    "THETA_HIGH",
]
