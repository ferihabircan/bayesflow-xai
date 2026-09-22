"""Lotka-Volterra-style example simulator subpackage. See README.md.

Pure numpy (no bayesflow/keras/torch dependency at import time); safe to
import in isolation for testing.
"""

from xai.simulation.lotka_volterra.lotka_volterra_model import (  # noqa: F401
    sample_fn,
    PARAM_NAMES,
    CHANNEL_NAMES,
)

__all__ = ["sample_fn", "PARAM_NAMES", "CHANNEL_NAMES"]
