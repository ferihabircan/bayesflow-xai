"""Revised prior/likelihood/simulator for the Gaussian Random Field (GRF)
image-generation scenario from BayesFlow's official example
(Spatial_Data_and_Parameters.html, section 3.2). Unlike grf_model.py
(section 3.1, parameter inference), here the field itself is the inference
target and (log_std, alpha) are conditions, broadcast to `params_expanded`
so the diffusion model can condition on them pixel-wise. The `unit_length`
passed to `generate_field` is the tutorial's fix for the extreme field-value
range of the section 3.1 simulator -- kept verbatim from the example.

The prior/likelihood code below is kept verbatim from the tutorial; only the
module-level `rng` seed source is wired to CONFIG.seed (via
`bayesflow_xai.utils.config.set_seed`) instead of the tutorial's hardcoded 42,
matching how grf_model.py is seeded.
"""

import numpy as np
import bayesflow as bf
from FyeldGenerator import generate_field

from bayesflow_xai.simulation.grf.grf_model import generate_power_spectrum, FIELD_SHAPE
from bayesflow_xai.utils.config import CONFIG

rng = np.random.default_rng(CONFIG.seed)


def distribution(shape):
    a = rng.normal(loc=0, scale=1.0, size=shape)
    b = rng.normal(loc=0, scale=1.0, size=shape)
    return a + 1j * b


def prior():
    log_std = rng.normal(scale=0.3)
    alpha = rng.normal(loc=3, scale=0.5)
    params_expanded = np.array([log_std, alpha])
    params_expanded = np.ones(FIELD_SHAPE + (2,)) * params_expanded[None, None, :]
    return {
        "log_std": log_std,
        "alpha": alpha,
        "params_expanded": params_expanded,
    }


def likelihood(log_std, alpha, field_shape=FIELD_SHAPE):
    field = generate_field(
        distribution,
        generate_power_spectrum(alpha, np.exp(log_std)),
        field_shape,
        unit_length=1 / (np.abs(alpha) + 1e-7),
    )
    return {"field": field[..., None]}


def build_grf_generative_simulator():
    return bf.make_simulator([prior, likelihood])
