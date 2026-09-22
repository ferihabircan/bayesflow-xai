"""Prior, likelihood, and BayesFlow wiring for the Gaussian Random Field
(GRF) parameter-inference scenario from BayesFlow's official example
(Spatial_Data_and_Parameters.html, section 3.1). The prior/likelihood/
simulator code below is kept verbatim from that example; the only change is
that the module-level `rng` is seeded (via CONFIG.seed, and reseedable
through `xai.utils.config.set_seed`) instead of left unseeded, so a full
run is reproducible like the rest of this project.

`distribution()` deliberately keeps using the global `np.random.normal`
call (not the local seeded `rng`), matching the original example verbatim
-- like lotka_volterra_model.py's sample_fn, this is covered by
`np.random.seed()` inside `set_seed()`, not by `rng` itself.
"""

import numpy as np
import bayesflow as bf
from FyeldGenerator import generate_field

from xai.utils.config import CONFIG

FIELD_SHAPE = (64, 64)
PARAM_NAMES = ["log_std", "alpha"]


def generate_power_spectrum(alpha, scale):
    def power_spectrum(k):
        return np.power(k, -alpha) * scale**2
    return power_spectrum


def distribution(shape):
    a = np.random.normal(loc=0, scale=np.sqrt(np.prod(shape)), size=shape)
    b = np.random.normal(loc=0, scale=np.sqrt(np.prod(shape)), size=shape)
    return a + 1j * b


rng = np.random.default_rng(CONFIG.seed)


def prior():
    return {"log_std": rng.normal(), "alpha": rng.normal(loc=4, scale=0.5)}


def likelihood(log_std, alpha, field_shape=(64, 64)):
    field = generate_field(
        distribution, generate_power_spectrum(alpha, np.exp(log_std)), field_shape
    )
    return {"field": field[..., None]}


def build_grf_simulator():
    return bf.make_simulator([prior, likelihood])
