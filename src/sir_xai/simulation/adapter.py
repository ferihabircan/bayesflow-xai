"""BayesFlow simulator + adapter wiring. Imports bayesflow, so kept separate
from the pure-numpy sir_model module."""

import bayesflow as bf

from sir_xai.simulation.sir_model import prior, stationary_SIR


def build_simulator():
    return bf.make_simulator([prior, stationary_SIR])


def build_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("cases")
        .concatenate(["lambd", "mu", "D", "I0", "psi"], into="inference_variables")
        .rename("cases", "summary_variables")
        .log(["inference_variables", "summary_variables"], p1=True)
    )
