"""SIR epidemic simulator subpackage. See README.md for a model summary.

`sir_model` is pure numpy (importable without bayesflow/keras/torch);
`adapter` additionally imports bayesflow, so it is exported here too but
only actually loads bayesflow when something imports it.
"""

from xai.simulation.sir.sir_model import prior, stationary_SIR  # noqa: F401

__all__ = ["prior", "stationary_SIR"]
