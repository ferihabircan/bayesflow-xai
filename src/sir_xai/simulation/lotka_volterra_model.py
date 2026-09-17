"""Simulator from the professor's Slack example (parallel to sir_model.py).

Despite the "Lotka-Volterra" name given by the source, `sample_fn` is NOT the
predator-prey ODE system -- it's a theta-parametrized sinusoidal signal with
additive Gaussian noise. Kept verbatim (including its use of the global
numpy RNG rather than this project's seeded `sir_model.RNG`) since fidelity
to the given example matters more here than project-style consistency.

No ML dependencies at module load, same as sir_model.py: `sample_fn` only
needs numpy, so this module stays importable/testable without bayesflow/
keras/torch. `build_lv_simulator()` lazily imports bayesflow only if/when a
BayesFlow-style simulator object is actually needed.
"""

import numpy as np

N_TIMEPOINTS, N_CHANNELS, DIM_THETA = 20, 2, 3

PARAM_NAMES = ["theta0", "theta1", "theta2"]
CHANNEL_NAMES = ["X1", "X2"]


def sample_fn(batch_shape):
    n = int(batch_shape[0])
    theta = np.random.uniform(-1.0, 1.0, size=(n, DIM_THETA)).astype("float32")
    t = np.linspace(0, 1, N_TIMEPOINTS)[None, :, None]
    signal = theta[:, None, :2] * np.sin(2 * np.pi * t * (1 + theta[:, None, 2:3]))
    x = (signal + 0.05 * np.random.randn(n, N_TIMEPOINTS, N_CHANNELS)).astype("float32")
    return {"parameters": theta, "observables": x}


def build_lv_simulator():
    """Wraps `sample_fn` as a BayesFlow `LambdaSimulator`, matching the
    professor's `simulator = bf.simulators.LambdaSimulator(sample_fn,
    is_batched=True)`. Only needed if/when the full BasicWorkflow (latent
    space / attention rollout style analyses) gets built for this
    simulator; the XAI stats path in xai/integrated_gradients.py calls
    `sample_fn` directly and never needs this."""
    import bayesflow as bf

    return bf.simulators.LambdaSimulator(sample_fn, is_batched=True)
