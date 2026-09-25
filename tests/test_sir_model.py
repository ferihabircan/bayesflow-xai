"""Unit tests for the pure-numpy simulator (no bayesflow/torch needed)."""

import numpy as np

from bayesflow_xai.simulation.sir_model import prior, stationary_SIR


def test_prior_keys():
    theta = prior()
    assert set(theta.keys()) == {"lambd", "mu", "D", "I0", "psi"}
    assert all(v > 0 for v in theta.values())


def test_stationary_sir_output_shape():
    theta = prior()
    out = stationary_SIR(**theta)
    assert "cases" in out
    assert len(out["cases"]) == 14  # CONFIG.horizon_days
    assert np.all(out["cases"] >= 0)


def test_stationary_sir_full_trajectories_sum_to_population_share():
    theta = prior()
    out = stationary_SIR(**theta, return_full=True)
    total = out["S"] + out["I"] + out["R"]
    # S + I + R should stay ~1 (normalized by N) at every timestep
    assert np.allclose(total, 1.0, atol=1e-3)
