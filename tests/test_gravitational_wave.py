import numpy as np
import pytest

pytest.importorskip("pycbc")

from bayesflow_xai.simulation import gravitational_wave as gw


def test_window_constants_match_embedding_input():
    assert gw.SAMPLING_RATE * (gw.SECONDS_BEFORE_EVENT + gw.SECONDS_AFTER_EVENT) == 8192
    assert gw.CHANNEL_NAMES == ["H1", "L1"]
    assert gw.PARAM_NAMES == ["mass1", "mass_ratio"]


def test_simulate_shapes_prior_and_seed():
    thetas, xs = gw.simulate(4, seed=0, num_workers=2, chunk_size=2)
    assert thetas.shape == (4, 2)
    assert xs.shape == (4, 2, 8192)
    assert np.isfinite(xs).all()
    assert np.all((thetas >= gw.THETA_LOW) & (thetas <= gw.THETA_HIGH))
    # Same seed -> same parameters and (up to FFT round-off) same strain.
    thetas2, xs2 = gw.simulate(4, seed=0, num_workers=1, chunk_size=2)
    assert np.array_equal(thetas, thetas2)
    assert np.allclose(xs, xs2, rtol=0, atol=1e-6 * np.abs(xs).max())


def test_mass_conversion_matches_original_script():
    # script-generate-gws.py feeds [mass1, mass_ratio * mass1] to the simulator.
    import torch
    from bayesflow_xai.simulation.gravitational_wave.sbi_practical_guide import CONFIG_PATH
    from bayesflow_xai.simulation.gravitational_wave.sbi_practical_guide.simulator import (
        GravitationalWaveBenchmarkSimulator,
    )

    seen = []
    sim = GravitationalWaveBenchmarkSimulator(CONFIG_PATH)
    sim._simulate_gw = lambda m1, m2: seen.append((float(m1), float(m2))) or (np.zeros(8192), np.zeros(8192))
    thetas = torch.tensor([[60.0, 0.5]])
    sim(torch.stack([thetas[:, 0], thetas[:, 1] * thetas[:, 0]], dim=1))
    assert seen == [(60.0, 30.0)]


@pytest.mark.parametrize("normalization", ["minmax", "zscore"])
def test_tensor_dataset(normalization):
    X, y = gw.build_gw_tensor_dataset(3, seed=0, normalization=normalization)
    assert tuple(X.shape) == (3, 8192, 2)
    assert tuple(y.shape) == (3, 2)
    assert bool(X.isfinite().all())
    if normalization == "minmax":
        assert float(X.min()) == 0.0 and float(X.max()) == 1.0
    else:
        assert abs(float(X.mean())) < 1e-3 and abs(float(X.std()) - 1) < 1e-3
    with pytest.raises(ValueError):
        gw.build_gw_tensor_dataset(1, normalization="nope")


def test_paper_embedding_matches_dataset():
    import torch
    from bayesflow_xai.simulation.gravitational_wave.embedding import GWPaperCNNSummaryNet

    X, _ = gw.build_gw_tensor_dataset(2, seed=0, normalization="zscore")
    net = GWPaperCNNSummaryNet(in_channels=2, n_timepoints=X.shape[1])
    with torch.no_grad():
        out = net(X)
    assert tuple(out.shape) == (2, net.summary_dim) == (2, 16)
    assert bool(out.isfinite().all())
    with pytest.raises(ValueError):
        GWPaperCNNSummaryNet(n_timepoints=1024)
