import numpy as np
import pytest

pytest.importorskip("pycbc")

from bayesflow_xai.simulation.gravitational_wave import simulator as gw


def test_prior_ranges_and_mass_ordering():
    p = gw.sample_prior(2000, rng=0)
    for name, (_, low, high) in gw.PRIORS.items():
        assert p[name].min() >= low and p[name].max() <= high
    for name, (low, high) in gw.THETA_PRIOR.items():
        assert p[name].min() >= low and p[name].max() <= high
    assert np.allclose(p["mass2"], p["mass_ratio"] * p["mass1"])
    assert np.all(p["mass1"] >= p["mass2"])


def test_noise_free_norm_equals_optimal_snr():
    from pycbc.filter import sigma
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import FrequencySeries

    params = dict(mass1=60.0, mass2=30.0, spin1z=-0.85, spin2z=0.85,
                  inclination=0.4, distance=400.0, tc=gw.TC_REF + 0.05, coa_phase=1.0)
    x = gw.simulate_one(params, rng=0, add_noise=False)
    psd = aLIGOZeroDetHighPower(gw.N_FREQ, gw.DELTA_F, gw.F_LOWER)
    hp, hc = gw._polarisations(params)
    for i, (fp, fc, _) in enumerate(gw.detector_response(params["tc"])):
        h = FrequencySeries(fp * hp + fc * hc, delta_f=gw.DELTA_F)
        snr = sigma(h, psd=psd, low_frequency_cutoff=gw.F_LOWER)
        assert np.isclose(np.linalg.norm(x[i]), snr, rtol=1e-3)


def test_detector_response_matches_pycbc():
    from pycbc.detector import Detector

    tc = gw.TC_REF + 0.03
    for (fp, fc, dt), name in zip(gw.detector_response(tc), gw.CHANNEL_NAMES):
        det = Detector(name)
        assert (fp, fc) == det.antenna_pattern(gw.RA, gw.DEC, gw.POLARIZATION, tc)
        assert dt == det.time_delay_from_earth_center(gw.RA, gw.DEC, tc)
    # H1-L1 separation is ~10 ms; any sky position must stay within it.
    (_, _, dt_h1), (_, _, dt_l1) = gw.detector_response(tc)
    assert abs(dt_h1 - dt_l1) < 0.0101


def test_tensor_dataset_shapes():
    X, y = gw.build_gw_tensor_dataset(3, rng=0)
    assert tuple(X.shape) == (3, 8192, 2)
    assert tuple(y.shape) == (3, 2)
    assert bool(X.isfinite().all())
    assert bool(((y[:, 1] >= 0.25) & (y[:, 1] <= 0.99)).all())


def test_paper_embedding_matches_dataset():
    import torch
    from bayesflow_xai.simulation.gravitational_wave.embedding import GWPaperCNNSummaryNet

    X, _ = gw.build_gw_tensor_dataset(2, rng=0)
    net = GWPaperCNNSummaryNet(in_channels=2, n_timepoints=X.shape[1])
    with torch.no_grad():
        out = net(X)
    assert tuple(out.shape) == (2, net.summary_dim) == (2, 16)
    assert bool(out.isfinite().all())
    with pytest.raises(ValueError):
        GWPaperCNNSummaryNet(n_timepoints=1024)
