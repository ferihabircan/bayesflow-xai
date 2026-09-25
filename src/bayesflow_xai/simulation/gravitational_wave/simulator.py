"""Compact-binary gravitational-wave simulator built on PyCBC/LALSuite.

IMPORTANT -- what this is and is not
------------------------------------
This is NOT a reproduction of the `GravitationalWaveBenchmarkSimulator` used
in the professor's `4_1_grav_waves.ipynb` notebook: that class lives in an
external package (`paper.fig8_grav_wave_npe...`) and relies on pre-generated
data files (`data/gws-train.h5`, ~2.5 GB), neither of which we have.

Instead it is a physically reasonable alternative that uses PyCBC's own
waveform generator with the settings of `config_file_pycbcmaster.ini`:

    approximant     = IMRPhenomPv2
    f_lower = f_ref = 20 Hz
    sampling_rate   = 2048 Hz
    waveform_length = 128 s

Pipeline per simulation:
  1. draw the inference targets from the notebook's prior (THETA_PRIOR:
     mass1 ~ U(40, 80), mass_ratio ~ U(0.25, 0.99), mass2 = ratio * mass1 --
     the notebook passes these thetas to the simulator, overriding the
     .ini's mass priors) and the remaining [variable_params] (spin1z,
     spin2z, inclination, distance, tc, coa_phase) from the .ini (PRIORS);
  2. `pycbc.waveform.get_fd_waveform` -> h_plus(f), h_cross(f)
     (delta_f = 1/128 Hz);
  3. full H1/L1 detector response for the .ini's [static_params]
     ra = 3.44615914, dec = -0.40808407, polarization = 0, using PyCBC's
     own `Detector(name).antenna_pattern(ra, dec, polarization, tc)` and
     `Detector(name).time_delay_from_earth_center(ra, dec, tc)`:
         strain_det(t) = F+ h_plus(t - tc - dt_det) + Fx h_cross(t - tc - dt_det)
     with tc the geocentric GPS coalescence time from the .ini's tc prior;
  4. whitening with PyCBC's design-sensitivity PSD `aLIGOZeroDetHighPower`
     and (optionally) coloured Gaussian noise drawn from that same PSD, so
     the whitened noise has unit variance per sample and the whitened
     signal satisfies sum(x**2) == optimal SNR**2 (noise-free);
  5. inverse FFT to a 128 s, 2048 Hz time series; the geocentric merger
     sits at MERGER_TIME + (tc - TC_REF), each detector's at that plus its
     own light-travel delay.

Priors are the .ini's (configs/gravitational_wave/config_file_pycbcmaster.ini).
Known differences from the notebook's simulator, which follows the same
.ini: it whitens real detector noise (noise_interval_width,
whitening_segment_duration); here the noise is Gaussian from the aLIGO
design PSD. The detector projection (antenna pattern + light-travel delay
for the .ini's sky position) and the 4 s / 8192-sample training window
around the H1 event time follow the .ini.
"""

from functools import lru_cache

import numpy as np

APPROXIMANT = "IMRPhenomPv2"
F_LOWER = 20.0
F_REF = 20.0
SAMPLING_RATE = 2048
WAVEFORM_LENGTH = 128  # seconds

DELTA_F = 1.0 / WAVEFORM_LENGTH
N_SAMPLES = WAVEFORM_LENGTH * SAMPLING_RATE
N_FREQ = N_SAMPLES // 2 + 1

# The geocentric merger at tc = TC_REF sits 2 s before the end of the 128 s
# window, so the whole in-band inspiral (at most ~2 s for THETA_PRIOR),
# the ringdown and the <= 21 ms detector delays fit without wrapping around.
MERGER_TIME = WAVEFORM_LENGTH - 2.0

# [static_params] sky position and polarisation (radians).
RA = 3.44615914
DEC = -0.40808407
POLARIZATION = 0.0

# [static_params] seconds_before_event / seconds_after_event: the training
# window around the H1 event (merger) time -> 4 s * 2048 Hz = 8192 samples,
# the input length PaperEmbedding (embedding.py) is built for.
SECONDS_BEFORE_EVENT = 3.5
SECONDS_AFTER_EVENT = 0.5

# Inference targets, prior from the notebook (cell 393ac18c):
# BoxUniform(low=[40, 0.25], high=[80, 0.99]) over (mass1, mass2/mass1).
THETA_PRIOR = {
    "mass1": (40.0, 80.0),
    "mass_ratio": (0.25, 0.99),
}

# Nuisance [variable_params], name -> (distribution, low, high), copied
# from the .ini's [prior-*] sections:
#   uniform        p(x) const on [low, high]
#   uniform_angle  p(x) const on [low, high] (radians; .ini default 0-2pi)
#   sin_angle      p(x) ~ sin(x)  (isotropic inclination; .ini default 0-pi)
#   uniform_radius p(x) ~ x**2   (uniform in Euclidean volume)
# The .ini's [prior-mass1]/[prior-mass2] (uniform 10-80 each) are not used:
# the notebook overrides them with THETA_PRIOR.
# tc is the geocentric GPS coalescence time; it sets the Earth's
# orientation for the antenna pattern, and tc - TC_REF shifts the merger
# within the 128 s series.
PRIORS = {
    "spin1z": ("uniform", -0.9, -0.8),
    "spin2z": ("uniform", 0.8, 0.9),
    "inclination": ("sin_angle", 0.0, np.pi),
    "distance": ("uniform_radius", 1.0, 1.1),  # Mpc
    "tc": ("uniform", 1187008882.4, 1187008882.5),  # GPS s
    "coa_phase": ("uniform_angle", 0.0, 2 * np.pi),
}
TC_REF = PRIORS["tc"][1]
VARIABLE_PARAMS = ["mass1", "mass2", *PRIORS]

# Inference targets exposed to the XAI workflow (y columns, in this order).
PARAM_NAMES = ["mass1", "mass_ratio"]  # mass_ratio = mass2 / mass1
CHANNEL_NAMES = ["H1", "L1"]


def _draw(dist, low, high, n, rng):
    if dist in ("uniform", "uniform_angle"):
        return rng.uniform(low, high, n)
    if dist == "sin_angle":
        return np.arccos(rng.uniform(np.cos(high), np.cos(low), n))
    if dist == "uniform_radius":
        return rng.uniform(low**3, high**3, n) ** (1.0 / 3.0)
    raise ValueError(f"unknown prior distribution '{dist}'")


def sample_prior(n, rng=None):
    """Draws `n` parameter sets as a dict of (n,) arrays: THETA_PRIOR
    targets (mass1, mass_ratio), the derived mass2 = mass_ratio * mass1
    (as the notebook's `simulator` wrapper does), and the PRIORS nuisances."""
    rng = np.random.default_rng(rng)
    out = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in THETA_PRIOR.items()}
    out["mass2"] = out["mass_ratio"] * out["mass1"]
    out.update({k: _draw(d, lo, hi, n, rng) for k, (d, lo, hi) in PRIORS.items()})
    return out


@lru_cache(maxsize=1)
def _whitening():
    """(frequencies, 1/sqrt(PSD) on the [F_LOWER, Nyquist) band, 0 elsewhere)."""
    from pycbc.psd import aLIGOZeroDetHighPower

    psd = aLIGOZeroDetHighPower(N_FREQ, DELTA_F, F_LOWER).numpy()
    freqs = np.arange(N_FREQ) * DELTA_F
    band = (freqs >= F_LOWER) & (psd > 0)
    w = np.zeros(N_FREQ)
    w[band] = 1.0 / np.sqrt(psd[band])
    return freqs, w, band


@lru_cache(maxsize=1)
def _detectors():
    from pycbc.detector import Detector

    return [Detector(name) for name in CHANNEL_NAMES]


def detector_response(tc):
    """[(F+, Fx, delay from Earth centre in s)] for H1, L1 at GPS time
    `tc`, for the .ini's static RA/DEC/POLARIZATION."""
    return [
        (*det.antenna_pattern(RA, DEC, POLARIZATION, tc), det.time_delay_from_earth_center(RA, DEC, tc))
        for det in _detectors()
    ]


def h1_event_time(tc):
    """Time (s from the start of the 128 s series) of the merger as seen
    in H1 -- the .ini's event time that the training window is aligned to."""
    return MERGER_TIME + (tc - TC_REF) + detector_response(tc)[0][2]


def _polarisations(params):
    from pycbc.waveform import get_fd_waveform

    hp, hc = get_fd_waveform(
        approximant=APPROXIMANT,
        mass1=params["mass1"],
        mass2=params["mass2"],
        spin1z=params["spin1z"],
        spin2z=params["spin2z"],
        inclination=params["inclination"],
        distance=params["distance"],
        coa_phase=params["coa_phase"],
        f_lower=F_LOWER,
        f_ref=F_REF,
        delta_f=DELTA_F,
    )
    hp.resize(N_FREQ)
    hc.resize(N_FREQ)
    return hp.numpy(), hc.numpy()


def simulate_one(params, rng=None, add_noise=True):
    """One 128 s whitened H1/L1 strain, shape (2, N_SAMPLES), for a dict of
    scalar [variable_params]. Units: whitened noise has unit variance per
    sample; without noise sum(x**2) equals the optimal SNR**2 of the
    projected signal in that detector."""
    rng = np.random.default_rng(rng)
    freqs, w, band = _whitening()
    hp, hc = _polarisations(params)
    t_geo = MERGER_TIME + (params["tc"] - TC_REF)

    out = np.empty((2, N_SAMPLES), dtype=np.float32)
    for i, (fp, fc, dt) in enumerate(detector_response(params["tc"])):
        hf = (fp * hp + fc * hc) * w * np.exp(-2j * np.pi * freqs * (t_geo + dt))
        if add_noise:
            # Whitened version of PyCBC's frequency_noise_from_psd:
            # E|n|^2 = PSD / (2 delta_f)  ->  1 / (2 delta_f) after whitening.
            sigma = np.sqrt(1.0 / (4.0 * DELTA_F))
            hf = hf + band * sigma * (rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ))
        out[i] = np.fft.irfft(hf, n=N_SAMPLES) * np.sqrt(2.0 * SAMPLING_RATE)
    return out


def simulate(n, rng=None, add_noise=True, window=None, decimate=1):
    """Batch simulator.

    Returns {"parameters": dict of (n,) arrays (8 variable_params +
    mass_ratio), "observables": (n, T, 2) float32}.

    `window=(before, after)` in seconds crops each series to
    [merger - before, merger + after] around the H1 merger time (default:
    the full 128 s). `decimate` > 1 downsamples with an anti-aliasing
    filter (scipy.signal.decimate), dropping content above the new Nyquist.
    """
    rng = np.random.default_rng(rng)
    params = sample_prior(n, rng)
    series = []
    for i in range(n):
        p = {k: float(params[k][i]) for k in VARIABLE_PARAMS}
        x = simulate_one(p, rng, add_noise=add_noise)
        if window is not None:
            start = int(round((h1_event_time(p["tc"]) - window[0]) * SAMPLING_RATE))
            x = x[:, start : start + int(round((window[0] + window[1]) * SAMPLING_RATE))]
        if decimate > 1:
            from scipy.signal import decimate as _decimate

            x = _decimate(x, decimate, ftype="fir", axis=-1, zero_phase=True).astype(np.float32)
        series.append(x.T)
    return {"parameters": params, "observables": np.stack(series)}


# Window used for the XAI tensor dataset: the .ini's 3.5 s before / 0.5 s
# after the H1 event, at the full 2048 Hz -> 8192 steps (PaperEmbedding's
# input length). Covers the whole in-band (>= 20 Hz) inspiral for every
# THETA_PRIOR mass pair (longest: 40 + 10 Msun, ~2 s).
DATASET_WINDOW = (SECONDS_BEFORE_EVENT, SECONDS_AFTER_EVENT)
DATASET_DECIMATE = 1


def build_gw_tensor_dataset(n_sims: int, rng=None):
    """(n_sims) -> (X, y) torch tensors for the registry: X (n, 8192, 2)
    whitened H1/L1 strain around merger, y (n, 2) = [mass1, mass_ratio].

    X is standardised with one global (loc, scale) over the generated set,
    like the notebook's gws-train.h5 (`original_loc`/`original_scale`
    attrs): at the .ini's 1-1.1 Mpc the whitened signal is ~1e2-1e4 times
    the unit noise level, far outside what PaperEmbedding's xavier/SELU
    init expects."""
    import torch

    data = simulate(n_sims, rng=rng, window=DATASET_WINDOW, decimate=DATASET_DECIMATE)
    obs = data["observables"]
    X = torch.tensor((obs - obs.mean()) / obs.std(), dtype=torch.float32)
    y = torch.tensor(np.stack([data["parameters"][k] for k in PARAM_NAMES], axis=1), dtype=torch.float32)
    return X, y
