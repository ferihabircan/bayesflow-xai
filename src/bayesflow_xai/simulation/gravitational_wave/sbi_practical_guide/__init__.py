"""The notebook's original `GravitationalWaveBenchmarkSimulator`, vendored
verbatim from sbi-practical-guide (MIT, Copyright (c) 2024 sbi):

    https://github.com/sbi-dev/sbi-practical-guide  (commit 18852e1)
    paper/fig8_grav_wave/workflow/scripts/external/{simulator,base}.py + *.ini

`simulator.py`, `base.py` and both .ini files are unmodified copies (the
notebook's `paper.fig8_grav_wave_npe.smk...` import path is an older
layout of the same directory). The class itself is Joeri Hermans'
`hypothesis` GW benchmark (BSD-3-Clause) with ra/dec/polarization read from
the .ini's static params instead of drawn.

This module only adds `build_gw_tensor_dataset`, which reproduces
paper/fig8_grav_wave/workflow/scripts/script-generate-gws.py (prior, mass
conversion, multiprocessing) and returns the registry's (X, y) format.
"""

import configparser
import os
from multiprocessing import get_context

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_file_pycbcmaster.ini")

_STATIC = configparser.ConfigParser(inline_comment_prefixes=(";",))
_STATIC.read(CONFIG_PATH)
_STATIC = _STATIC["static_params"]
SAMPLING_RATE = int(float(_STATIC["sampling_rate"]))  # 2048 Hz
# Every sample is aligned so the H1 event (merger) is this far into the window.
SECONDS_BEFORE_EVENT = float(_STATIC["seconds_before_event"])  # 3.5 s
SECONDS_AFTER_EVENT = float(_STATIC["seconds_after_event"])  # 0.5 s -> 8192 samples in total

PARAM_NAMES = ["mass1", "mass_ratio"]  # mass_ratio = mass2 / mass1
CHANNEL_NAMES = ["H1", "L1"]

# script-generate-gws.py: BoxUniform(low=[40, 0.25], high=[80, 0.99]) over
# (mass1, mass_ratio = mass2 / mass1).
THETA_LOW = np.array([40.0, 0.25])
THETA_HIGH = np.array([80.0, 0.99])


def _simulate_chunk(args):
    import torch

    from .simulator import GravitationalWaveBenchmarkSimulator

    thetas, seed = args
    # The simulator's nuisance draws (pval.rvs) and noise_from_psd's seed
    # both come from numpy's global RNG: seed each chunk separately so forked
    # workers don't produce identical noise.
    np.random.seed(seed)
    thetas = torch.as_tensor(thetas, dtype=torch.float32)
    masses = torch.stack([thetas[:, 0], thetas[:, 1] * thetas[:, 0]], dim=1)
    return GravitationalWaveBenchmarkSimulator(CONFIG_PATH)(masses).numpy()


def simulate(n_sims: int, seed: int = 0, num_workers: int | None = None, chunk_size: int = 25):
    """Draws (mass1, mass_ratio) from the original prior and simulates them.
    Returns (thetas (n, 2), xs (n, 2, 8192) float64 raw whitened strain)."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(THETA_LOW, THETA_HIGH, size=(n_sims, 2))
    chunks = [
        (thetas[i : i + chunk_size], seed * 1_000_003 + i) for i in range(0, n_sims, chunk_size)
    ]
    num_workers = num_workers or min(32, os.cpu_count() or 1)
    # "spawn", not "fork": forking a process whose torch/OpenMP thread pools
    # are already running can deadlock the workers (seen under pytest).
    with get_context("spawn").Pool(num_workers) as pool:
        xs = np.concatenate(pool.map(_simulate_chunk, chunks))
    return thetas, xs


def build_gw_tensor_dataset(n_sims: int, seed: int = 0, normalization: str = "minmax"):
    """(n_sims) -> (X, y) torch tensors for the registry: X (n, 8192, 2)
    [H1, L1], y (n, 2) = [mass1, mass_ratio].

    normalization="minmax" (default) scales X with one global (min, max)
    over the generated set, like gws-split-denovo.py (norm_style="uniform")
    that produced the notebook's gws-train.h5. "zscore" uses one global
    (mean, std) instead."""
    import torch

    thetas, xs = simulate(n_sims, seed=seed)
    if normalization == "minmax":
        xs = (xs - xs.min()) / (xs.max() - xs.min())
    elif normalization == "zscore":
        xs = (xs - xs.mean()) / xs.std()
    else:
        raise ValueError(f"Unknown normalization '{normalization}', expected 'minmax' or 'zscore'")
    X = torch.tensor(xs.transpose(0, 2, 1), dtype=torch.float32)
    y = torch.tensor(thetas, dtype=torch.float32)
    return X, y
