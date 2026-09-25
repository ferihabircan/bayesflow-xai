"""The notebook's original `GravitationalWaveBenchmarkSimulator`, vendored
verbatim from sbi-practical-guide (MIT, Copyright (c) 2024 sbi):

    https://github.com/sbi-dev/sbi-practical-guide  (commit 18852e1)
    paper/fig8_grav_wave/workflow/scripts/external/{simulator,base}.py + *.ini

`simulator.py`, `base.py` and both .ini files are unmodified copies (the
notebook's `paper.fig8_grav_wave_npe.smk...` import path is an older
layout of the same directory). The class itself is Joeri Hermans'
`hypothesis` GW benchmark (BSD-3-Clause) with ra/dec/polarization read from
the .ini's static params instead of drawn.

This module only adds `build_guide_tensor_dataset`, which reproduces
paper/fig8_grav_wave/workflow/scripts/script-generate-gws.py (prior, mass
conversion, multiprocessing) and returns the registry's (X, y) format.
"""

import os
from multiprocessing import get_context

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_file_pycbcmaster.ini")

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
    """Draws (mass1, mass_ratio) from the guide's prior and simulates them.
    Returns (thetas (n, 2), xs (n, 2, 8192) float64 raw whitened strain)."""
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(THETA_LOW, THETA_HIGH, size=(n_sims, 2))
    chunks = [
        (thetas[i : i + chunk_size], seed * 1_000_003 + i) for i in range(0, n_sims, chunk_size)
    ]
    num_workers = num_workers or min(32, os.cpu_count() or 1)
    with get_context("fork").Pool(num_workers) as pool:
        xs = np.concatenate(pool.map(_simulate_chunk, chunks))
    return thetas, xs


def build_guide_tensor_dataset(n_sims: int, seed: int = 0, normalization: str = "minmax"):
    """(n_sims) -> (X, y) torch tensors for the registry: X (n, 8192, 2)
    [H1, L1], y (n, 2) = [mass1, mass_ratio].

    normalization="minmax" (default) scales X with one global (min, max)
    over the generated set, like gws-split-denovo.py (norm_style="uniform")
    that produced the notebook's gws-train.h5. "zscore" uses one global
    (mean, std) instead, like our own build_gw_tensor_dataset."""
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
