"""Builds the explicit (S, I, R) tensor dataset used by the XAI surrogate
models in xai/integrated_gradients.py and xai/attention_rollout.py."""

import numpy as np
import torch

from sir_xai.simulation.sir_model import prior, stationary_SIR
from sir_xai.utils.config import CONFIG, PARAM_NAMES


def ensure_sir_fraction_scale(traj: np.ndarray) -> np.ndarray:
    """Keep S/I/R on a comparable scale.

    The simulator currently returns normalized trajectories, but this guard makes
    the XAI pipeline robust if raw counts are ever passed in.
    """
    if np.max(traj) > 1.5:
        return traj / CONFIG.population_size
    return traj


def build_sir_tensor_dataset(n_sims: int):
    """Returns X: (N, T, 3) [S, I, R] trajectories, y: (N, 5) log1p(theta)."""
    thetas, trajs = [], []
    for _ in range(n_sims):
        theta = prior()
        sim = stationary_SIR(**theta, return_full=True)
        traj = np.stack([sim["S"], sim["I"], sim["R"]], axis=-1)[-CONFIG.horizon_days:]
        traj = ensure_sir_fraction_scale(traj)
        thetas.append([theta[name] for name in PARAM_NAMES])
        trajs.append(traj)

    X = torch.tensor(np.stack(trajs), dtype=torch.float32)
    y = torch.tensor(np.log1p(np.array(thetas)), dtype=torch.float32)
    return X, y
