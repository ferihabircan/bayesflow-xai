"""Central run configuration. Import this before anything else that touches
keras/bayesflow/torch, since CUDA_VISIBLE_DEVICES and KERAS_BACKEND must be
set before those libraries are imported."""

import os
import random
import sys

# =============================================================
# GPU & BACKEND SETTINGS
# =============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
# Keras backend set to 'torch' for Captum / PyTorch compatibility
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402  (must come after the env vars above)
import torch  # noqa: E402  (must come after the env vars above)
from dataclasses import dataclass


def _report_device():
    print(f"CUDA available?: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected, falling back to CPU.")
    return torch.device("cpu")


DEVICE = _report_device()


def set_seed(seed: int = 42) -> None:
    """Reseeds every RNG this project touches, so a given training/XAI
    function produces bit-identical results across separate process runs
    when called with the same seed.

    Covers: python's `random`, numpy's global RNG (used directly by
    `lotka_volterra_model.sample_fn`), torch/CUDA, and `sir_model.RNG` /
    `grf_model.rng` -- standalone `np.random.Generator` instances that
    `np.random.seed()` does NOT reach, so they must be reseeded explicitly
    here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Lazy import to avoid a circular import (sir_model imports CONFIG from
    # this module at module load time).
    from bayesflow_xai.simulation.sir import sir_model

    sir_model.RNG = np.random.default_rng(seed)

    # grf_model / grf_generative_model pull in bayesflow/FyeldGenerator at
    # module load time, so each is only reseeded if some other code has
    # already imported it -- this function must not become a hard
    # dependency on the GRF modules for callers (SIR/LV surrogates,
    # transformer) that never touch them. Looked up by their canonical
    # dotted path so the reseed applies no matter whether the caller
    # reached them via the new (bayesflow_xai.simulation.grf.*) or the
    # backward-compat flat (bayesflow_xai.simulation.grf_model) import path --
    # both populate the same sys.modules entry below.
    grf_model = sys.modules.get("bayesflow_xai.simulation.grf.grf_model")
    if grf_model is not None:
        grf_model.rng = np.random.default_rng(seed)

    grf_generative_model = sys.modules.get("bayesflow_xai.simulation.grf.grf_generative_model")
    if grf_generative_model is not None:
        grf_generative_model.rng = np.random.default_rng(seed)


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    population_size: float = 83e6
    # Increase to 30 or 60 if mu and D need a longer outbreak window to identify cleanly.
    horizon_days: int = int(os.getenv("SIR_HORIZON_DAYS", "60"))

    n_train_sims: int = 6000
    n_val_sims: int = 300
    epochs: int = 100
    batch_size: int = 64

    n_diag_datasets: int = 300
    n_diag_samples: int = 1000

    xai_target_param: str = "lambd"  # best-recovered parameter, per tutorial diagnostics
    xai_n_latent_sims: int = 2000
    xai_n_ig_samples: int = 200
    xai_surrogate_n_sims: int = 6000
    xai_transformer_n_sims: int = 4000

    outputs_dir: str = "outputs"  # .npz results, logs, models
    # Scripts point this at outputs/figures/<simulator> (see
    # figures_dir_for) so each simulator's figures get their own folder.
    figures_dir: str = "outputs/figures"
    models_dir: str = "outputs/models"
    logs_dir: str = "outputs/logs"


CONFIG = Config()
PARAM_NAMES = ["lambd", "mu", "D", "I0", "psi"]


def figures_dir_for(simulator: str) -> str:
    """outputs/figures/<simulator>: one figure folder per simulator
    (sir, lotka_volterra, grf, gravitational_wave)."""
    return os.path.join(CONFIG.outputs_dir, "figures", simulator)


def use_simulator_figures_dir(simulator: str) -> str:
    """Points CONFIG.figures_dir (and so every show_and_save call) at
    outputs/figures/<simulator>. CONFIG is frozen; this is the one
    sanctioned way to change it, called once at the top of a script."""
    path = figures_dir_for(simulator)
    object.__setattr__(CONFIG, "figures_dir", path)
    return path
