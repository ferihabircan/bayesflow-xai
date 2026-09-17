"""Central run configuration. Import this before anything else that touches
keras/bayesflow/torch, since CUDA_VISIBLE_DEVICES and KERAS_BACKEND must be
set before those libraries are imported."""

import os
import random

# =============================================================
# GPU & BACKEND SETTINGS
# =============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
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
    `lotka_volterra_model.sample_fn`), torch/CUDA, and `sir_model.RNG` --
    a standalone `np.random.Generator` instance that `np.random.seed()`
    does NOT reach, so it must be reseeded explicitly here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Lazy import to avoid a circular import (sir_model imports CONFIG from
    # this module at module load time).
    from sir_xai.simulation import sir_model

    sir_model.RNG = np.random.default_rng(seed)


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

    figures_dir: str = "outputs/figures"
    models_dir: str = "outputs/models"
    logs_dir: str = "outputs/logs"


CONFIG = Config()
PARAM_NAMES = ["lambd", "mu", "D", "I0", "psi"]
