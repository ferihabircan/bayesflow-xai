"""Central run configuration. Import this before anything else that touches
keras/bayesflow/torch, since CUDA_VISIBLE_DEVICES and KERAS_BACKEND must be
set before those libraries are imported."""

import os

# =============================================================
# GPU & BACKEND SETTINGS
# =============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
# Keras backend set to 'torch' for Captum / PyTorch compatibility
os.environ.setdefault("KERAS_BACKEND", "torch")

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


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    population_size: float = 83e6
    horizon_days: int = 60

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
