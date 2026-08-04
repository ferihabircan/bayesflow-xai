# sir-xai

XAI analysis for a BayesFlow SIR-like posterior estimation model:
BayesFlow health diagnostics (loss, parameter recovery, SBC calibration)
plus three XAI methods (latent-space UMAP/t-SNE, Integrated Gradients via
Captum, attention rollout on a transformer summary net).

## Structure

    src/sir_xai/
      simulation/   prior, ODE simulator, adapter        (no ML deps)
      training/     BasicWorkflow build/train              (bayesflow, keras)
      diagnostics/  loss / recovery / calibration plots     (bayesflow.diagnostics)
      xai/          latent space, integrated gradients,      (torch, captum, umap)
                    attention rollout (+ surrogate nets)
      utils/        shared plotting/config helpers
    scripts/        thin CLI entry points, one per pipeline stage
    tests/          unit tests per module
    configs/        run configuration (yaml)
    outputs/        figures / trained models / logs (gitignored)

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # sets KERAS_BACKEND=torch and CUDA_VISIBLE_DEVICES=7

`src/sir_xai/utils/config.py` sets `CUDA_VISIBLE_DEVICES` and
`KERAS_BACKEND=torch` at import time (before torch/keras/bayesflow load),
picks `DEVICE = cuda` if available else `cpu`, and prints which GPU is
active. Every training loop (`training/workflow.py`, `xai/surrogate_models.py`,
`xai/attention_rollout.py`) moves its model and tensors to `DEVICE`. To pin a
different GPU, change `CUDA_VISIBLE_DEVICES` in `.env` before running.

## Run

    python scripts/run_diagnostics.py       # train workflow + 3 BayesFlow diagnostic plots
    python scripts/run_xai.py --target lambd # latent space + IG + attention rollout
    python scripts/run_all.py                # both, in sequence

Each script writes figures to `outputs/figures/` and prints stats to stdout
(and `outputs/logs/`).
