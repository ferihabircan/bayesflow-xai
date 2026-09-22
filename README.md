# xai

Explainable AI for BayesFlow simulation-based inference. Three simulators
(SIR epidemic model, a Lotka-Volterra-style example, Gaussian Random
Field) trained with BayesFlow, explained with four XAI methods
(Integrated Gradients, Saliency Maps, Attention Rollout,
Masking/Occlusion), all wired through one plugin registry so a new
simulator or XAI method is a decorator away.

Full docs (install/usage/models/XAI methods/results, with figures): open
[`outputs/index.html`](outputs/index.html) in a browser.

## Components

| folder | what it does |
|---|---|
| `src/xai/registry.py` + `registrations.py` | plugin registry: `SIMULATORS`, `SUMMARY_NETWORKS`, `INFERENCE_NETWORKS`, `XAI_METHODS`, all selectable by name |
| `src/xai/simulation/` | 3 simulator subpackages (`sir/`, `lotka_volterra/`, `grf/`) + shared tensor-dataset builders |
| `src/xai/training/` | BayesFlow `BasicWorkflow` build/train (posterior inference path) |
| `src/xai/diagnostics/` | BayesFlow health-check plots: loss, parameter recovery, SBC calibration |
| `src/xai/methods/` | the 4 XAI methods + their surrogate models (per-simulator and registry-generic) |
| `src/xai/utils/` | `CONFIG`, `DEVICE`, `set_seed()`, `show_and_save()` |
| `scripts/` | CLI entry points -- `run_workflow.py` is the general one |
| `configs/` | `workflow_example.yaml` drives `run_workflow.py` |
| `tests/` | unit tests, one file per module |
| `outputs/` | figures / models / logs (gitignored) + the docs page |

Each folder above has its own short README with more detail.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # or: pip install -r requirements.txt
cp .env.example .env                # sets KERAS_BACKEND, CUDA_VISIBLE_DEVICES
```

`src/xai/utils/config.py` sets `CUDA_VISIBLE_DEVICES`/`KERAS_BACKEND`
and picks `DEVICE` at import time -- check its default GPU index matches
an idle GPU before running on a shared machine (this repo's `.env` isn't
auto-loaded by any code here; the default lives directly in `config.py`).

## Choose a workflow

| I want to... | run |
|---|---|
| Try any simulator x XAI method combination | `python scripts/run_workflow.py --config configs/workflow_example.yaml` |
| Train the SIR posterior + see BayesFlow diagnostics | `python scripts/run_diagnostics.py` |
| SIR: latent space + Integrated Gradients + Attention Rollout | `python scripts/run_xai.py --target lambd` |
| Lotka-Volterra: channel/time importance stats | `python scripts/run_lv_xai.py --target theta0 --stats` |
| GRF: pixel-level Integrated Gradients | `python scripts/run_grf_xai.py --target alpha` |
| Everything for SIR in one run | `python scripts/run_all.py` |
| Add your own simulator / XAI method | see `src/xai/registry.py`'s docstring, or the "Docs / Extend" section of `outputs/index.html` |

Every script writes figures to `outputs/figures/` and prints stats to
stdout.
