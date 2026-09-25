# bayesflow-xai

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
| `src/bayesflow_xai/registry.py` + `registrations.py` | plugin registry: `SIMULATORS`, `SUMMARY_NETWORKS`, `INFERENCE_NETWORKS`, `XAI_METHODS`, all selectable by name |
| `src/bayesflow_xai/simulation/` | 3 simulator subpackages (`sir/`, `lotka_volterra/`, `grf/`) + shared tensor-dataset builders |
| `src/bayesflow_xai/training/` | BayesFlow `BasicWorkflow` build/train (posterior inference path) |
| `src/bayesflow_xai/diagnostics/` | BayesFlow health-check plots: loss, parameter recovery, SBC calibration |
| `src/bayesflow_xai/methods/` | the 4 XAI methods + their surrogate models (per-simulator and registry-generic) |
| `src/bayesflow_xai/utils/` | `CONFIG`, `DEVICE`, `set_seed()`, `show_and_save()` |
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

`src/bayesflow_xai/utils/config.py` sets `CUDA_VISIBLE_DEVICES`/`KERAS_BACKEND`
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
| Add your own simulator / XAI method | see `src/bayesflow_xai/registry.py`'s docstring, or the "Docs / Extend" section of `outputs/index.html` |

Every script writes figures to `outputs/figures/` and prints stats to
stdout.

## Performance

Wall-clock times actually measured on this project's hardware (1x NVIDIA
RTX 2080 Ti) -- not estimates. The BasicWorkflow rows use BayesFlow's own
`Training completed in ...` log line; the Lotka-Volterra row is the whole
script's wall-clock time (`/usr/bin/time`).

| pipeline | hardware | measured time |
|---|---|---|
| SIR `BasicWorkflow` training (100 epochs, 6000 sims) | 1x RTX 2080 Ti | 8.87 min |
| GRF parameter-inference training (`BasicWorkflow`, 20 epochs) | 1x RTX 2080 Ti | 44.42 min |
| GRF diffusion model training (`ResidualUViT`, 20 epochs) | 1x RTX 2080 Ti | 1.48 hours (~89 min) |
| Lotka-Volterra channel/time importance stats (200 samples) | 1x RTX 2080 Ti | 24.5 sec |

Sources: `outputs/run_grf_tutorial_31.log` and `_32.log` for the two GRF
rows; SIR and Lotka-Volterra timed directly for this table by running
`scripts/run_diagnostics.py` and `scripts/run_lv_xai.py --target theta0
--stats` end to end.

## Outputs

`outputs/figures/` file names encode which script produced them:

| pattern | produced by |
|---|---|
| `workflow_<simulator>_<xai_method>_*.png` | registry-driven runs: `scripts/run_workflow.py --config ...` |
| `xai_NN_*.png` / `xai_lv_NN_*.png` | the original per-pipeline scripts (`run_xai.py`, `run_lv_xai.py`) that predate the registry |
| `grf_tutorial/NN_*.png` | the BayesFlow GRF tutorial (sections 3.1/3.2), reproduced verbatim |
| `NN_*.png` at the root (`01_loss_trajectory.png`, ...) | BayesFlow diagnostics, `run_diagnostics.py` |
| `saliency_samples/sample_NNN.png` | one Integrated Gradients plot per validation sample, `run_xai.py` |

`outputs/models/` holds saved network weights (currently just
`transformer_summary_net.pt`). `outputs/logs/` is where stdout from long
runs is meant to be captured, though in practice most of this repo's own
logs were saved as `outputs/*.log` (redirected `nohup` output) rather
than inside that subfolder.

## License & Citation

MIT licensed -- see [`LICENSE`](LICENSE).

This started as a research internship project. If it's useful to you, a
mention is appreciated but not required:

```
Feriha Bircan, "bayesflow-xai: Explainable AI for BayesFlow simulation-based inference" (2026).
https://github.com/ferihabircan/bayesflow-xai
```

Questions / contact: ferihabircan4@gmail.com
