# bayesflow-xai

Explainable AI for BayesFlow simulation-based inference. Four simulators
(SIR epidemic model, a Lotka-Volterra-style toy signal, a Gaussian Random
Field and a gravitational-wave benchmark) are paired with neural summary
networks and explained with four XAI methods (Integrated Gradients,
Saliency Maps, Attention Rollout, Masking/Occlusion). Everything is wired
through one plugin registry, so a new simulator or XAI method is a
decorator away.

- **Project page:** <https://ferihabircan.github.io/bayesflow-xai/> (source: [`docs/index.html`](docs/index.html))
- **Paper:** [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source: [`paper/paper.tex`](paper/paper.tex), build with `tectonic paper/paper.tex`)
- **Gravitational-wave findings:** [`outputs/gravitational_wave/xai_findings_summary.md`](outputs/gravitational_wave/xai_findings_summary.md)

## What is explained, exactly

Two model families live in this repo, and it matters which one an XAI
figure is about:

1. **BayesFlow posterior estimators** (`training/`, SIR and GRF only): a `BasicWorkflow`
   with a summary network and a coupling-flow inference network, trained
   on simulated data and checked with loss / parameter-recovery / SBC
   diagnostics. For SIR the network sees the noisy reported-case series.
   The latent-space (UMAP / t-SNE) analysis is run on this summary network.
2. **Surrogate regressors** (`methods/surrogate_models.py`,
   `methods/generic_surrogate.py`): a plain-PyTorch summary network of the
   same family (GRU, CNN, Transformer, or the GW PaperEmbedding CNN) plus a
   linear head, trained with MSE to predict the parameters directly. The
   gradient-based methods (Integrated Gradients, Saliency), Masking and
   Attention Rollout are computed on these surrogates, because Captum needs
   a plain `torch.nn.Module` with a scalar output. For SIR the surrogates
   see the full S/I/R trajectories rather than the case counts.

So the attribution maps explain *what a summary network of that
architecture learns to use from the data*, not the full normalizing-flow
posterior.

## Components

| folder | what it does |
|---|---|
| `src/bayesflow_xai/registry.py` + `registrations.py` | plugin registry: `SIMULATORS`, `SUMMARY_NETWORKS`, `INFERENCE_NETWORKS`, `XAI_METHODS`, all selectable by name |
| `src/bayesflow_xai/simulation/` | 4 simulator subpackages (`sir/`, `lotka_volterra/`, `grf/`, `gravitational_wave/`) + shared tensor-dataset builders. The flat `sir_model.py`, `grf_model.py`, ... files are backward-compatibility shims |
| `src/bayesflow_xai/training/` | BayesFlow `BasicWorkflow` build/train (SIR, GRF parameter inference, GRF diffusion) |
| `src/bayesflow_xai/diagnostics/` | BayesFlow health-check plots: loss, parameter recovery, SBC calibration |
| `src/bayesflow_xai/methods/` | the 4 XAI methods, latent-space projection, and the surrogate models |
| `src/bayesflow_xai/utils/` | `CONFIG`, `DEVICE`, `set_seed()`, `show_and_save()` |
| `scripts/` | CLI entry points; `run_workflow.py` is the general one |
| `configs/` | YAML configs for `run_workflow.py` and `run_gw_xai_comparison.py` |
| `tests/` | unit tests (`pytest`) |
| `outputs/` | generated figures and findings; `outputs/figures/<simulator>/` holds the figures shown on the project page |
| `docs/` | GitHub Pages project page (+ a copy of the figures it shows) |
| `paper/` | paper-style write-up (LaTeX source + PDF; `docs/paper.pdf` is a copy for the project page) |

Each folder above has its own short README with more detail.

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # or: pip install -r requirements.txt
pip install -e ".[gw]"              # optional: PyCBC, only for the gravitational-wave simulator
pip install -e ".[dev]"             # optional: pytest, black, ruff
```

`src/bayesflow_xai/utils/config.py` sets `KERAS_BACKEND=torch` and a default
`CUDA_VISIBLE_DEVICES` **at import time** (currently `"8"`, the GPU used on
the lab server). On any other machine, set it yourself before running, or
the GPU will be hidden and everything silently runs on the CPU:

```bash
export CUDA_VISIBLE_DEVICES=0       # or "" to force CPU
```

`.env.example` documents the two variables, but no code loads `.env`
automatically.

Run the tests with `pytest` (the gravitational-wave tests are skipped when
PyCBC is not installed).

## Choose a workflow

| I want to... | run |
|---|---|
| Try any simulator x XAI method combination | `python scripts/run_workflow.py --config configs/workflow_example.yaml` |
| Quick smoke run of the same | `python scripts/run_workflow.py --config configs/workflow_example.yaml --n_sims 200 --epochs 2` |
| Train the SIR posterior + see BayesFlow diagnostics | `python scripts/run_diagnostics.py` |
| SIR: latent space + Integrated Gradients + Attention Rollout | `python scripts/run_xai.py --target lambd` (add `--stats` for channel/time statistics) |
| SIR: IG vs Saliency vs Masking on one surrogate | `python scripts/run_sir_method_comparison.py --target lambd` |
| Lotka-Volterra: channel/time importance stats | `python scripts/run_lv_xai.py --target theta0 --stats` |
| GRF: pixel-level Integrated Gradients | `python scripts/run_grf_xai.py --target alpha` |
| GRF: reproduce the BayesFlow spatial-data tutorial | `python scripts/run_grf_tutorial_31.py` / `run_grf_tutorial_32.py` (`--smoke-test` for a quick check) |
| Gravitational wave: IG with min-max vs z-score inputs | `python scripts/run_workflow.py --config configs/gravitational_wave_workflow.yaml` (and `..._zscore_workflow.yaml`), then `python scripts/plot_gw_ig_time.py` |
| Gravitational wave: IG vs Saliency vs Masking on one model | `python scripts/run_gw_xai_comparison.py --config configs/gravitational_wave_zscore_workflow.yaml` |
| Gravitational wave: look at raw simulated strain | `python scripts/plot_gw_samples.py` |
| Everything for SIR in one run | `python scripts/run_all.py` |
| Add your own simulator / XAI method | see the docstring of `src/bayesflow_xai/registry.py` |

Every script writes figures to `outputs/figures/<simulator>/` and prints
stats to stdout. Note that this overwrites the committed figures.

## Performance

Wall-clock times measured on the project's hardware (1x NVIDIA RTX 2080
Ti), not estimates. The BasicWorkflow rows use BayesFlow's own
`Training completed in ...` log line; the Lotka-Volterra row is the whole
script's wall-clock time (`/usr/bin/time`).

| pipeline | hardware | measured time |
|---|---|---|
| SIR `BasicWorkflow` training (100 epochs, 6000 sims) | 1x RTX 2080 Ti | 8.87 min |
| GRF parameter-inference training (`BasicWorkflow`, 20 epochs) | 1x RTX 2080 Ti | 44.42 min |
| GRF diffusion model training (`ResidualUViT`, 20 epochs) | 1x RTX 2080 Ti | 1.48 hours (~89 min) |
| Lotka-Volterra channel/time importance stats (200 samples) | 1x RTX 2080 Ti | 24.5 sec |

The small SIR surrogate / XAI pipelines also run on a laptop CPU in a few
minutes.

## Outputs

`outputs/figures/<simulator>/` file names encode which script produced them:

| pattern | produced by |
|---|---|
| `workflow_<simulator>_<xai_method>_*.png` | registry-driven runs: `scripts/run_workflow.py --config ...` |
| `xai_NN_*.png` / `xai_lv_NN_*.png` | the original per-pipeline scripts (`run_xai.py`, `run_lv_xai.py`) |
| `grf/grf_tutorial/NN_*.png` | the BayesFlow GRF tutorial (sections 3.1/3.2) |
| `sir/01_loss_trajectory.png`, `02_...`, `03_...` | BayesFlow diagnostics, `run_diagnostics.py` |
| `sir/saliency_samples/sample_NNN.png` | one Integrated Gradients plot per validation sample, `run_xai.py --all-samples` |
| `gravitational_wave/*.png` | `run_workflow.py` with the GW configs, `run_gw_xai_comparison.py`, `plot_gw_*.py` |

Model checkpoints (`outputs/models/`) and `.npz` attribution files are
not committed.

## License & Citation

MIT licensed, see [`LICENSE`](LICENSE). The vendored gravitational-wave
simulator keeps its own MIT / BSD-3-Clause notices (see
`src/bayesflow_xai/simulation/gravitational_wave/README.md`).

This started as a research internship project. If it is useful to you, a
mention is appreciated:

```
Feriha Bircan, "bayesflow-xai: Explainable AI for BayesFlow simulation-based inference" (2026).
https://github.com/ferihabircan/bayesflow-xai
```

Questions / contact: ferihabircan4@gmail.com
