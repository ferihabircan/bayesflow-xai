# Gravitational wave (PyCBC, IMRPhenomPv2)

This simulator is the notebook's (`4_1_grav_waves.ipynb`) **original
`GravitationalWaveBenchmarkSimulator`**, taken unchanged from
[sbi-dev/sbi-practical-guide](https://github.com/sbi-dev/sbi-practical-guide)
(commit `18852e1`, `paper/fig8_grav_wave/workflow/scripts/external/`, MIT
license). The notebook imports it as
`paper.fig8_grav_wave_npe.smk.workflow.scripts.external.simulator`, which is
an older layout of the same directory. The class itself is Joeri Hermans'
`hypothesis` GW benchmark (BSD-3-Clause), with ra/dec/polarization fixed by
the `.ini`.

| file | origin |
|---|---|
| `sbi_practical_guide/simulator.py`, `base.py`, `config_file_pycbcmaster.ini`, `config_file.ini` | verbatim copies from sbi-practical-guide |
| `sbi_practical_guide/__init__.py` | registry adapter that reproduces sbi-practical-guide's `script-generate-gws.py` and `gws-split-denovo.py` |
| `embedding.py` | `PaperEmbedding` + `init_weights` verbatim from the notebook; `GWPaperCNNSummaryNet` adapter |

## Key Finding: IG exposes a non-physical whitening artefact in the original simulator

**Integrated Gradients showed that a model trained on the original
`GravitationalWaveBenchmarkSimulator` focuses on a non-physical ~20 Hz
artefact caused by whitening, not on the merger.** The simulator is
deliberately left unmodified. The artefact is a result of the XAI analysis,
not a bug to patch away.

What IG showed (5000 simulations, 40 epochs, identical data for both
normalisations, IG with the zero-strain baseline and n_steps=200):

| input normalisation | val MSE | IG peak | samples peaking within ±50 ms of merger | IG share in last 0.2 s |
|---|---|---|---|---|
| min-max (original, `configs/gravitational_wave_workflow.yaml`) | 9.27 | **−212 ms** | **0%** | 20% |
| z-score (`configs/gravitational_wave_zscore_workflow.yaml`) | 1.41 | −4.9 ms | 100% | 40% |

With the original min-max inputs, IG attribution sits in sharp blocks at
fixed times (≈ −0.7 s, −0.2 s, +0.3 s) where the merger contributes almost
nothing. With z-score inputs the model does find the merger, but a large
share of attribution still falls on the artefact region.

Where the artefact comes from (traced stage by stage for one H1
simulation, mass 40/38; see `scripts/plot_gw_samples.py` →
`outputs/figures/gravitational_wave/gw_samples.png`):

1. PyCBC's `get_td_waveform` starts the waveform below `f_lower = 20 Hz`
   (11 s before merger). The ~15–20 Hz early inspiral is physical.
2. The noise comes from `aLIGOZeroDetHighPower(low_freq_cutoff=20)`, so it
   has **no power below 20 Hz**. At 1–1.1 Mpc the signal is far above the
   noise anyway.
3. `strain.whiten(...)` estimates the PSD from the data itself, so it
   **amplifies** the sub-20 Hz inspiral instead of suppressing it: std at
   −1 s goes from 33 (whitened noise alone) to 2.5·10⁵.
4. `highpass_fir(20 Hz, order=512)` only reduces it ~6×, because the content
   sits right at the cutoff.

The result is a nearly monochromatic ~20 Hz oscillation with a bell-shaped
envelope from −1.5 s to +0.5 s (it even continues **after** the merger),
which dominates every sample. The physical chirp (a frequency sweep up to
~300 Hz) is visible only in the last ~50 ms. Min-max scaling spends almost
the whole [0, 1] range on this artefact, which is why that model learned it
instead of the merger.

Full findings, including IG vs Saliency Map vs Masking on one trained model:
[`outputs/gravitational_wave/xai_findings_summary.md`](../../../../outputs/gravitational_wave/xai_findings_summary.md).

## What the simulator does

For each `(mass1, mass2)` input, `_simulate_gw` in `simulator.py`:

1. Draws the other parameters from `config_file_pycbcmaster.ini`'s
   `[variable_params]`: inclination, distance 1–1.1 Mpc, tc, spins,
   coa_phase. ra, dec and polarization are fixed `[static_params]`.
2. Generates `h_plus` and `h_cross` with `get_td_waveform` (IMRPhenomPv2,
   2048 Hz, 128 s buffer).
3. Projects them onto H1 and L1 (antenna pattern evaluated at
   `t_gps=100`) and shifts L1 by its light-travel delay from H1.
4. Adds 32 s of Gaussian noise from `aLIGOZeroDetHighPower`.
5. Applies `whiten()` (4 s segments) and a 20 Hz `highpass_fir`.
6. Cuts the window from 3.5 s before to 0.5 s after the H1 event, giving
   2 × 8192 samples.

## Registry / XAI dataset

The simulator is registered as `gravitational_wave`, with
`input_kind="timeseries"`, channels H1 and L1, and targets `mass1` and
`mass_ratio`.

`build_gw_tensor_dataset(n, seed=0, normalization="minmax")` follows the
original `script-generate-gws.py`:

- Prior: `mass1 ~ U(40, 80)`, `mass_ratio ~ U(0.25, 0.99)`,
  `mass2 = mass_ratio · mass1`.
- Simulation runs in parallel, with a separate numpy seed per chunk so that
  forked workers don't repeat the same noise.
- Normalisation follows `gws-split-denovo.py`: global min-max (the
  default), or `normalization="zscore"`.

It returns `X` with shape `(n, 8192, 2)` and `y` with shape `(n, 2)`.
Simulation takes about 0.3–0.5 s per sample on one CPU core.

## Summary network: `gw_paper_cnn`

`embedding.py` holds `PaperEmbedding` and `init_weights`, copied verbatim
from notebook cell `7b3b7d22`. The architecture is a 1×1 Conv1d, then 13
dilated Conv1d layers (kernel 2, dilation 2⁰…2¹²) with SELU, then Flatten.
The unpadded stack shortens the sequence by 2¹³−1, so 8192 samples map to
1 sample, and the output is `(batch, 16)`.

`GWPaperCNNSummaryNet` is the project-side adapter. It transposes the
registry's `(B, T, C)` input to `(B, C, T)`, applies `init_weights`, and
exposes `summary_dim=16`. It is registered as `gw_paper_cnn`.
