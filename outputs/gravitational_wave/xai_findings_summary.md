# Gravitational wave: XAI findings summary

Simulator: the notebook's original `GravitationalWaveBenchmarkSimulator`
(taken unchanged from [sbi-dev/sbi-practical-guide](https://github.com/sbi-dev/sbi-practical-guide),
registered as `gravitational_wave`). Summary network: the notebook's
PaperEmbedding CNN (`gw_paper_cnn`). Target: `mass1`. All runs use 5000
simulations, 40 epochs and a 10% validation split. XAI is computed on 64
validation samples. Time is measured relative to the H1 merger (t = 0).

## 1. Key finding: IG exposes a non-physical whitening artefact

**Integrated Gradients showed that the model trained on the original
simulator, with its original min-max input normalisation, bases its
prediction on a non-physical ~20 Hz artefact produced by whitening, not on
the merger.** We deliberately keep the simulator unmodified. This is a
result of the XAI analysis.

| input normalisation (same simulated data, data seed 0) | val MSE | IG peak | samples peaking within ±50 ms of merger | IG share in last 0.2 s |
|---|---|---|---|---|
| min-max (original) | 9.27 | **−212 ms** | **0%** | 20% |
| z-score | 1.41 | −4.9 ms | 100% | 40% |

Cause, traced stage by stage through the simulator
(`scripts/plot_gw_samples.py` → `figures/gravitational_wave/gw_samples.png`):
PyCBC starts the waveform below 20 Hz. The noise PSD
(`aLIGOZeroDetHighPower`, `low_freq_cutoff=20`) has no power there, so
`whiten()`, which estimates the PSD from the data, **amplifies** the
sub-20 Hz inspiral (std at −1 s: 33 → 2.5·10⁵). The 20 Hz FIR high-pass
only removes about 6× of it. Every sample is dominated by a ~20 Hz bell from −1.5 s to +0.5 s that
continues past the merger. The real chirp is visible only in the last
~50 ms. Figures: `figures/gravitational_wave/gw_ig_time_comparison.png`
(min-max vs z-score), `figures/gravitational_wave/gw_samples.png`.

## 2. IG vs Saliency Map vs Masking (one trained model)

`scripts/run_gw_xai_comparison.py --config configs/gravitational_wave_zscore_workflow.yaml`:
z-score inputs, data seed 42, val MSE **0.60**. The checkpoint is
`models/gwxai_gravitational_wave_zscore_gw_paper_cnn.pt`. Attention Rollout
is not applicable because PaperEmbedding is a CNN.

- **IG**: baseline = zero-strain level, n_steps=200, mean |convergence
  delta| 0.046 (vs. mean |f(x) − f(baseline)| 12.3).
- **Saliency**: |∂ prediction / ∂ input|.
- **Masking**: 32-sample (15.6 ms) windows. Each channel is masked
  separately to the zero-strain level; the value is |prediction shift|.

| | Integrated Gradients | Saliency Map | Masking |
|---|---|---|---|
| peak of mean attribution | −5.4 ms | **−3010 ms** | −10.7 ms |
| samples peaking within ±50 ms of merger | 97% | **25%** | 100% |
| share in last 0.2 s | 46% | **11%** | 41% |
| share in early inspiral (−3.5 to −1 s, ~no signal) | 7.5% | **58%** | 7.8% |
| share after merger (+0.05 to +0.5 s) | 13% | 17% | 14% |
| H1 / L1 attribution | 57 / 43 | 60 / 40 | 59 / 41 |
| H1 / L1 signal energy | 45 / 55 | 45 / 55 | 45 / 55 |
| rank correlation of time profiles | IG–Masking **0.99** | IG–Saliency 0.47 | Saliency–Masking 0.47 |

**Do they all focus on the merger?** IG and Masking do, and they agree
almost exactly (rank correlation 0.99). Saliency does not: 58% of its mass
sits in the first 2.5 s, where the whitened strain is essentially zero.
Saliency measures the model's *sensitivity* to each input position, not how
much each position's *content* contributed. IG and Masking both measure
change from the zero-strain baseline, so empty regions score ~0 by
construction. Saliency's early-window peaks fall on a regular 0.25 s grid
(512 samples, one of PaperEmbedding's dilation factors, 2⁹), which points
to the architecture rather than the data. This is an observation, not a
tested claim.

**Is the H1/L1 preference consistent?** Yes, across all three methods for
this model: about 57–60% H1, although H1 carries only 45% of the signal
energy. It is **not consistent across trainings**, though. The earlier
z-score model (data seed 0) had H1/L1 = 46/54 IG attribution, in line with
its signal energy (42/58). So the detector preference is a property of an
individual trained model, not a robust feature of the data.

**Note on run-to-run variability:** the two z-score trainings (data seeds 0
and 42, otherwise identical) reached val MSE 1.41 and 0.60. Single-run
numbers in this report should be read with that spread in mind.

Figure: `figures/gravitational_wave/gwxai_gravitational_wave_zscore_xai_comparison.png`.

## 3. Takeaways

1. The original simulator, with its original normalisation, produces a
   model that ignores the merger. IG made this visible, and the cause is a
   whitening artefact in the simulator.
2. For judging what the model uses, IG and Masking agree and are the
   trustworthy pair here. Saliency is dominated by gradient sensitivity in
   signal-free regions.
3. The H1/L1 preference changes between trainings, so it should not be
   reported as a physical result without multiple seeds.
