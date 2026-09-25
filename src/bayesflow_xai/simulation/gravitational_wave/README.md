# Gravitational wave (PyCBC, IMRPhenomPv2)

> **Not the notebook's simulator.** `4_1_grav_waves.ipynb` uses
> `GravitationalWaveBenchmarkSimulator` from an external package
> (`paper.fig8_grav_wave_npe...`) plus pre-generated data
> (`data/gws-train.h5`, ~2.5 GB) -- neither is available here. This
> subpackage is **not identical** to it. It is a physically reasonable
> alternative that uses PyCBC's own waveform generator with the parameters
> of `config_file_pycbcmaster.ini`.

## What it does

`simulator.py` does the following for each draw:

1. It draws the targets from the notebook's prior: `mass1 ~ U(40, 80)` and
   `mass_ratio ~ U(0.25, 0.99)`, with `mass2 = mass_ratio · mass1`. The
   notebook overrides the `.ini`'s 10–80 mass priors this way. The other
   `[variable_params]` come from the `.ini`'s `[prior-*]` sections.
2. It generates `h_plus(f)` and `h_cross(f)` with
   `pycbc.waveform.get_fd_waveform`, using `IMRPhenomPv2`,
   `f_lower=f_ref=20 Hz` and `delta_f=1/128 Hz`. The output covers a 128 s
   window at 2048 Hz.
3. It projects the signal onto the real H1 and L1 detectors. This is the
   **full antenna pattern**, using the `.ini`'s `[static_params]`:
   `ra=3.44615914`, `dec=-0.40808407` and `polarization=0`. Each channel is
   `F+·h_plus + Fx·h_cross`, shifted by that detector's light-travel delay.
   F+, Fx and the delay come from PyCBC's
   `Detector(name).antenna_pattern(ra, dec, polarization, tc)` and
   `Detector(name).time_delay_from_earth_center(ra, dec, tc)`, where `tc` is
   the geocentric GPS time.
4. It whitens with PyCBC's `aLIGOZeroDetHighPower` PSD and adds Gaussian
   noise drawn from that PSD. Whitened noise has unit variance per sample.
   Without noise, `sum(x**2)` equals the optimal SNR² (checked against
   `pycbc.filter.sigma`). The notebook's simulator instead whitens real
   detector noise (`noise_interval_width`, `whitening_segment_duration`).

## Priors (`configs/gravitational_wave/config_file_pycbcmaster.ini`)

| param | distribution | range | source |
|---|---|---|---|
| mass1 | uniform | 40–80 Msun | notebook `BoxUniform` |
| mass_ratio (= mass2/mass1) | uniform | 0.25–0.99 | notebook `BoxUniform` |
| spin1z | uniform | −0.9 to −0.8 | .ini |
| spin2z | uniform | 0.8 to 0.9 | .ini |
| inclination | sin_angle | 0–π | .ini |
| distance | uniform_radius | 1–1.1 Mpc | .ini |
| tc | uniform | GPS 1187008882.4–1187008882.5 (geocentric) | .ini |
| coa_phase | uniform_angle | 0–2π | .ini |

At 1–1.1 Mpc the signal is ~10²–10⁴ times the noise level, so the data are
effectively noise-free chirps.

## Registry / XAI dataset

The simulator is registered as `gravitational_wave`, with
`input_kind="timeseries"`, channels H1 and L1, and targets `mass1` and
`mass_ratio`.

`build_gw_tensor_dataset(n)` crops each series to the `.ini`'s
`seconds_before_event=3.5` and `seconds_after_event=0.5` around the H1
event time (the geocentric `tc` plus H1's light-travel delay). At 2048 Hz,
without decimation, this gives `X` with shape `(n, 8192, 2)`. `X` is then standardised with one global (loc, scale), as
the notebook's `gws-train.h5` is. `y` has shape `(n, 2)`. Use `simulate(n)`
to get the full 128 s series.

## Summary network: `gw_paper_cnn`

`embedding.py` holds `PaperEmbedding` and `init_weights`, copied verbatim
from notebook cell `7b3b7d22`. The architecture is a 1×1 Conv1d, then 13
dilated Conv1d layers (kernel 2, dilation 2⁰…2¹²) with SELU, then Flatten.
The unpadded stack shortens the sequence by 2¹³−1, so 8192 samples map to
1 sample, and the output is `(batch, 16)`.

`GWPaperCNNSummaryNet` is the project-side adapter. It transposes the
registry's `(B, T, C)` input to `(B, C, T)`, applies `init_weights`, and
exposes `summary_dim=16`. It is registered as `gw_paper_cnn`.

Speed: about 50 ms per simulation on one CPU core.
