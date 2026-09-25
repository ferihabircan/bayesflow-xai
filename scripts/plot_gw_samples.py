#!/usr/bin/env python
"""Visual sanity check of the vendored original gravitational-wave simulator
(gravitational_wave, from sbi-dev/sbi-practical-guide): simulates a few (mass1, mass_ratio) pairs and
plots raw H1/L1 whitened strain, a zoom on the merger and a spectrogram, and
prints chirp checks (frequency rising towards merger, amplitude peak near
t = 0). CPU only and a handful of simulations, so it can run next to a
training job:

    python scripts/plot_gw_samples.py
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from bayesflow_xai.simulation.gravitational_wave import CHANNEL_NAMES, SAMPLING_RATE, SECONDS_BEFORE_EVENT
from bayesflow_xai.simulation.gravitational_wave.sbi_practical_guide import _simulate_chunk

# (mass1, mass_ratio) spanning the original prior U(40, 80) x U(0.25, 0.99).
THETAS = np.array([[40.0, 0.95], [55.0, 0.60], [70.0, 0.40], [80.0, 0.25]])
# Windows (s, relative to the H1 merger) for the zero-crossing frequency estimate.
FREQ_WINDOWS = [(-1.0, -0.8), (-0.5, -0.4), (-0.2, -0.15), (-0.08, -0.05), (-0.04, -0.01)]


def _zero_crossing_freq(x, t, lo, hi):
    seg = x[(t >= lo) & (t < hi)]
    return np.count_nonzero(np.diff(np.signbit(seg))) / 2 / (hi - lo)


def _envelope_peak(x, t, width=41):
    env = np.convolve(np.abs(x), np.ones(width) / width, mode="same")
    return t[np.argmax(env)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="outputs/figures/gravitational_wave/gw_samples.png")
    args = parser.parse_args()

    xs = _simulate_chunk((THETAS, args.seed))  # (n, 2, 8192) raw whitened strain
    t = np.arange(xs.shape[-1]) / SAMPLING_RATE - SECONDS_BEFORE_EVENT

    fig, axes = plt.subplots(len(THETAS), 3, figsize=(16, 3.2 * len(THETAS)), gridspec_kw={"width_ratios": [2, 1.3, 1.3]})
    colors = ["#1f77b4", "#ff7f0e"]
    for i, ((m1, q), x) in enumerate(zip(THETAS, xs)):
        label = f"mass1={m1:.0f}, mass2={m1 * q:.1f} (q={q:.2f})"
        print(f"== sample {i}: {label}")
        for c, name in enumerate(CHANNEL_NAMES):
            freqs = [_zero_crossing_freq(x[c], t, lo, hi) for lo, hi in FREQ_WINDOWS]
            rising = all(b >= a for a, b in zip(freqs, freqs[1:]))
            print(
                f"  {name}: |x| peak at {_envelope_peak(x[c], t) * 1e3:+.1f} ms, max|x|={np.abs(x[c]).max():.0f}, "
                f"std early (<-2 s)={x[c][t < -2].std():.2f}, std after +0.1 s={x[c][t > 0.1].std():.2f}"
            )
            print(
                "      zero-crossing freq [Hz] in "
                + ", ".join(f"[{lo:+.2f},{hi:+.2f}): {f:.0f}" for (lo, hi), f in zip(FREQ_WINDOWS, freqs))
                + f"  -> {'rising' if rising else 'NOT monotonic'}"
            )

        ax_full, ax_zoom, ax_spec = axes[i]
        for c, name in enumerate(CHANNEL_NAMES):
            ax_full.plot(t, x[c], color=colors[c], lw=0.5, alpha=0.8, label=name)
            m = (t >= -0.3) & (t <= 0.1)
            ax_zoom.plot(t[m], x[c][m], color=colors[c], lw=0.8, label=name)
        for ax in (ax_full, ax_zoom):
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_ylabel("whitened strain")
        ax_full.set_xlim(t[0], t[-1])
        ax_full.set_title(f"{label}: full 4 s window")
        ax_full.legend(loc="upper left", fontsize=8)
        ax_zoom.set_xlim(-0.3, 0.1)
        ax_zoom.set_title("zoom on merger")

        f, ts, S = spectrogram(x[0], fs=SAMPLING_RATE, nperseg=128, noverlap=112)
        ts = ts + t[0]
        fm = f <= 512
        ax_spec.pcolormesh(ts, f[fm], np.log10(S[fm] + 1e-12), shading="auto", cmap="viridis")
        ax_spec.axvline(0, color="w", ls="--", lw=0.8)
        ax_spec.set_xlim(-1.0, 0.3)
        ax_spec.set_ylabel("frequency [Hz]")
        ax_spec.set_title("H1 spectrogram (log power)")
    for ax in axes[-1]:
        ax.set_xlabel("time relative to H1 merger [s]")

    fig.suptitle("gravitational_wave (original GravitationalWaveBenchmarkSimulator, sbi-practical-guide): raw whitened strain", y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
