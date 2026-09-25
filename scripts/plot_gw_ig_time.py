#!/usr/bin/env python
"""Time-resolved Integrated Gradients plot for the gravitational-wave
workflows: which part of the 4 s window (inspiral / merger / ringdown) and
which detector (H1 / L1) the surrogate's prediction relies on.

Reads the .npz written by `integrated_gradients_generic(save_attributions=True)`
(see configs/gravitational_wave*_workflow.yaml), so run the workflow first:

    python scripts/run_workflow.py --config configs/gravitational_wave_workflow.yaml
    python scripts/plot_gw_ig_time.py --target mass1

Pass several --npz/--label pairs to compare simulators, one row each:

    python scripts/plot_gw_ig_time.py \\
        --npz outputs/workflow_gravitational_wave_integrated_gradients_integrated_gradients.npz --label ours \\
        --npz outputs/workflow_gravitational_wave_guide_integrated_gradients_integrated_gradients.npz --label original \\
        --out outputs/figures/gw_ig_time_comparison.png

Both simulators align every sample so the H1 merger is at t = 0
(seconds_before_event = 3.5 s into the window), which makes attributions
directly averageable across samples.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesflow_xai.simulation.gravitational_wave.simulator import (
    CHANNEL_NAMES,
    SAMPLING_RATE,
    SECONDS_BEFORE_EVENT,
)

DEFAULT_NPZ = "outputs/workflow_gravitational_wave_integrated_gradients_integrated_gradients.npz"

# Time bins relative to the H1 merger (s).
SEGMENTS = [
    ("early inspiral", -3.5, -1.0),
    ("late inspiral", -1.0, -0.2),
    ("pre-merger", -0.2, -0.03),
    ("merger", -0.03, 0.01),
    ("ringdown", 0.01, 0.05),
    ("post-ringdown", 0.05, 0.5),
]
COLORS = ["#1f77b4", "#ff7f0e"]


def _smooth(x, width):
    return np.convolve(x, np.ones(width) / width, mode="same")


def summarize(path, label):
    """Loads one attribution .npz and prints/returns its time/channel breakdown."""
    d = np.load(path)
    X, A = d["inputs"], d["attributions"]  # (n, T, 2)
    n, T, _ = A.shape
    t = np.arange(T) / SAMPLING_RATE - SECONDS_BEFORE_EVENT
    absA = np.abs(A)
    # |signal| relative to the baseline (the zero-strain level), so z-score
    # and min-max-normalised inputs are comparable.
    baseline = float(d["baseline_value"]) if "baseline_value" in d else 0.0
    signal_env = np.abs(X - baseline).mean(axis=(0, 2))

    total = absA.sum()
    s = {
        "label": label, "n": n, "t": t, "absA": absA, "signal_env": signal_env,
        "delta": np.abs(d["delta"]).mean(),
        "n_steps": int(d["n_steps"]) if "n_steps" in d else 50,
        "channel_share": absA.sum(axis=(0, 1)) / total,
        "signal_channel_share": ((X - baseline) ** 2).sum(axis=(0, 1)) / ((X - baseline) ** 2).sum(),
        "segments": [(name, lo, hi, absA[:, (t >= lo) & (t < hi)].sum() / total) for name, lo, hi in SEGMENTS],
        "last_0p2s": absA[:, (t >= -0.2) & (t < 0.01)].sum() / total,
    }
    s["peak_t"] = t[np.argmax(_smooth(absA.sum(axis=2).mean(axis=0), 20))]
    s["sig_peak_t"] = t[np.argmax(_smooth(signal_env, 20))]
    per_sample_peak = t[np.argmax(np.stack([_smooth(a, 20) for a in absA.sum(axis=2)]), axis=1)]
    s["median_sample_peak"] = np.median(per_sample_peak)
    s["frac_peak_near_merger"] = np.mean(np.abs(per_sample_peak) < 0.05)

    print(f"== {label}: {path}")
    print(f"n_samples={n}, T={T}, n_steps={s['n_steps']}, baseline={baseline:.4g}, mean |convergence delta|={s['delta']:.4g}")
    print("channel share of total |attribution|: " + ", ".join(f"{c}={v:.1%}" for c, v in zip(CHANNEL_NAMES, s["channel_share"])))
    print("channel share of signal energy:       " + ", ".join(f"{c}={v:.1%}" for c, v in zip(CHANNEL_NAMES, s["signal_channel_share"])))
    print(f"{'segment':15s} {'window [s]':>16s} {'share':>7s} {'per 10 ms':>10s}")
    for name, lo, hi, share in s["segments"]:
        print(f"{name:15s} [{lo:+.2f}, {hi:+.2f}) {share:7.1%} {share / ((hi - lo) / 0.01):10.2%}")
    print(f"peak of mean |attribution| (10 ms smoothing): t = {s['peak_t'] * 1e3:+.1f} ms")
    print(f"peak of mean |signal|: t = {s['sig_peak_t'] * 1e3:+.1f} ms")
    print(
        f"per-sample attribution peaks: median {s['median_sample_peak'] * 1e3:+.1f} ms, "
        f"{s['frac_peak_near_merger']:.0%} within +-50 ms of merger"
    )
    return s


def plot_row(ax_full, ax_zoom, ax_bar, s, target):
    t, absA = s["t"], s["absA"]
    for ax, (lo, hi), w in [(ax_full, (-3.5, 0.5), 41), (ax_zoom, (-0.3, 0.08), 5)]:
        m = (t >= lo) & (t <= hi)
        for c, name in enumerate(CHANNEL_NAMES):
            ax.plot(t[m], _smooth(absA[:, :, c].mean(axis=0), w)[m], color=COLORS[c], lw=1, label=f"|IG| {name}")
        ax2 = ax.twinx()
        ax2.fill_between(t[m], _smooth(s["signal_env"], w)[m], color="grey", alpha=0.2, lw=0)
        ax2.set_yticks([])
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("mean |attribution|")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    ax_full.legend(loc="upper left")
    ax_full.set_title(
        f"{s['label']}: IG over time, target={target} (n={s['n']}, n_steps={s['n_steps']}, "
        f"mean |delta|={s['delta']:.3f}; grey: mean |signal|; dashed: H1 merger)"
    )
    ax_zoom.set_xlabel("time relative to H1 merger [s]")
    ax_zoom.set_title(f"{s['label']}: zoom on merger ({s['last_0p2s']:.0%} of |IG| in last 0.2 s)")

    x = np.arange(len(CHANNEL_NAMES))
    ax_bar.bar(x - 0.2, s["channel_share"] * 100, width=0.4, color=COLORS, label="|attribution|")
    ax_bar.bar(x + 0.2, s["signal_channel_share"] * 100, width=0.4, color=COLORS, alpha=0.35, hatch="//", label="signal energy")
    ax_bar.set_xticks(x, CHANNEL_NAMES)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("% of total")
    ax_bar.set_title(f"{s['label']}: detector share")
    ax_bar.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", action="append", help="attribution .npz (repeatable)")
    parser.add_argument("--label", action="append", help="row label per --npz")
    parser.add_argument("--target", default="mass1")
    parser.add_argument("--out", default="outputs/figures/workflow_gravitational_wave_integrated_gradients_time.png")
    args = parser.parse_args()

    paths = args.npz or [DEFAULT_NPZ]
    labels = args.label or [os.path.basename(p).split("_integrated_gradients")[0] for p in paths]
    if len(labels) != len(paths):
        parser.error("give one --label per --npz")
    summaries = [summarize(p, l) for p, l in zip(paths, labels)]

    fig = plt.figure(figsize=(13, 8 * len(summaries)))
    gs = fig.add_gridspec(2 * len(summaries), 3)
    for i, s in enumerate(summaries):
        plot_row(
            fig.add_subplot(gs[2 * i, :]), fig.add_subplot(gs[2 * i + 1, :2]), fig.add_subplot(gs[2 * i + 1, 2]),
            s, args.target,
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
