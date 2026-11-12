#!/usr/bin/env python
"""Integrated Gradients vs Saliency Map vs Masking on ONE trained SIR
surrogate (the same SIRGRUSummaryNet that scripts/run_xai.py explains), on
the same validation samples, so the three methods are compared on one model.

Masking is computed per channel: each (time step, channel) entry is set to 0
in turn and |prediction shift| is recorded, which gives both a channel
ranking and a time profile.

    python scripts/run_sir_method_comparison.py --target lambd
    # -> outputs/figures/sir/xai_05_method_comparison.png + printed table
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from bayesflow_xai.utils.config import CONFIG, PARAM_NAMES, use_simulator_figures_dir
from bayesflow_xai.methods.surrogate_models import train_sir_surrogate, TargetWrapper
from bayesflow_xai.utils.plotting import show_and_save

CHANNELS = ["S", "I", "R"]


def main(target: str, n_samples: int):
    from captum.attr import IntegratedGradients

    use_simulator_figures_dir("sir")
    target_idx = PARAM_NAMES.index(target)
    body, head, X_val, _ = train_sir_surrogate(n_sims=CONFIG.xai_surrogate_n_sims)
    model = TargetWrapper(body, head, target_idx)
    model.train()  # cuDNN RNN backward needs train mode; the GRU has no dropout
    x = X_val[:n_samples].clone()

    inputs = x.clone().requires_grad_(True)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        ig = IntegratedGradients(model).attribute(inputs, torch.zeros_like(inputs))
        out = model(inputs)
        (grad,) = torch.autograd.grad(out.sum(), inputs)
    ig = ig.detach().abs().cpu().numpy()
    sal = grad.detach().abs().cpu().numpy()

    with torch.no_grad():
        ref = model(x).squeeze(-1)
        mask = np.zeros_like(ig)
        for t in range(x.shape[1]):
            for c in range(x.shape[2]):
                xm = x.clone()
                xm[:, t, c] = 0.0
                mask[:, t, c] = (model(xm).squeeze(-1) - ref).abs().cpu().numpy()

    results = {"Integrated Gradients": ig, "Saliency": sal, "Masking": mask}
    print(f"\n== SIR, target={target}, one surrogate, n={len(x)} validation samples ==")
    print(f"{'method':22s} " + " ".join(f"{c:>8s}" for c in CHANNELS) + "   share S/I/R   dominant S/I/R")
    profiles = {}
    for name, a in results.items():
        ch = a.sum(axis=1).mean(axis=0)
        dom = np.bincount(a.sum(axis=1).argmax(axis=1), minlength=3)
        share = ch / ch.sum()
        profiles[name] = a.sum(axis=2).mean(axis=0)
        print(f"{name:22s} " + " ".join(f"{v:8.4f}" for v in ch)
              + "   " + "/".join(f"{100 * s:.0f}" for s in share)
              + "      " + "/".join(str(d) for d in dom))
    names = list(profiles)
    for i in range(3):
        for j in range(i + 1, 3):
            rho = spearmanr(profiles[names[i]], profiles[names[j]]).correlation
            print(f"rank correlation of time profiles, {names[i]} vs {names[j]}: {rho:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    for ax, (name, a) in zip(axes, results.items()):
        prof = a.mean(axis=0)
        for c in range(3):
            ax.plot(prof[:, c], color=colors[c], label=CHANNELS[c])
        share = a.sum(axis=1).mean(axis=0)
        share = 100 * share / share.sum()
        ax.set_title(f"{name}  (S/I/R share {share[0]:.0f}/{share[1]:.0f}/{share[2]:.0f}%)", fontsize=10)
        ax.set_xlabel("Day")
    axes[0].set_ylabel("mean |attribution|")
    axes[0].legend()
    fig.suptitle(f"SIR, target={target}: three methods on one surrogate (n={len(x)})", y=1.02)
    fig.tight_layout()
    show_and_save(fig, "xai_05_method_comparison", "SIR method comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="lambd", choices=PARAM_NAMES)
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()
    main(args.target, args.n_samples)
