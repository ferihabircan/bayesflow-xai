"""XAI Part 2: Integrated Gradients (Captum) attribution over the
project's PyTorch surrogate trained on full (S, I, R) trajectories."""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sir_xai.utils.config import CONFIG, DEVICE
from sir_xai.simulation.sir_model import prior, stationary_SIR
from sir_xai.utils.plotting import show_and_save
from sir_xai.xai.surrogate_models import train_sir_surrogate, TargetWrapper

CHANNEL_NAMES = ["S", "I", "R"]


def _compute_ig_attributions(target_idx: int):
    from captum.attr import IntegratedGradients

    body, head, X_val, _ = train_sir_surrogate(n_sims=CONFIG.xai_surrogate_n_sims)
    model = TargetWrapper(body, head, target_idx)
    model.train()

    ig = IntegratedGradients(model)
    n = min(CONFIG.xai_n_ig_samples, len(X_val))
    inputs = X_val[:n].clone().requires_grad_(True)
    baseline = torch.zeros_like(inputs)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        attributions, delta = ig.attribute(inputs, baseline, return_convergence_delta=True)

    return inputs, attributions.detach().cpu().numpy(), delta


def _sample_outbreak_trajectory(
    max_tries: int = 100,
    peak_threshold: float = 0.05,
    min_post_peak_drop: float = 0.2,
):
    for try_idx in range(1, max_tries + 1):
        theta = prior()
        traj = stationary_SIR(**theta, return_full=True)
        sample = np.stack([traj["S"], traj["I"], traj["R"]], axis=-1)
        peak_idx = int(np.argmax(traj["I"]))
        peak_value = float(traj["I"][peak_idx])
        final_value = float(traj["I"][-1])
        if peak_value > peak_threshold and peak_idx < len(traj["I"]) - 10 and final_value < peak_value * min_post_peak_drop:
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            return sample_tensor, theta, peak_idx, peak_value, final_value, try_idx

    raise RuntimeError(
        f"Could not find an outbreak sample with peak I > {peak_threshold} and a clear post-peak decline after {max_tries} tries. "
        "This suggests the current prior() or SIR dynamics rarely produce a visible outbreak."
    )


def integrated_gradients_analysis(
    target_idx: int = 0,
    target_name: str = "lambd",
):
    from scipy import stats as sps

    inputs, attributions, delta = _compute_ig_attributions(target_idx)

    channel_importance = np.abs(attributions).sum(axis=1)
    stats_df = pd.DataFrame({
        "channel": CHANNEL_NAMES,
        "mean_abs_attribution": channel_importance.mean(axis=0),
        "std_abs_attribution": channel_importance.std(axis=0),
        "median_abs_attribution": np.median(channel_importance, axis=0),
    })
    print(f"\n== IG channel importance for target = {target_name} ==")
    print(stats_df.to_string(index=False))
    print(f"mean convergence delta: {delta.abs().mean().item():.4f}")

    for i in range(3):
        for j in range(i + 1, 3):
            t, p = sps.ttest_rel(channel_importance[:, i], channel_importance[:, j])
            print(f"{CHANNEL_NAMES[i]} vs {CHANNEL_NAMES[j]}: t={t:.3f}, p={p:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(CHANNEL_NAMES, stats_df["mean_abs_attribution"], yerr=stats_df["std_abs_attribution"], capsize=5)
    axes[0].set_title(f"Mean |IG attribution| by population (target={target_name})")
    axes[0].set_ylabel("Sum |attribution| over time")

    mean_over_time = np.abs(attributions).mean(axis=0)
    for c in range(3):
        axes[1].plot(mean_over_time[:, c], label=CHANNEL_NAMES[c], marker="o")
    axes[1].set_title("Time-resolved mean |IG attribution|")
    axes[1].set_xlabel("Day")
    axes[1].legend()

    show_and_save(fig, "xai_02_integrated_gradients", "XAI 2: Integrated Gradients")
    return fig, stats_df, attributions


def plot_dot_pixel_saliency_map(
    sample_index: int = 0,
    target_idx: int = 0,
    target_name: str = "lambd",
    inputs: torch.Tensor | None = None,
    attributions: np.ndarray | None = None,
):
    if inputs is None or attributions is None:
        sample_input_tensor, theta, peak_idx, peak_value, final_value, n_tries = _sample_outbreak_trajectory()
        sample_input = sample_input_tensor.detach().cpu().numpy()
        selected_label = (
            f"real outbreak sample (found in {n_tries} tries, peak I={peak_value:.4f} at t={peak_idx}, final I={final_value:.4f}, "
            f"R0={theta['lambd'] / theta['mu']:.2f})"
        )

        from captum.attr import IntegratedGradients

        body, head, _, _ = train_sir_surrogate(n_sims=CONFIG.xai_surrogate_n_sims)
        model = TargetWrapper(body, head, target_idx)
        model.train()

        inputs = sample_input_tensor.unsqueeze(0).to(DEVICE)
        baseline = torch.zeros_like(inputs)
        ig = IntegratedGradients(model)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            attributions, _ = ig.attribute(inputs, baseline, return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()
        sample_attr = np.abs(attributions[0])
    else:
        sample_index = max(0, min(sample_index, len(inputs) - 1))
        sample_input = inputs[sample_index].detach().cpu().numpy()
        sample_attr = np.abs(attributions[sample_index])
        peak_idx = int(np.argmax(sample_input[:, 1]))
        peak_value = float(sample_input[peak_idx, 1])
        selected_label = f"sample={sample_index} (peak I={peak_value:.4f} at t={peak_idx})"

    time_steps = np.arange(sample_input.shape[0])

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    attr_s_norm = normalize(sample_attr[:, 0])
    attr_i_norm = normalize(sample_attr[:, 1])
    attr_r_norm = normalize(sample_attr[:, 2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_steps, sample_input[:, 0], color="#2ca02c", lw=2, label="Susceptible (S)")
    ax1.plot(time_steps, sample_input[:, 1], color="#d62728", lw=2, ls="--", label="Infected (I)")
    ax1.plot(time_steps, sample_input[:, 2], color="#1f77b4", lw=2, ls=":", label="Recovered (R)")
    ax1.set_ylabel("Population Fraction")
    ax1.set_title(f"SIR Trajectory Signal ({selected_label})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    sc_s = ax2.scatter(
        time_steps,
        sample_input[:, 0],
        c=attr_s_norm,
        cmap="Greens",
        s=attr_s_norm * 140 + 15,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="S Effect",
    )
    sc_i = ax2.scatter(
        time_steps,
        sample_input[:, 1],
        c=attr_i_norm,
        cmap="Reds",
        s=attr_i_norm * 140 + 15,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="I Effect",
    )
    sc_r = ax2.scatter(
        time_steps,
        sample_input[:, 2],
        c=attr_r_norm,
        cmap="Blues",
        s=attr_r_norm * 140 + 15,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        label="R Effect",
    )

    ax2.plot(time_steps, sample_input[:, 0], color="#2ca02c", alpha=0.3, lw=1)
    ax2.plot(time_steps, sample_input[:, 1], color="#d62728", alpha=0.3, lw=1, ls="--")
    ax2.plot(time_steps, sample_input[:, 2], color="#1f77b4", alpha=0.3, lw=1, ls=":")

    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Population Fraction (Pixel-Weighted)")
    ax2.set_title(f"Dot-Pixel Saliency Map (target={target_name}, {selected_label})")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    cbar_s = fig.colorbar(sc_s, ax=ax2, orientation="horizontal", pad=0.16, fraction=0.03)
    cbar_s.set_label("S Importance")
    cbar_i = fig.colorbar(sc_i, ax=ax2, orientation="horizontal", pad=0.22, fraction=0.03)
    cbar_i.set_label("I Importance")
    cbar_r = fig.colorbar(sc_r, ax=ax2, orientation="horizontal", pad=0.28, fraction=0.03)
    cbar_r.set_label("R Importance")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(os.path.join("outputs", "sir_dot_pixel_saliency_map.png"), dpi=300, bbox_inches="tight")
    return fig
