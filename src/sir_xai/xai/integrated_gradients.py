"""XAI Part 2: Integrated Gradients (Captum) attribution over the
project's PyTorch surrogate trained on full (S, I, R) trajectories."""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sir_xai.utils.config import CONFIG, DEVICE
from sir_xai.simulation.sir_model import prior, stationary_SIR
from sir_xai.simulation.dataset import ensure_sir_fraction_scale
from sir_xai.utils.plotting import show_and_save
from sir_xai.xai.surrogate_models import train_sir_surrogate, TargetWrapper

CHANNEL_NAMES = ["S", "I", "R"]


def _compute_ig_attributions(
    target_idx: int,
    n_samples: int | None = None,
    train_surrogate_fn=train_sir_surrogate,
    n_sims: int | None = None,
):
    from captum.attr import IntegratedGradients

    body, head, X_val, _ = train_surrogate_fn(n_sims=n_sims if n_sims is not None else CONFIG.xai_surrogate_n_sims)
    model = TargetWrapper(body, head, target_idx)
    model.train()

    ig = IntegratedGradients(model)
    n = min(n_samples if n_samples is not None else CONFIG.xai_n_ig_samples, len(X_val))
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
        traj = {k: ensure_sir_fraction_scale(np.asarray(v)) for k, v in traj.items()}
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
    mean_vals = stats_df["mean_abs_attribution"].to_numpy()
    std_vals = stats_df["std_abs_attribution"].to_numpy()
    lower_err = np.minimum(std_vals, mean_vals)
    upper_err = std_vals
    axes[0].bar(CHANNEL_NAMES, mean_vals, yerr=np.vstack([lower_err, upper_err]), capsize=5)
    axes[0].set_ylim(bottom=0)
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


def _render_dot_pixel_saliency(fig, ax1, ax2, sample_input, sample_attr, selected_label, target_name):
    """Draws the S/I/R trajectory (ax1) and the pixel-weighted dot saliency
    map (ax2) for a single sample onto the given axes. Shared by
    plot_dot_pixel_saliency_map and plot_dot_pixel_saliency_for_all_samples
    so both produce the exact same figure layout/styling."""
    time_steps = np.arange(sample_input.shape[0])

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    attr_s_norm = normalize(sample_attr[:, 0])
    attr_i_norm = normalize(sample_attr[:, 1])
    attr_r_norm = normalize(sample_attr[:, 2])

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    _render_dot_pixel_saliency(fig, ax1, ax2, sample_input, sample_attr, selected_label, target_name)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(os.path.join("outputs", "sir_dot_pixel_saliency_map.png"), dpi=300, bbox_inches="tight")
    return fig


def plot_dot_pixel_saliency_for_all_samples(
    target_idx: int = 0,
    target_name: str = "lambd",
    n_samples: int = 10,
):
    """Same figure as plot_dot_pixel_saliency_map (trajectory on top,
    pixel-weighted dot saliency below), but instead of hunting for a single
    random "real outbreak" sample, renders one figure per sample for the
    first `n_samples` validation-set trajectories (X_val[:n_samples]) and
    saves each to outputs/figures/saliency_samples/sample_XX.png."""
    inputs, attributions, _ = _compute_ig_attributions(target_idx, n_samples=n_samples)
    n = attributions.shape[0]  # actual count, capped by available val data

    output_dir = os.path.join("outputs", "figures", "saliency_samples")
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for i in range(n):
        sample_input = inputs[i].detach().cpu().numpy()
        sample_attr = np.abs(attributions[i])
        peak_idx = int(np.argmax(sample_input[:, 1]))
        peak_value = float(sample_input[peak_idx, 1])
        selected_label = f"sample {i + 1}/{n} (peak I={peak_value:.4f} at t={peak_idx})"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        _render_dot_pixel_saliency(fig, ax1, ax2, sample_input, sample_attr, selected_label, target_name)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"sample_{i + 1:02d}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"  saved {out_path}")

    return saved_paths


_DEFAULT_CHANNEL_COLORS = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#8c564b", "#e377c2"]


def compute_channel_and_time_importance_stats(
    target_idx: int = 0,
    target_name: str = "lambd",
    n_samples: int = 200,
    channel_names: list[str] = CHANNEL_NAMES,
    train_surrogate_fn=train_sir_surrogate,
    n_sims: int | None = None,
    fig_name: str = "xai_04_channel_time_importance_stats",
    time_unit_label: str = "Day",
):
    """Runs the same dot-pixel-saliency IG attribution over `n_samples`
    synthetic trajectories (instead of just one) and aggregates, per
    sample: which channel has the largest total |attribution|, and which
    5 time steps are most/least important.

    Generalized over the underlying simulator/surrogate: `channel_names`
    labels whatever channels the surrogate's input has (e.g. S/I/R for the
    SIR surrogate, X1/X2 for the Lotka-Volterra-example surrogate), and
    `train_surrogate_fn` supplies the trained (body, head, X_val, y_val)
    for that simulator (e.g. train_sir_surrogate / train_lv_surrogate from
    xai/surrogate_models.py) -- this is the actual coupling point to a
    specific simulator, since the raw simulator itself is only reached
    indirectly through the surrogate's training dataset.

    Endpoint artifact: every surrogate here (SIRGRUSummaryNet,
    LVGRUSummaryNet) reads out only the GRU's final hidden state (see
    surrogate_models.py), so the last time step's input reaches the output
    through a single recurrent update while earlier steps' influence is
    diluted through many compounded updates. This makes the first and last
    time step dominate |IG attribution| almost by construction, regardless
    of IG baseline (verified for the SIR surrogate with both zero- and
    mean-baseline IG: the last step was argmax in 100% of samples under
    both). top5_idx/bottom5_idx therefore exclude both endpoints so the
    reported "important time steps" reflect the trajectory dynamics rather
    than this architectural readout effect. Since this is a property of
    the last-hidden-state GRU readout, not of any particular simulator, it
    applies the same way to the Lotka-Volterra-example surrogate.
    """

    inputs, attributions, delta = _compute_ig_attributions(
        target_idx, n_samples=n_samples, train_surrogate_fn=train_surrogate_fn, n_sims=n_sims,
    )
    n_samples = attributions.shape[0]  # actual count, capped by available val data
    n_days = attributions.shape[1]
    n_channels = attributions.shape[2]

    abs_attr = np.abs(attributions)  # (n, T, n_channels)

    channel_importance = abs_attr.sum(axis=1)  # (n, n_channels): total |attribution| per channel
    dominant_channel = np.argmax(channel_importance, axis=1)  # (n,)

    time_importance = abs_attr.sum(axis=2)  # (n, T): total |attribution| per time step

    # Exclude the first and last time step (endpoint architecture artifact,
    # see docstring) before ranking steps by importance. Ranks are computed
    # on the interior slice [1:-1], then shifted by +1 to map back to the
    # true time-step indices (0-based) for display/reporting.
    interior_time_importance = time_importance[:, 1:-1]  # (n, T-2)
    top5_idx = np.argsort(-interior_time_importance, axis=1)[:, :5] + 1  # (n, 5)
    bottom5_idx = np.argsort(interior_time_importance, axis=1)[:, :5] + 1  # (n, 5)

    channel_counts = np.bincount(dominant_channel, minlength=n_channels)

    print(f"\n== IG channel/time importance stats over {n_samples} samples (target={target_name}) ==")
    for name, count in zip(channel_names, channel_counts):
        print(f"  {name}: dominant channel in {count}/{n_samples} samples ({100 * count / n_samples:.1f}%)")
    print(f"mean convergence delta: {delta.abs().mean().item():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    bar_colors = [_DEFAULT_CHANNEL_COLORS[i % len(_DEFAULT_CHANNEL_COLORS)] for i in range(n_channels)]
    bars = axes[0].bar(channel_names, channel_counts, color=bar_colors)
    axes[0].set_ylabel("# samples where channel is most important")
    axes[0].set_title(f"Dominant channel across {n_samples} samples\n(target={target_name})")
    axes[0].bar_label(bars)

    endpoint_note = "(endpoints excluded: architecture artifact from last-hidden-state GRU readout)"

    axes[1].hist(top5_idx.flatten(), bins=n_days, range=(0, n_days), color="#ff7f0e", edgecolor="black")
    axes[1].set_xlabel(time_unit_label)
    axes[1].set_ylabel("Frequency (pooled top-5 per sample)")
    axes[1].set_title(f"Most important time steps\n(top-5 per sample, pooled)\n{endpoint_note}", fontsize=9)

    axes[2].hist(bottom5_idx.flatten(), bins=n_days, range=(0, n_days), color="#7f7f7f", edgecolor="black")
    axes[2].set_xlabel(time_unit_label)
    axes[2].set_ylabel("Frequency (pooled bottom-5 per sample)")
    axes[2].set_title(f"Least important time steps\n(bottom-5 per sample, pooled)\n{endpoint_note}", fontsize=9)

    show_and_save(
        fig,
        fig_name,
        f"XAI 4: Channel & Time Importance Stats ({target_name}, n={n_samples})",
    )

    stats_df = pd.DataFrame({
        "channel": channel_names,
        "dominant_count": channel_counts,
        "dominant_fraction": channel_counts / n_samples,
    })

    return fig, stats_df, dominant_channel, top5_idx, bottom5_idx
