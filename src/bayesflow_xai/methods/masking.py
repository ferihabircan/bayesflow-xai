"""XAI: Masking / Occlusion.

The most model-agnostic attribution method here: cover part of the input
with a baseline value (zero) and see how much the prediction moves. No
gradients involved at all -- unlike Integrated Gradients or Saliency Maps,
which both need autograd -- so it works even for a black-box predictor.
The tradeoff is cost: one forward pass per masked region instead of one
backward pass total.

For a (T, channels) time series (SIR/Lotka-Volterra), each time step is
masked in turn (every channel at that step set to 0) and the resulting
shift in the target prediction is recorded, producing a time-resolved
importance curve -- this is the "SIR/LV: mask each time point in turn"
method the registry exposes as XAI_METHODS["masking"].

For a (H, W) field (GRF), the same idea is applied to non-overlapping
spatial patches instead of time steps, producing an importance heatmap.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayesflow_xai.utils.plotting import show_and_save
from bayesflow_xai.methods.generic_surrogate import TargetWrapper


def _predict(model, x):
    with torch.no_grad():
        return model(x).squeeze(-1).cpu().numpy()


def masking_importance_timeseries(
    body,
    head,
    X_val,
    target_idx: int,
    target_name: str,
    n_samples: int = 50,
    fig_prefix: str = "workflow",
):
    """Occludes each time step of X_val[:n_samples] in turn (setting every
    channel at that step to 0) and measures |prediction shift| from the
    unmasked baseline prediction. Saves a time-resolved importance bar
    plot and returns (importance, mean_importance, fig), where importance
    has shape (n_samples, T)."""
    model = TargetWrapper(body, head, target_idx)
    model.eval()

    n = min(n_samples, len(X_val))
    x = X_val[:n].clone()
    T = x.shape[1]

    baseline_pred = _predict(model, x)
    importance = np.zeros((n, T))

    for t in range(T):
        x_masked = x.clone()
        x_masked[:, t, :] = 0.0
        masked_pred = _predict(model, x_masked)
        importance[:, t] = np.abs(baseline_pred - masked_pred)

    mean_importance = importance.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(T), mean_importance, color="#555555")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean |prediction shift| when masked")
    ax.set_title(f"Masking/Occlusion importance over time (target={target_name}, n={n})")
    show_and_save(fig, f"{fig_prefix}_masking_time_importance", "Masking / Occlusion")

    return importance, mean_importance, fig


def masking_importance_image(
    body,
    head,
    X_val,
    target_idx: int,
    target_name: str,
    n_samples: int = 50,
    patch_size: int = 8,
    fig_prefix: str = "workflow",
):
    """Occludes non-overlapping (patch_size, patch_size) spatial patches of
    X_val[:n_samples] in turn and measures the mean |prediction shift|,
    producing a coarse importance heatmap over the field."""
    model = TargetWrapper(body, head, target_idx)
    model.eval()

    n = min(n_samples, len(X_val))
    x = X_val[:n].clone()
    H, W = x.shape[-2], x.shape[-1]
    n_rows, n_cols = H // patch_size, W // patch_size

    baseline_pred = _predict(model, x)
    heatmap = np.zeros((n_rows, n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            x_masked = x.clone()
            x_masked[:, :, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = 0.0
            masked_pred = _predict(model, x_masked)
            heatmap[r, c] = np.abs(baseline_pred - masked_pred).mean()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(heatmap, cmap="hot")
    ax.set_title(f"Masking/Occlusion importance ({patch_size}x{patch_size} patches, target={target_name})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    show_and_save(fig, f"{fig_prefix}_masking_patch_importance", "Masking / Occlusion")

    return heatmap, fig


def masking_importance(spec, body, head, X_val, y_val, target_idx, target_name,
                        fig_prefix: str = "workflow", **kwargs):
    """Dispatches to the timeseries or image variant based on
    `spec.input_kind`. Signature matches XAI_METHODS's convention (see
    bayesflow_xai.registry.register_xai_method) so it can be registered
    directly; `y_val` is accepted but unused (masking only needs
    predictions, not ground truth)."""
    if spec.input_kind == "timeseries":
        return masking_importance_timeseries(
            body, head, X_val, target_idx, target_name, fig_prefix=fig_prefix,
        )
    if spec.input_kind == "image":
        return masking_importance_image(
            body, head, X_val, target_idx, target_name, fig_prefix=fig_prefix,
        )
    raise ValueError(f"masking_importance: unknown input_kind {spec.input_kind!r}")
