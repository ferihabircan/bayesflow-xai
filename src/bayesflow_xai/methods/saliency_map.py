"""XAI: Saliency Map.

The simplest possible gradient-based XAI method: take the raw gradient of
the model's output with respect to its input, at the input itself. No
baseline, no path integral (contrast with Integrated Gradients, which
averages the gradient along a whole path from a baseline to the input) --
just "how much would nudging this input value change the prediction, right
now." Cheap to compute (a single backward pass) but noisier and less
theoretically grounded than Integrated Gradients.

Simulator-agnostic: works for any registered `SimulatorSpec` (see
bayesflow_xai.registry), timeseries or image. This is the entry point used by
XAI_METHODS["saliency_map"] (see bayesflow_xai.registrations).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayesflow_xai.utils.plotting import show_and_save
from bayesflow_xai.methods.generic_surrogate import TargetWrapper


def saliency_map_generic(
    spec,
    body,
    head,
    X_val,
    y_val,
    target_idx: int,
    target_name: str,
    fig_prefix: str = "workflow",
    n_samples: int | None = None,
):
    model = TargetWrapper(body, head, target_idx)
    model.train()

    n = len(X_val) if n_samples is None else min(n_samples, len(X_val))
    inputs = X_val[:n].clone().requires_grad_(True)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        out = model(inputs)
        out.sum().backward()
    grad = np.abs(inputs.grad.detach().cpu().numpy())

    if spec.input_kind == "timeseries":
        channel_importance = grad.sum(axis=1).mean(axis=0)  # (C,)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(spec.channel_names, channel_importance, color="#777777")
        ax.set_ylabel("Sum |gradient| over time")
        ax.set_title(f"Saliency channel importance ({spec.name}, target={target_name})")
    else:
        mean_grad = grad.mean(axis=0)[0]  # (H, W)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(mean_grad, cmap="hot")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Saliency pixel map ({spec.name}, target={target_name})")

    show_and_save(fig, f"{fig_prefix}_saliency_map", "Saliency Map")
    return fig, grad
