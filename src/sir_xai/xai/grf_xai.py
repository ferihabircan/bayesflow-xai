"""XAI for the GRF (Gaussian Random Field) example: pixel-level Integrated
Gradients (Captum) saliency over the 2D field, analogous to the SIR
dot-pixel-saliency map in xai/integrated_gradients.py but for a
(H, W) pixel grid instead of a (T, channels) time series.

Attribution runs against a plain-torch CNN surrogate (GRFCNNSummaryNet +
linear head) trained directly to regress (log_std, alpha) from the field,
kept separate from the production bf.networks.ConvolutionalNetwork for the
same Captum-compatibility reason documented in xai/surrogate_models.py."""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sir_xai.simulation.grf_model import prior, likelihood, FIELD_SHAPE, PARAM_NAMES
from sir_xai.utils.config import DEVICE, set_seed
from sir_xai.utils.plotting import show_and_save
from sir_xai.xai.surrogate_models import TargetWrapper


class GRFCNNSummaryNet(nn.Module):
    """Plain-torch CNN mirroring the production summary network's shape
    (widths=(8, 16, 32, 64), max-pool downsampling, flatten head)."""

    def __init__(self, field_shape=FIELD_SHAPE, widths=(8, 16, 32, 64), summary_dim: int = 6):
        super().__init__()
        layers = []
        in_ch = 1
        for w in widths:
            layers += [
                nn.Conv2d(in_ch, w, kernel_size=3, padding=1),
                nn.GroupNorm(1, w),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.conv = nn.Sequential(*layers)

        down_factor = 2 ** len(widths)
        flat_dim = widths[-1] * (field_shape[0] // down_factor) * (field_shape[1] // down_factor)
        self.head = nn.Linear(flat_dim, summary_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.head(h)


def build_grf_tensor_dataset(n_sims: int):
    """Returns X: (N, 1, H, W) fields, y: (N, 2) [log_std, alpha]."""
    thetas, fields = [], []
    for _ in range(n_sims):
        theta = prior()
        obs = likelihood(**theta)
        fields.append(np.asarray(obs["field"])[..., 0])
        thetas.append([theta[name] for name in PARAM_NAMES])

    X = torch.tensor(np.stack(fields), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.array(thetas, dtype=np.float32))
    return X, y


def train_grf_surrogate(n_sims: int, epochs: int = 30, batch_size: int = 32, seed: int = 42):
    """Trains GRFCNNSummaryNet + linear head to predict (log_std, alpha)
    from the field. Returns the trained body/head (not yet wrapped to a
    single target) plus a held-out validation split for attribution,
    mirroring xai/surrogate_models.py:train_sir_surrogate."""
    set_seed(seed)

    with torch.enable_grad():
        X, y = build_grf_tensor_dataset(n_sims)
        n_val = max(1, int(0.1 * len(X)))
        X_train, X_val = X[:-n_val].to(DEVICE), X[-n_val:].to(DEVICE)
        y_train, y_val = y[:-n_val].to(DEVICE), y[-n_val:].to(DEVICE)

        body = GRFCNNSummaryNet().to(DEVICE)
        head = nn.Linear(6, len(PARAM_NAMES)).to(DEVICE)

        body.train()
        head.train()

        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()

        n_batches = max(1, len(X_train) // batch_size)
        log_every = max(1, epochs // 3)
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            for i in range(n_batches):
                idx = perm[i * batch_size:(i + 1) * batch_size]
                xb, yb = X_train[idx], y_train[idx]

                opt.zero_grad()
                pred = head(body(xb))
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                epoch_loss += loss.item()

            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                body.eval()
                head.eval()
                with torch.no_grad():
                    val_loss = loss_fn(head(body(X_val)), y_val).item()
                print(f"  [GRF CNN surrogate] epoch {epoch+1}: train={epoch_loss/n_batches:.4f} val={val_loss:.4f}")
                body.train()
                head.train()

    body.eval()
    head.eval()

    return body, head, X_val, y_val


def grf_pixel_saliency_analysis(
    target_idx: int = 1,
    target_name: str = "alpha",
    n_sims: int = 2000,
    epochs: int = 30,
    sample_index: int = 0,
    seed: int = 42,
    fig_name: str = "grf_pixel_saliency",
):
    """Trains the CNN surrogate, computes Integrated Gradients pixel
    attribution for one validation-set field targeting `target_name`, and
    plots the original field next to the |attribution| heatmap
    (side-by-side, same pixel grid), saved to outputs/figures/."""
    from captum.attr import IntegratedGradients

    body, head, X_val, y_val = train_grf_surrogate(n_sims=n_sims, epochs=epochs, seed=seed)
    model = TargetWrapper(body, head, target_idx)
    model.train()

    sample_index = max(0, min(sample_index, len(X_val) - 1))
    inputs = X_val[sample_index:sample_index + 1].clone().requires_grad_(True)
    baseline = torch.zeros_like(inputs)

    ig = IntegratedGradients(model)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        attributions, delta = ig.attribute(inputs, baseline, return_convergence_delta=True)

    print(f"\n== GRF pixel IG attribution for target = {target_name} ==")
    print(f"mean convergence delta: {delta.abs().mean().item():.4f}")

    field = inputs[0, 0].detach().cpu().numpy()
    attr_map = np.abs(attributions[0, 0].detach().cpu().numpy())
    true_value = y_val[sample_index, target_idx].item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    im1 = ax1.imshow(field, cmap="RdBu_r")
    ax1.set_title(f"GRF sample (field)\ntrue {target_name}={true_value:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(attr_map, cmap="hot")
    ax2.set_title(f"|IG attribution|\ntarget={target_name}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    show_and_save(fig, fig_name, f"GRF XAI: Pixel Saliency (target={target_name})")

    return fig, field, attr_map, delta
