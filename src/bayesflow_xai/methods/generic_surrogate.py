"""Simulator-agnostic surrogate network bodies + trainer, used by the
registry-driven workflow (scripts/run_workflow.py + bayesflow_xai.registrations).

The project's existing per-simulator surrogates (methods/surrogate_models.py's
SIRGRUSummaryNet/LVGRUSummaryNet, methods/grf_xai.py's GRFCNNSummaryNet) stay
exactly as they are and keep powering scripts/run_xai.py,
run_lv_xai.py, run_grf_xai.py unchanged. This module exists so a new
simulator only needs a dataset builder + shape info (see
`bayesflow_xai.registry.SimulatorSpec`) to be trained against any registered
summary network, instead of hand-writing a new nn.Module + training loop
each time.
"""

import torch
import torch.nn as nn

from bayesflow_xai.utils.config import DEVICE, set_seed


class GenericGRUSummaryNet(nn.Module):
    """GRU summary network over a (batch, T, in_channels) time series."""

    def __init__(self, in_channels: int, hidden: int = 64, summary_dim: int = 8):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden, batch_first=True)
        self.head_in = nn.Linear(hidden, summary_dim)
        self.summary_dim = summary_dim

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head_in(h_n[-1])


class GenericCNNSummaryNet(nn.Module):
    """Small CNN summary network over a (batch, 1, H, W) field, same shape
    as methods/grf_xai.py:GRFCNNSummaryNet."""

    def __init__(self, field_shape=(64, 64), widths=(8, 16, 32, 64), summary_dim: int = 6):
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
        self.head_in = nn.Linear(flat_dim, summary_dim)
        self.summary_dim = summary_dim

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.head_in(h)


class TargetWrapper(nn.Module):
    """Wraps body+head to expose a single scalar target, so Captum-style
    attribution methods have a clean scalar output to attribute (mirrors
    methods/surrogate_models.py:TargetWrapper)."""

    def __init__(self, body: nn.Module, head: nn.Linear, target_idx: int):
        super().__init__()
        self.body = body
        self.head = head
        self.target_idx = target_idx

    def forward(self, x):
        return self.head(self.body(x))[:, self.target_idx : self.target_idx + 1]


def train_generic_surrogate(
    body: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 64,
    seed: int = 42,
    lr: float = 1e-3,
    log_prefix: str = "surrogate",
):
    """Trains `body` + a linear head to regress y from X. Shared training
    loop for every registry-driven summary network (extracted from the
    near-identical loops in methods/surrogate_models.py and methods/grf_xai.py).
    Returns (body, head, X_val, y_val)."""
    set_seed(seed)

    with torch.enable_grad():
        n_val = max(1, int(0.1 * len(X)))
        X_train, X_val = X[:-n_val].to(DEVICE), X[-n_val:].to(DEVICE)
        y_train, y_val = y[:-n_val].to(DEVICE), y[-n_val:].to(DEVICE)

        body = body.to(DEVICE)
        head = nn.Linear(body.summary_dim, y.shape[1]).to(DEVICE)
        body.train()
        head.train()

        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        n_batches = max(1, len(X_train) // batch_size)
        log_every = max(1, epochs // 3)
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train))
            epoch_loss = 0.0
            for i in range(n_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                xb, yb = X_train[idx], y_train[idx]
                opt.zero_grad()
                loss = loss_fn(head(body(xb)), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                body.eval()
                head.eval()
                with torch.no_grad():
                    val_loss = loss_fn(head(body(X_val)), y_val).item()
                print(f"  [{log_prefix}] epoch {epoch + 1}: train={epoch_loss / n_batches:.4f} val={val_loss:.4f}")
                body.train()
                head.train()

    body.eval()
    head.eval()
    return body, head, X_val, y_val
