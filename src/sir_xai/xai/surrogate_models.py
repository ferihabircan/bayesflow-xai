"""Surrogate torch models used only for attribution/explanation, trained on
the explicit (S, I, R) trajectories rather than the aggregated `cases`
series the original BasicWorkflow sees. Kept separate from
training/networks.py since these are XAI tooling, not the production model."""

import torch
import torch.nn as nn

from sir_xai.simulation.dataset import build_sir_tensor_dataset, build_lv_tensor_dataset
from sir_xai.simulation.lotka_volterra_model import DIM_THETA as LV_DIM_THETA, N_CHANNELS as LV_N_CHANNELS
from sir_xai.utils.config import PARAM_NAMES, DEVICE, set_seed


class SIRGRUSummaryNet(nn.Module):
    """GRU summary network with explicit 3-channel (S, I, R) input."""

    def __init__(self, hidden: int = 64, summary_dim: int = 8):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, summary_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


class LVGRUSummaryNet(nn.Module):
    """GRU summary network with explicit 2-channel Lotka-Volterra-example
    input (see simulation/lotka_volterra_model.py). Same last-hidden-state
    readout as SIRGRUSummaryNet, so it inherits the same endpoint-artifact
    caveat documented on compute_channel_and_time_importance_stats."""

    def __init__(self, hidden: int = 64, summary_dim: int = 8):
        super().__init__()
        self.gru = nn.GRU(input_size=LV_N_CHANNELS, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, summary_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


class TargetWrapper(nn.Module):
    """Wraps a summary net + regression head, exposing a single scalar
    target (one parameter) so Captum's IntegratedGradients has a clean
    scalar output to attribute."""

    def __init__(self, body: nn.Module, head: nn.Linear, target_idx: int):
        super().__init__()
        self.body = body
        self.head = head
        self.target_idx = target_idx

    def forward(self, x):
        return self.head(self.body(x))[:, self.target_idx:self.target_idx + 1]


def train_sir_surrogate(n_sims: int, epochs: int = 60, batch_size: int = 64, seed: int = 42):
    """Trains SIRGRUSummaryNet + linear head to predict log1p(theta) from
    (S, I, R). Returns the trained body/head (not yet wrapped to a single
    target) plus a held-out validation split for attribution."""
    set_seed(seed)

    # BayesFlow'un kapatmış olabileceği autograd'ı sürrogat eğitim için zorunlu olarak açıyoruz
    with torch.enable_grad():
        X, y = build_sir_tensor_dataset(n_sims)
        n_val = int(0.1 * len(X))
        X_train, X_val = X[:-n_val].to(DEVICE), X[-n_val:].to(DEVICE)
        y_train, y_val = y[:-n_val].to(DEVICE), y[-n_val:].to(DEVICE)

        body = SIRGRUSummaryNet().to(DEVICE)
        head = nn.Linear(8, len(PARAM_NAMES)).to(DEVICE)
        
        # Modelleri explicit olarak train moduna alıyoruz
        body.train()
        head.train()

        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()

        n_batches = len(X_train) // batch_size
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
                
            if (epoch + 1) % 10 == 0:
                body.eval()
                head.eval()
                with torch.no_grad():
                    val_loss = loss_fn(head(body(X_val)), y_val).item()
                print(f"  [surrogate GRU] epoch {epoch+1}: train={epoch_loss/n_batches:.4f} val={val_loss:.4f}")
                body.train()
                head.train()

    # XAI (Integrated Gradients) türev alacağı için değerlendirme moduna alıyoruz
    body.eval()
    head.eval()

    return body, head, X_val, y_val


def train_lv_surrogate(n_sims: int, epochs: int = 60, batch_size: int = 64, seed: int = 42):
    """Trains LVGRUSummaryNet + linear head to predict theta (already in
    [-1, 1], no log1p needed) from the 2-channel observables produced by
    lotka_volterra_model.sample_fn. Returns the trained body/head (not yet
    wrapped to a single target) plus a held-out validation split for
    attribution, exactly like train_sir_surrogate."""
    set_seed(seed)

    with torch.enable_grad():
        X, y = build_lv_tensor_dataset(n_sims)
        n_val = int(0.1 * len(X))
        X_train, X_val = X[:-n_val].to(DEVICE), X[-n_val:].to(DEVICE)
        y_train, y_val = y[:-n_val].to(DEVICE), y[-n_val:].to(DEVICE)

        body = LVGRUSummaryNet().to(DEVICE)
        head = nn.Linear(8, LV_DIM_THETA).to(DEVICE)

        body.train()
        head.train()

        opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()

        n_batches = len(X_train) // batch_size
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

            if (epoch + 1) % 10 == 0:
                body.eval()
                head.eval()
                with torch.no_grad():
                    val_loss = loss_fn(head(body(X_val)), y_val).item()
                print(f"  [surrogate LV-GRU] epoch {epoch+1}: train={epoch_loss/n_batches:.4f} val={val_loss:.4f}")
                body.train()
                head.train()

    body.eval()
    head.eval()

    return body, head, X_val, y_val