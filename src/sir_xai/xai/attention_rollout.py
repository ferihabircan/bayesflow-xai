"""XAI Part 3: Transformer summary network + manual attention-rollout
(Abnar & Zuidema, 2020). BertViz only supports HuggingFace models, so
rollout is implemented directly against our custom nn.MultiheadAttention
layers instead."""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sir_xai.simulation.dataset import build_sir_tensor_dataset
from sir_xai.utils.config import CONFIG, PARAM_NAMES, DEVICE
from sir_xai.utils.plotting import show_and_save


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerSummaryNet(nn.Module):
    def __init__(self, in_dim: int = 3, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, summary_dim: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.input_scale = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            for _ in range(n_layers)
        ])
        self.pool = nn.Linear(d_model, summary_dim)
        self.attn_maps: list[torch.Tensor] = []

    def forward(self, x):
        self.attn_maps = []
        h = self.in_proj(x) * self.input_scale
        h = self.pos_encoding(h)
        for attn, norm, ffn in zip(self.layers, self.norms, self.ffns):
            attn_out, attn_w = attn(h, h, h, need_weights=True, average_attn_weights=True)
            h = norm(h + attn_out)
            h = norm(h + ffn(h))
            self.attn_maps.append(attn_w.detach())
        return self.pool(h.mean(dim=1))


def attention_rollout(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    batch, T, _ = attn_maps[0].shape
    eye = torch.eye(T, device=attn_maps[0].device).unsqueeze(0).expand(batch, -1, -1)
    result = eye.clone()
    for A in attn_maps:
        A_hat = 0.5 * A + 0.5 * eye
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
        result = torch.bmm(A_hat, result)
    return result


def _train_transformer(n_sims: int, epochs: int, batch_size: int = 64):
    X, y = build_sir_tensor_dataset(n_sims)
    n_val = int(0.1 * len(X))
    X_train, X_val = X[:-n_val].to(DEVICE), X[-n_val:].to(DEVICE)
    y_train, y_val = y[:-n_val].to(DEVICE), y[-n_val:].to(DEVICE)

    model = TransformerSummaryNet().to(DEVICE)
    head = nn.Linear(8, len(PARAM_NAMES)).to(DEVICE)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    with torch.enable_grad():
        n_batches = len(X_train) // batch_size
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train))
            ep_loss = 0.0
            for i in range(n_batches):
                idx = perm[i * batch_size:(i + 1) * batch_size]
                xb, yb = X_train[idx], y_train[idx]
                opt.zero_grad()
                loss = loss_fn(head(model(xb)), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    val_loss = loss_fn(head(model(X_val)), y_val).item()
                print(f"  [transformer] epoch {epoch+1}: train={ep_loss/n_batches:.4f} val={val_loss:.4f}")

    return model, X_val


def attention_rollout_analysis(target_idx: int = 0, target_name: str = "lambd",
                                epochs: int = 60):
    model, X_val = _train_transformer(CONFIG.xai_transformer_n_sims, epochs)
    model.eval()

    n = min(CONFIG.xai_n_ig_samples, len(X_val))
    with torch.no_grad():
        _ = model(X_val[:n])
    rollout = attention_rollout(model.attn_maps)

    received = rollout.mean(dim=1).mean(dim=0).cpu().numpy()

    fig_bar, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(received)), received)
    ax.set_xlabel("Day")
    ax.set_ylabel("Rolled-out attention (avg)")
    ax.set_title(f"Attention rollout - which days matter most (target={target_name})")
    show_and_save(fig_bar, "xai_03a_attention_rollout_bar", "XAI 3a: Attention Rollout (per day)")

    fig_matrix, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(rollout.mean(dim=0).cpu().numpy(), cmap="viridis")
    ax2.set_title("Mean attention-rollout matrix (query day x key day)")
    ax2.set_xlabel("Key day")
    ax2.set_ylabel("Query day")
    fig_matrix.colorbar(im, ax=ax2)
    show_and_save(fig_matrix, "xai_03b_attention_rollout_matrix", "XAI 3b: Attention Rollout Matrix")

    return fig_bar, fig_matrix, rollout
