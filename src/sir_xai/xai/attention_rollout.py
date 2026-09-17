"""XAI Part 3: Transformer summary network + manual attention-rollout
(Abnar & Zuidema, 2020). BertViz only supports HuggingFace models, so
rollout is implemented directly against our custom nn.MultiheadAttention
layers instead."""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sir_xai.simulation.dataset import build_sir_tensor_dataset
from sir_xai.utils.config import CONFIG, PARAM_NAMES, DEVICE, set_seed
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


def plot_bertviz_style_connections(
    rollout_matrix,
    day_labels: list[str] | None = None,
    top_k: int = 15,
    save_name: str = "xai_03b_attention_connections",
):
    """BertViz "head view"-style rendering of a (T, T) attention-rollout
    matrix: a query-day column on the left, a key-day column on the right,
    and a line between them for each of the top_k strongest connections per
    query day (keeps the plot legible instead of drawing all T^2 lines).
    Line width/opacity/color all scale with the attention weight."""

    rollout_matrix = np.asarray(rollout_matrix)
    T = rollout_matrix.shape[0]
    if day_labels is None:
        day_labels = [str(i) for i in range(T)]

    top_k = min(top_k, T)
    max_weight = rollout_matrix.max() if rollout_matrix.max() > 0 else 1.0

    fig_height = max(6.0, T * 0.18)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    query_x, key_x = 0.0, 1.0
    # Day 0 at the top, day T-1 at the bottom, both columns aligned by day.
    y_of_day = {i: T - 1 - i for i in range(T)}

    cmap = plt.get_cmap("Blues")

    for q in range(T):
        row = rollout_matrix[q]
        top_keys = np.argsort(-row)[:top_k]
        for k in top_keys:
            w = row[k]
            if w <= 0:
                continue
            w_norm = w / max_weight
            ax.plot(
                [query_x, key_x],
                [y_of_day[q], y_of_day[k]],
                color=cmap(0.3 + 0.7 * w_norm),
                linewidth=0.5 + 4.0 * w_norm,
                alpha=0.15 + 0.75 * w_norm,
                solid_capstyle="round",
                zorder=1,
            )

    ax.scatter([query_x] * T, [y_of_day[i] for i in range(T)], color="#08306b", s=18, zorder=2)
    ax.scatter([key_x] * T, [y_of_day[i] for i in range(T)], color="#08306b", s=18, zorder=2)

    label_step = 1 if T <= 20 else 5
    for i in range(T):
        if i % label_step != 0 and i != T - 1:
            continue
        y = y_of_day[i]
        ax.text(query_x - 0.04, y, day_labels[i], ha="right", va="center", fontsize=8)
        ax.text(key_x + 0.04, y, day_labels[i], ha="left", va="center", fontsize=8)

    ax.text(query_x, T + 0.5, "Query days", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(key_x, T + 0.5, "Key days", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlim(query_x - 0.3, key_x + 0.3)
    ax.set_ylim(-1, T + 1.5)
    ax.axis("off")
    ax.set_title("Attention Rollout: Day-to-Day Information Flow (BertViz-style)", fontsize=12, pad=12)

    legend_elements = [
        Line2D([0], [0], color=cmap(0.3), lw=1.0, alpha=0.3, label="weak"),
        Line2D([0], [0], color=cmap(1.0), lw=4.0, alpha=0.9, label="strong"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.04), fontsize=9, title=f"attention weight (top-{top_k}/query day)")

    show_and_save(fig, save_name, "XAI 3b: Attention Connections (BertViz-style)")
    return fig


def _train_transformer(n_sims: int, epochs: int, batch_size: int = 64, seed: int = 42):
    set_seed(seed)
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

    rollout_matrix = rollout.mean(dim=0).cpu().numpy()

    fig_matrix, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(rollout_matrix, cmap="viridis")
    ax2.set_title("Mean attention-rollout matrix (query day x key day)")
    ax2.set_xlabel("Key day")
    ax2.set_ylabel("Query day")
    fig_matrix.colorbar(im, ax=ax2)
    show_and_save(fig_matrix, "xai_03b_attention_rollout_matrix", "XAI 3b: Attention Rollout Matrix")

    fig_connections = plot_bertviz_style_connections(rollout_matrix)

    return fig_bar, fig_matrix, fig_connections, rollout
