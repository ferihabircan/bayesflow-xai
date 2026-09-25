#!/usr/bin/env python
"""Integrated Gradients vs Saliency Map vs Masking on ONE trained
gravitational-wave surrogate (attention_rollout is skipped: gw_paper_cnn is
a CNN, same as GRF).

Trains once (seeded data) and saves a checkpoint to
outputs/models/<run>_gw_paper_cnn.pt; later runs reuse it. Every method's
attributions are written as outputs/<run>_<method>.npz in the format
scripts/plot_gw_ig_time.py reads (inputs, attributions (n, T, 2), ...), so:

    python scripts/run_gw_xai_comparison.py --config configs/gravitational_wave_zscore_workflow.yaml
    # -> outputs/figures/<run>_xai_comparison.png + printed summary table

Masking here is GW-specific: non-overlapping windows of --mask_window
samples, each channel masked separately (set to the zero-strain baseline),
|prediction shift| spread evenly over the window's samples. The generic
masking method masks single time steps across all channels, which gives no
H1/L1 split and needs T=8192 forward passes.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bayesflow_xai.registrations  # noqa: F401,E402
from bayesflow_xai.methods.generic_surrogate import TargetWrapper, train_generic_surrogate  # noqa: E402
from bayesflow_xai.registry import SIMULATORS, SUMMARY_NETWORKS, XAI_METHODS  # noqa: E402
from bayesflow_xai.utils.config import CONFIG, DEVICE, use_simulator_figures_dir  # noqa: E402
from run_workflow import _load_config, _save_loss_history  # noqa: E402

OUT_DIR = CONFIG.outputs_dir


def _train_or_load(config, spec, run):
    ckpt = os.path.join(OUT_DIR, "models", f"{run}_gw_paper_cnn.pt")
    body = SUMMARY_NETWORKS[config["summary_network"]](spec)
    if os.path.exists(ckpt):
        print(f"=== loading checkpoint {ckpt} (no retraining) ===")
        state = torch.load(ckpt, map_location=DEVICE)
        body.load_state_dict(state["body"])
        head = torch.nn.Linear(body.summary_dim, len(spec.param_names))
        head.load_state_dict(state["head"])
        body, head = body.to(DEVICE).eval(), head.to(DEVICE).eval()
        return body, head, state["X_val"].to(DEVICE), state["y_val"].to(DEVICE)

    n_sims, seed = config.get("n_sims", 2000), config.get("seed", 42)
    print(f"=== training on {n_sims} simulations ({config.get('epochs', 40)} epochs), data seed {seed} ===")
    X, y = spec.build_tensor_dataset(n_sims, seed=seed, **config.get("simulator_kwargs", {}))
    history = []
    body, head, X_val, y_val = train_generic_surrogate(
        body, X, y, epochs=config.get("epochs", 40), batch_size=config.get("batch_size", 64), seed=seed,
        log_prefix=run, history=history,
    )
    # Checkpoint first, so nothing after training can lose it.
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(
        {"body": body.state_dict(), "head": head.state_dict(), "X_val": X_val.cpu(), "y_val": y_val.cpu(),
         "config": config, "history": history},
        ckpt,
    )
    print(f"  saved checkpoint {ckpt}")
    _save_loss_history(history, f"{run}_gw_paper_cnn", n_sims)
    return body, head, X_val, y_val


def _masking(body, head, X, target_idx, window, baseline_value):
    model = TargetWrapper(body, head, target_idx).eval()
    n, T, C = X.shape
    attr = np.zeros((n, T, C), dtype=np.float32)
    with torch.no_grad():
        ref = model(X).squeeze(-1)
        for c in range(C):
            for start in range(0, T, window):
                xm = X.clone()
                xm[:, start:start + window, c] = baseline_value
                shift = (model(xm).squeeze(-1) - ref).abs().cpu().numpy()
                attr[:, start:start + window, c] = shift[:, None] / window
    return attr


def _save(path, X, attributions, y_val, target_idx, baseline_value, **extra):
    np.savez_compressed(
        path, inputs=X.detach().cpu().numpy(), attributions=attributions, y_val=y_val.detach().cpu().numpy(),
        target_idx=target_idx, baseline_value=baseline_value, **extra,
    )
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gravitational_wave_zscore_workflow.yaml")
    parser.add_argument("--mask_window", type=int, default=32, help="masking window in samples (32 = 15.6 ms)")
    args = parser.parse_args()

    config = _load_config(args.config)
    sim_name = config["simulator"]
    run = f"gwxai_{sim_name}" + (f"_{config['run_tag']}" if config.get("run_tag") else "")
    spec = SIMULATORS[sim_name]()
    use_simulator_figures_dir(sim_name)
    target = config["target"]
    target_idx = spec.param_names.index(target)
    body, head, X_val, y_val = _train_or_load(config, spec, run)

    xai_kwargs = dict(config.get("xai_kwargs", {}))
    n = min(xai_kwargs.get("n_samples", 64), len(X_val))
    X, y = X_val[:n], y_val[:n]
    baseline_value = X.median().item() if xai_kwargs.get("baseline") == "median" else 0.0
    paths = {}

    print("=== integrated_gradients ===")
    xai_kwargs["save_attributions"] = True
    XAI_METHODS["integrated_gradients"](spec, body, head, X_val, y_val, target_idx, target, fig_prefix=run, **xai_kwargs)
    os.replace(os.path.join(OUT_DIR, f"{run}_integrated_gradients.npz"), os.path.join(OUT_DIR, f"{run}_ig.npz"))
    paths["Integrated Gradients"] = os.path.join(OUT_DIR, f"{run}_ig.npz")

    print("=== saliency_map ===")
    _, grad = XAI_METHODS["saliency_map"](spec, body, head, X, y, target_idx, target, fig_prefix=run)
    paths["Saliency Map"] = os.path.join(OUT_DIR, f"{run}_saliency.npz")
    _save(paths["Saliency Map"], X, grad, y, target_idx, baseline_value)

    print(f"=== masking ({args.mask_window}-sample windows, per channel, mask value {baseline_value:.3g}) ===")
    attr = _masking(body, head, X, target_idx, args.mask_window, baseline_value)
    paths["Masking"] = os.path.join(OUT_DIR, f"{run}_masking.npz")
    _save(paths["Masking"], X, attr, y, target_idx, baseline_value, mask_window=args.mask_window)

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_gw_ig_time.py"),
           "--target", target, "--out", os.path.join(CONFIG.figures_dir, f"{run}_xai_comparison.png")]
    for label, p in paths.items():
        cmd += ["--npz", p, "--label", label]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
