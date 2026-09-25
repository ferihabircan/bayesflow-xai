#!/usr/bin/env python
"""Single config-driven entry point for the whole project: pick a
simulator, a summary network, an inference network, a target parameter,
and an XAI method in a YAML file, and this script runs the pipeline end
to end by looking each name up in the bayesflow_xai.registry.

Usage:
    python scripts/run_workflow.py --config configs/workflow_example.yaml

To add your own simulator / summary network / XAI method, see the
docstrings in bayesflow_xai/registry.py and bayesflow_xai/registrations.py, then
either add your registration to registrations.py or import your own
module before this script's registry lookups run (e.g. by editing the
`import bayesflow_xai.registrations` line below to also import your module).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Populates bayesflow_xai.registry's SIMULATORS / SUMMARY_NETWORKS /
# INFERENCE_NETWORKS / XAI_METHODS dicts. Must happen before any lookup.
import bayesflow_xai.registrations  # noqa: F401
from bayesflow_xai.registry import SIMULATORS, SIMULATOR_TARGETS, SUMMARY_NETWORKS, INFERENCE_NETWORKS, XAI_METHODS
from bayesflow_xai.methods.generic_surrogate import train_generic_surrogate
from bayesflow_xai.utils.config import CONFIG


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_loss_history(history, prefix: str, n_sims: int):
    """Writes per-epoch (train, val) MSE to outputs/<prefix>_loss.npz and
    plots it to outputs/figures/<prefix>_loss.png."""
    epoch, train, val = (np.array(c) for c in zip(*history))
    np.savez(os.path.join(os.path.dirname(CONFIG.figures_dir), f"{prefix}_loss.npz"), epoch=epoch, train=train, val=val)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epoch, train, label="train")
    ax.plot(epoch, val, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (all target params, raw units)")
    ax.set_title(f"Surrogate training loss ({prefix.removeprefix('workflow_')}, n_sims={n_sims})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(CONFIG.figures_dir, f"{prefix}_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved loss curve to {path}")


def run_workflow(config: dict):
    # -----------------------------------------------------------------
    # 1) SIMULATOR: look up the registered builder, validate the target
    #    against that simulator's registered parameter list.
    # -----------------------------------------------------------------
    sim_name = config["simulator"]
    if sim_name not in SIMULATORS:
        raise ValueError(f"Unknown simulator '{sim_name}'. Registered simulators: {sorted(SIMULATORS)}")

    target = config["target"]
    valid_targets = SIMULATOR_TARGETS.get(sim_name, [])
    if target not in valid_targets:
        raise ValueError(
            f"Invalid target '{target}' for simulator '{sim_name}'. "
            f"Registered targets for '{sim_name}': {valid_targets}"
        )

    print(f"=== 1) Simulator: {sim_name} (target={target}) ===")
    spec = SIMULATORS[sim_name]()
    target_idx = spec.param_names.index(target)

    # -----------------------------------------------------------------
    # 2) SUMMARY NETWORK: build the surrogate body from the registry.
    # -----------------------------------------------------------------
    summary_name = config["summary_network"]
    if summary_name not in SUMMARY_NETWORKS:
        raise ValueError(f"Unknown summary_network '{summary_name}'. Registered: {sorted(SUMMARY_NETWORKS)}")

    print(f"=== 2) Summary network: {summary_name} ===")
    body = SUMMARY_NETWORKS[summary_name](spec)

    # -----------------------------------------------------------------
    # 3) INFERENCE NETWORK: validated + built here for parity with a full
    #    BasicWorkflow pipeline (see training/workflow.py). The XAI methods
    #    below attribute directly against the (body, head) surrogate and
    #    don't need it, but a config referencing an unregistered inference
    #    network still fails fast, here, with a clear message.
    # -----------------------------------------------------------------
    inference_name = config["inference_network"]
    if inference_name not in INFERENCE_NETWORKS:
        raise ValueError(f"Unknown inference_network '{inference_name}'. Registered: {sorted(INFERENCE_NETWORKS)}")

    print(f"=== 3) Inference network: {inference_name} (registered OK) ===")
    INFERENCE_NETWORKS[inference_name]()

    # -----------------------------------------------------------------
    # 4) TRAIN the surrogate: simulate n_sims examples from the chosen
    #    simulator, train summary-network body + linear head to regress
    #    the target parameters.
    # -----------------------------------------------------------------
    n_sims = config.get("n_sims", 2000)
    epochs = config.get("epochs", 40)
    batch_size = config.get("batch_size", 64)
    seed = config.get("seed", 42)

    print(f"=== 4) Training surrogate on {n_sims} simulations ({epochs} epochs) ===")
    X, y = spec.build_tensor_dataset(n_sims)
    history = []
    body, head, X_val, y_val = train_generic_surrogate(
        body, X, y, epochs=epochs, batch_size=batch_size, seed=seed, log_prefix=f"{sim_name}/{summary_name}",
        history=history,
    )
    _save_loss_history(history, f"workflow_{sim_name}_{summary_name}", n_sims)

    # -----------------------------------------------------------------
    # 5) XAI METHOD: run the selected method against the trained surrogate.
    # -----------------------------------------------------------------
    xai_name = config["xai_method"]
    if xai_name not in XAI_METHODS:
        raise ValueError(f"Unknown xai_method '{xai_name}'. Registered: {sorted(XAI_METHODS)}")

    print(f"=== 5) XAI method: {xai_name} (target={target}) ===")
    fig_prefix = f"workflow_{sim_name}_{xai_name}"
    result = XAI_METHODS[xai_name](
        spec, body, head, X_val, y_val, target_idx, target, fig_prefix=fig_prefix,
        **config.get("xai_kwargs", {}),
    )

    print(f"\nDone. Figure(s) saved under outputs/figures/{fig_prefix}_*.png")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/workflow_example.yaml", help="path to a workflow YAML config")
    parser.add_argument("--n_sims", type=int, help="override the config's n_sims (e.g. for a quick smoke run)")
    parser.add_argument("--epochs", type=int, help="override the config's epochs")
    args = parser.parse_args()

    config = _load_config(args.config)
    for key in ("n_sims", "epochs"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run_workflow(config)
