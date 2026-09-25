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

import yaml

# Populates bayesflow_xai.registry's SIMULATORS / SUMMARY_NETWORKS /
# INFERENCE_NETWORKS / XAI_METHODS dicts. Must happen before any lookup.
import bayesflow_xai.registrations  # noqa: F401
from bayesflow_xai.registry import SIMULATORS, SIMULATOR_TARGETS, SUMMARY_NETWORKS, INFERENCE_NETWORKS, XAI_METHODS
from bayesflow_xai.methods.generic_surrogate import train_generic_surrogate


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    body, head, X_val, y_val = train_generic_surrogate(
        body, X, y, epochs=epochs, batch_size=batch_size, seed=seed, log_prefix=f"{sim_name}/{summary_name}",
    )

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
    )

    print(f"\nDone. Figure(s) saved under outputs/figures/{fig_prefix}_*.png")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/workflow_example.yaml", help="path to a workflow YAML config")
    args = parser.parse_args()

    run_workflow(_load_config(args.config))
