#!/usr/bin/env python
"""Trains the BasicWorkflow and shows the 3 BayesFlow health-check plots."""

from bayesflow_xai.utils.config import CONFIG  # noqa: F401  (sets KERAS_BACKEND first)
from bayesflow_xai.training.workflow import build_workflow, train_workflow
from bayesflow_xai.diagnostics.plots import run_all_diagnostics


def main():
    print("=== Building and training BayesFlow BasicWorkflow ===")
    workflow, summary_net = build_workflow()
    history = train_workflow(workflow)

    print("\n=== BayesFlow diagnostics: loss / recovery / calibration ===")
    run_all_diagnostics(workflow, history)

    return workflow, summary_net, history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()
