#!/usr/bin/env python
"""Full pipeline: train workflow, show diagnostics, then run all XAI analyses
on the best-recovered parameter (lambd), each figure in its own window."""

import matplotlib.pyplot as plt

from bayesflow_xai.utils.config import CONFIG  # noqa: F401
from bayesflow_xai.training.workflow import build_workflow, train_workflow
from bayesflow_xai.diagnostics.plots import run_all_diagnostics
from bayesflow_xai.methods.latent_space import latent_space_analysis
from bayesflow_xai.methods.integrated_gradients import (
    integrated_gradients_analysis,
    plot_dot_pixel_saliency_map,
)
from bayesflow_xai.methods.attention_rollout import attention_rollout_analysis


def main():
    print("=== Building and training BayesFlow BasicWorkflow ===")
    workflow, summary_net = build_workflow()
    history = train_workflow(workflow)

    print("\n=== BayesFlow diagnostics: loss / recovery / calibration ===")
    run_all_diagnostics(workflow, history)

    target = CONFIG.xai_target_param
    target_idx = ["lambd", "mu", "D", "I0", "psi"].index(target)

    print(f"\n=== XAI 1: Latent space, colored by {target} ===")
    latent_space_analysis(summary_net, n_sims=CONFIG.xai_n_latent_sims, color_by=target)

    print(f"\n=== XAI 2: Integrated Gradients - target = {target} ===")
    integrated_gradients_analysis(target_idx=target_idx, target_name=target)
    print(f"\n=== XAI 2b: Dot-pixel saliency map - target = {target} ===")
    plot_dot_pixel_saliency_map(sample_index=0, target_idx=target_idx, target_name=target)

    print(f"\n=== XAI 3: Attention Rollout - target = {target} ===")
    attention_rollout_analysis(target_idx=target_idx, target_name=target)

    print("\nAll diagnostic and XAI figures generated. Close windows to exit.")


if __name__ == "__main__":
    main()
    plt.show()
