#!/usr/bin/env python
"""Runs the 3 XAI analyses against a fresh/trained summary network.
Usage: python scripts/run_xai.py --target lambd
"""

import argparse

from sir_xai.utils.config import CONFIG  # noqa: F401
from sir_xai.training.workflow import build_workflow, train_workflow
from sir_xai.xai.latent_space import latent_space_analysis
from sir_xai.xai.integrated_gradients import (
    integrated_gradients_analysis,
    plot_dot_pixel_saliency_map,
    compute_channel_and_time_importance_stats,
)
from sir_xai.xai.attention_rollout import attention_rollout_analysis
from sir_xai.utils.config import PARAM_NAMES


def main(target: str, skip_training: bool = False, sample_index: int = 0,
         run_stats: bool = False, n_stats_samples: int = 200):
    target_idx = PARAM_NAMES.index(target)

    print("=== Building and training BayesFlow BasicWorkflow (for latent-space XAI) ===")
    workflow, summary_net = build_workflow()
    if not skip_training:
        train_workflow(workflow)

    print(f"\n=== XAI Part 1: Latent space (UMAP / t-SNE), colored by {target} ===")
    latent_space_analysis(summary_net, n_sims=CONFIG.xai_n_latent_sims, color_by=target)

    print(f"\n=== XAI Part 2: Integrated Gradients (Captum) - target = {target} ===")
    integrated_gradients_analysis(target_idx=target_idx, target_name=target)
    print(f"\n=== XAI Part 2b: Dot-pixel saliency map - target = {target}, sample = {sample_index} ===")
    plot_dot_pixel_saliency_map(sample_index=sample_index, target_idx=target_idx, target_name=target)

    print(f"\n=== XAI Part 3: Attention Rollout - target = {target} ===")
    attention_rollout_analysis(target_idx=target_idx, target_name=target)

    if run_stats:
        print(f"\n=== XAI Part 4: Channel & time importance stats over {n_stats_samples} samples - target = {target} ===")
        compute_channel_and_time_importance_stats(
            target_idx=target_idx, target_name=target, n_samples=n_stats_samples,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="lambd", choices=PARAM_NAMES)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="which validation sample to use for the dot-pixel saliency map")
    parser.add_argument("--skip-training", action="store_true",
                         help="skip BasicWorkflow training if you only need XAI parts 2-3 (surrogates train regardless)")
    parser.add_argument("--stats", action="store_true",
                         help="run XAI Part 4: channel & time importance stats over many synthetic samples")
    parser.add_argument("--n-stats-samples", type=int, default=200,
                         help="number of synthetic outbreak samples to use for --stats (default: 200)")
    args = parser.parse_args()

    main(args.target, args.skip_training, args.sample_index, args.stats, args.n_stats_samples)
    plt.show()
