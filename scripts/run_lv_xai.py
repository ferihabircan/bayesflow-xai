#!/usr/bin/env python
"""Runs XAI Part 4 (channel & time importance stats) against the
Lotka-Volterra-example surrogate (see simulation/lotka_volterra_model.py
and methods/surrogate_models.py:LVGRUSummaryNet). Parallel to
scripts/run_xai.py --stats, but only that part: the LV example here has no
BasicWorkflow/latent-space/attention-rollout counterpart, just the
standalone GRU surrogate + Integrated Gradients pipeline.

Usage: python scripts/run_lv_xai.py --target theta0 --stats --n-stats-samples 200
"""

import argparse

from bayesflow_xai.utils.config import CONFIG  # noqa: F401
from bayesflow_xai.simulation.lotka_volterra.lotka_volterra_model import PARAM_NAMES, CHANNEL_NAMES
from bayesflow_xai.methods.surrogate_models import train_lv_surrogate
from bayesflow_xai.methods.integrated_gradients import compute_channel_and_time_importance_stats


def main(target: str, run_stats: bool = False, n_stats_samples: int = 200, n_surrogate_sims: int = 6000):
    target_idx = PARAM_NAMES.index(target)

    if run_stats:
        print(f"\n=== LV XAI Part 4: Channel & time importance stats over {n_stats_samples} samples - target = {target} ===")
        compute_channel_and_time_importance_stats(
            target_idx=target_idx,
            target_name=target,
            n_samples=n_stats_samples,
            channel_names=CHANNEL_NAMES,
            train_surrogate_fn=train_lv_surrogate,
            n_sims=n_surrogate_sims,
            fig_name="xai_lv_04_channel_time_importance_stats",
            time_unit_label="Time step",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="theta0", choices=PARAM_NAMES)
    parser.add_argument("--stats", action="store_true",
                         help="run XAI Part 4: channel & time importance stats over many synthetic LV samples")
    parser.add_argument("--n-stats-samples", type=int, default=200,
                         help="number of synthetic LV samples to use for --stats (default: 200)")
    parser.add_argument("--n-surrogate-sims", type=int, default=6000,
                         help="number of LV simulations used to train the GRU surrogate (default: 6000)")
    args = parser.parse_args()

    main(args.target, args.stats, args.n_stats_samples, args.n_surrogate_sims)
