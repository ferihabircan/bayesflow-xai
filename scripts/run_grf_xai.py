#!/usr/bin/env python
"""Simplified GRF (Gaussian Random Field) analogue of scripts/run_xai.py:
optionally trains the BasicWorkflow (bf.BasicWorkflow + ConvolutionalNetwork
summary net), then computes pixel-level Integrated Gradients saliency for
one target parameter via a separate plain-torch CNN surrogate (see
methods/grf_xai.py), saved to outputs/figures/.

Usage: python scripts/run_grf_xai.py --target alpha --skip-training
"""

import argparse

from xai.utils.config import CONFIG  # noqa: F401
from xai.simulation.grf.grf_model import PARAM_NAMES
from xai.training.grf_workflow import build_grf_workflow, train_grf_workflow
from xai.methods.grf_xai import grf_pixel_saliency_analysis


def main(
    target: str,
    skip_training: bool = False,
    n_train_sims: int = 2000,
    n_val_sims: int = 100,
    epochs: int = 50,
    batch_size: int = 32,
    n_surrogate_sims: int = 2000,
    surrogate_epochs: int = 30,
    sample_index: int = 0,
):
    target_idx = PARAM_NAMES.index(target)

    print("=== Building and training BayesFlow BasicWorkflow (GRF) ===")
    workflow, summary_net = build_grf_workflow()
    if not skip_training:
        train_grf_workflow(
            workflow,
            n_train_sims=n_train_sims,
            n_val_sims=n_val_sims,
            epochs=epochs,
            batch_size=batch_size,
        )

    print(f"\n=== GRF XAI: Pixel-level Integrated Gradients - target = {target} ===")
    grf_pixel_saliency_analysis(
        target_idx=target_idx,
        target_name=target,
        n_sims=n_surrogate_sims,
        epochs=surrogate_epochs,
        sample_index=sample_index,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="alpha", choices=PARAM_NAMES)
    parser.add_argument("--skip-training", action="store_true",
                         help="skip BasicWorkflow training (the CNN surrogate used for XAI trains regardless)")
    parser.add_argument("--n-train-sims", type=int, default=2000,
                         help="number of GRF simulations for BasicWorkflow training (default: 2000)")
    parser.add_argument("--n-val-sims", type=int, default=100,
                         help="number of GRF simulations for BasicWorkflow validation (default: 100)")
    parser.add_argument("--epochs", type=int, default=50,
                         help="BasicWorkflow training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="BasicWorkflow training batch size (default: 32)")
    parser.add_argument("--n-surrogate-sims", type=int, default=2000,
                         help="number of GRF simulations for the XAI CNN surrogate (default: 2000)")
    parser.add_argument("--surrogate-epochs", type=int, default=30,
                         help="XAI CNN surrogate training epochs (default: 30)")
    parser.add_argument("--sample-index", type=int, default=0,
                         help="which validation sample to use for the pixel saliency map (default: 0)")
    args = parser.parse_args()

    main(
        args.target, args.skip_training, args.n_train_sims, args.n_val_sims,
        args.epochs, args.batch_size, args.n_surrogate_sims, args.surrogate_epochs,
        args.sample_index,
    )
    plt.show()
