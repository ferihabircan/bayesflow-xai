#!/usr/bin/env python
"""Reproduces BayesFlow's Spatial_Data_and_Parameters.html tutorial, section
3.1 (parameter inference on Gaussian Random Fields): the power-spectrum
plot, the multi-alpha example-field grid, and BasicWorkflow training +
recovery/calibration diagnostics.

Reuses this project's grf_model.py (prior/likelihood/simulator) and
grf_workflow.py (ConvolutionalNetwork summary net + coupling_flow
BasicWorkflow), but trains with the tutorial's exact `fit_online` call
(num_batches_per_epoch=1000, epochs=20, batch_size=32) rather than the
project's offline early-stopping trainer, to keep tutorial parity.

Usage:
    python scripts/run_grf_tutorial_31.py --smoke-test
    python scripts/run_grf_tutorial_31.py --epochs 20 --num-batches-per-epoch 1000
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf
from FyeldGenerator import generate_field

from bayesflow_xai.utils.config import CONFIG, use_simulator_figures_dir
from bayesflow_xai.simulation.grf.grf_model import generate_power_spectrum, distribution, FIELD_SHAPE
from bayesflow_xai.training.grf_workflow import build_grf_workflow

use_simulator_figures_dir("grf")  # figures -> outputs/figures/grf/

FIGURES_DIR = Path(CONFIG.figures_dir) / "grf_tutorial"


def save_fig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def plot_power_spectrum():
    power_spectrum = generate_power_spectrum(3.0, 1.0)
    fig = plt.figure(figsize=(4, 2.5))
    k = np.logspace(0.001, 1, 300)
    plt.plot(k, power_spectrum(k))
    plt.title("Power Spectrum")
    plt.xlabel("k")
    plt.ylabel("Amplitude [a.u.]")
    save_fig(fig, "31_01_power_spectrum.png")


def plot_multi_alpha_grid():
    n_examples = 5
    alphas = np.linspace(2, 5, n_examples)
    log_std = 0
    cmap = "Spectral"

    fig, axs = plt.subplots(1, n_examples, figsize=(n_examples * 1.6, 1.7))
    for alpha, ax in zip(alphas, axs):
        power_spectrum = generate_power_spectrum(alpha, np.exp(log_std))
        field = generate_field(distribution, power_spectrum, FIELD_SHAPE)
        max_magnitude = np.max(np.abs(field))
        ax.imshow(field, cmap=cmap, vmin=-max_magnitude, vmax=max_magnitude)
        ax.set_title(f"$\\alpha={alpha:.2f}$")
        ax.set_axis_off()
    save_fig(fig, "31_02_multi_alpha_fields.png")


def train_and_diagnose(epochs, num_batches_per_epoch, batch_size, n_val, n_test):
    workflow, _summary_net = build_grf_workflow()
    simulator = workflow.simulator

    validation_data = simulator.sample(n_val)
    test_data = simulator.sample(n_test)

    workflow.fit_online(
        num_batches_per_epoch=num_batches_per_epoch,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
    )

    figs = workflow.plot_custom_diagnostics(
        test_data=test_data,
        plot_fns={
            "recovery": bf.diagnostics.recovery,
            "calibration": bf.diagnostics.calibration_ecdf,
        },
    )
    for name, fig in figs.items():
        save_fig(fig, f"31_03_{name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny epochs/sims, just to verify the pipeline runs end-to-end")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-batches-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.num_batches_per_epoch = 5
        args.n_val = 20
        args.n_test = 50

    plot_power_spectrum()
    plot_multi_alpha_grid()
    train_and_diagnose(
        args.epochs, args.num_batches_per_epoch, args.batch_size,
        args.n_val, args.n_test,
    )


if __name__ == "__main__":
    main()
