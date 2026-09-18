#!/usr/bin/env python
"""Reproduces BayesFlow's Spatial_Data_and_Parameters.html tutorial, section
3.2 (image generation with a diffusion model): field+histogram grids and
min/max value distributions for both simulator versions (the original GRF
simulator from section 3.1 and the unit_length-normalized revision), then
DiffusionModel (ResidualUViT subnet) training via BasicWorkflow, its loss
curve, and a real-vs-generated field comparison.

The tutorial itself runs the field+histogram grid
(`fig, axs = plt.subplots(2, n_examples, ...)`) only once, before the
unit_length revision. Per request, this script runs it for BOTH simulator
versions, reusing the same plotting code with the revised simulator's
`generate_field(..., unit_length=...)` call.

Usage:
    python scripts/run_grf_tutorial_32.py --smoke-test
    python scripts/run_grf_tutorial_32.py --epochs 20 --num-batches-per-epoch 1000
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf
from FyeldGenerator import generate_field

from sir_xai.utils.config import CONFIG
from sir_xai.simulation.grf_model import generate_power_spectrum, FIELD_SHAPE
from sir_xai.simulation.grf_model import distribution as original_distribution
from sir_xai.simulation.grf_model import build_grf_simulator
from sir_xai.simulation.grf_generative_model import (
    distribution as unit_length_distribution,
    build_grf_generative_simulator,
)
from sir_xai.training.grf_diffusion_workflow import (
    build_grf_diffusion_workflow,
    train_grf_diffusion_workflow,
)

FIGURES_DIR = Path(CONFIG.figures_dir) / "grf_tutorial"


def save_fig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def plot_field_histogram_grid(distribution_fn, use_unit_length, filename, log_std=0.0):
    n_examples = 5
    alphas = np.linspace(2, 5, n_examples)
    cmap = "Spectral"

    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples * 1.6, 1.9 * 1.7))
    for a, alpha in enumerate(alphas):
        power_spectrum = generate_power_spectrum(alpha, np.exp(log_std))
        kwargs = {"unit_length": 1 / (np.abs(alpha) + 1e-7)} if use_unit_length else {}
        field = generate_field(distribution_fn, power_spectrum, FIELD_SHAPE, **kwargs)
        max_magnitude = np.max(np.abs(field))
        axs[0, a].imshow(field, cmap=cmap, vmin=-max_magnitude, vmax=max_magnitude)
        axs[0, a].set_title(f"$\\alpha={alpha:.2f}$")
        axs[0, a].set_axis_off()
        axs[1, a].hist(field.flatten(), bins=40, color="blue", alpha=0.5)
        axs[1, a].spines[["left", "right", "top"]].set_visible(False)

    plt.tight_layout()
    save_fig(fig, filename)


def plot_minmax_histogram(simulator, n_samples, filename):
    samples = simulator.sample(n_samples)
    fig = plt.figure()
    plt.hist(samples["field"].min(axis=(1, 2)).flatten(), bins=40, color="blue", alpha=0.7)
    plt.hist(samples["field"].max(axis=(1, 2)).flatten(), bins=40, color="red", alpha=0.7)
    plt.legend(["min", "max"])
    plt.title("Distribution of Min and Max Values in the Field")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    save_fig(fig, filename)


def plot_real_vs_generated(workflow, log_std=0.0):
    n_examples = 5
    alphas = np.linspace(2, 5, n_examples)
    cmap = "Spectral"
    field_shape = FIELD_SHAPE

    def plot_distribution(shape):
        rng = np.random.default_rng(seed=1234123)
        a = rng.normal(loc=0, scale=1.0, size=shape)
        b = rng.normal(loc=0, scale=1.0, size=shape)
        return a + 1j * b

    simulator_spectra = [generate_power_spectrum(alpha, np.exp(log_std)) for alpha in alphas]
    fields_simulated = np.stack(
        [
            generate_field(plot_distribution, spectra, field_shape, unit_length=1 / (np.abs(alpha) + 1e-7))
            for (alpha, spectra) in zip(alphas, simulator_spectra)
        ],
        axis=0,
    )
    params_expanded = []
    for alpha in alphas:
        param_expanded = np.array([log_std, alpha])
        params_expanded.append(np.ones(field_shape + (2,)) * param_expanded[None, None, :])
    params_expanded = np.stack(params_expanded, axis=0)

    fields_generated = workflow.sample(
        num_samples=1, conditions={"params_expanded": params_expanded}
    )["field"][:, 0]

    fig, axs = plt.subplots(2, n_examples, figsize=(n_examples * 2, 4))
    for i, (field_gen, field_sim, alpha) in enumerate(zip(fields_generated, fields_simulated, alphas)):
        vmin = np.minimum(field_sim.min(), field_gen.min())
        vmax = np.maximum(field_sim.max(), field_gen.max())
        axs[0, i].imshow(field_sim, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0, i].set_title(rf"$\alpha={alpha:.2f}$")
        axs[1, i].imshow(field_gen[:, :, 0], cmap=cmap, vmin=vmin, vmax=vmax)

    for ax in axs.flat:
        ax.set_axis_off()
        ax.set_aspect("equal")

    plt.tight_layout()
    save_fig(fig, "32_06_real_vs_generated_fields.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny epochs/sims, just to verify the pipeline runs end-to-end")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-batches-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-minmax-original", type=int, default=1000)
    parser.add_argument("--n-minmax-unit-length", type=int, default=10000)
    parser.add_argument("--val-sims", type=int, default=100)
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.num_batches_per_epoch = 5
        args.n_minmax_original = 50
        args.n_minmax_unit_length = 50
        args.val_sims = 10

    plot_field_histogram_grid(
        original_distribution, False, "32_01_field_histogram_grid_original.png"
    )
    plot_field_histogram_grid(
        unit_length_distribution, True, "32_02_field_histogram_grid_unit_length.png"
    )

    original_simulator = build_grf_simulator()
    plot_minmax_histogram(
        original_simulator, args.n_minmax_original, "32_03_minmax_histogram_original.png"
    )

    revised_simulator = build_grf_generative_simulator()
    plot_minmax_histogram(
        revised_simulator, args.n_minmax_unit_length, "32_04_minmax_histogram_unit_length.png"
    )

    workflow = build_grf_diffusion_workflow()
    history = train_grf_diffusion_workflow(
        workflow,
        epochs=args.epochs,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
        validation_data=args.val_sims,
    )
    fig = bf.diagnostics.plots.loss(history)
    save_fig(fig, "32_05_diffusion_loss.png")

    plot_real_vs_generated(workflow)


if __name__ == "__main__":
    main()
