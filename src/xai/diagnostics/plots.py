"""Wraps the 3 requested BayesFlow health-check diagnostics: loss
trajectory, parameter recovery, and SBC calibration histogram."""

import bayesflow as bf

from xai.utils.config import CONFIG
from xai.utils.plotting import show_and_save


def plot_loss(history):
    fig = bf.diagnostics.plots.loss(history)
    return show_and_save(fig, "01_loss_trajectory", "1. Loss Trajectory")


def plot_recovery(samples, test_sims):
    fig = bf.diagnostics.plots.recovery(samples, test_sims)
    return show_and_save(fig, "02_parameter_recovery", "2. Parameter Recovery")


def plot_calibration_histogram(samples, test_sims):
    fig = bf.diagnostics.plots.calibration_histogram(samples, test_sims)
    return show_and_save(fig, "03_sbc_calibration", "3. SBC Calibration Histogram")


def run_all_diagnostics(workflow, history):
    """Runs and displays all 3 diagnostics; returns figures + the shared
    test_sims/samples so callers can reuse them (e.g. for XAI)."""
    fig_loss = plot_loss(history)

    test_sims = workflow.simulate(CONFIG.n_diag_datasets)
    samples = workflow.sample(
        conditions=test_sims, num_samples=CONFIG.n_diag_samples, batch_size=50
    )

    fig_recovery = plot_recovery(samples, test_sims)
    fig_calibration = plot_calibration_histogram(samples, test_sims)

    return {
        "fig_loss": fig_loss,
        "fig_recovery": fig_recovery,
        "fig_calibration": fig_calibration,
        "test_sims": test_sims,
        "samples": samples,
    }
