"""Builds and trains the BayesFlow BasicWorkflow for the Gaussian Random
Field (GRF) parameter-inference example -- same shape as
training/workflow.py's SIR pipeline, but wired to grf_model's
simulator/prior and a convolutional summary network instead of the SIR
GRU/adapter pipeline."""

import bayesflow as bf
from keras.callbacks import EarlyStopping

from sir_xai.simulation.grf_model import build_grf_simulator, PARAM_NAMES
from sir_xai.utils.config import set_seed


def build_grf_summary_network():
    return bf.networks.ConvolutionalNetwork(
        summary_dim=6,
        widths=(8, 16, 32, 64),
        blocks_per_stage=1,
        down_mode="max_pool",
        pool_head="flatten",
        norm="group",
        groups=1,
    )


def build_grf_workflow(seed: int = 42):
    set_seed(seed)
    simulator = build_grf_simulator()
    summary_network = build_grf_summary_network()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        summary_network=summary_network,
        inference_network="coupling_flow",
        inference_variables=PARAM_NAMES,
        summary_variables=["field"],
        standardize="all",
    )
    return workflow, summary_network


def train_grf_workflow(
    workflow,
    n_train_sims: int = 5000,
    n_val_sims: int = 300,
    epochs: int = 50,
    batch_size: int = 32,
    use_early_stopping: bool = True,
    seed: int = 42,
):
    set_seed(seed)
    training_data = workflow.simulate(n_train_sims)
    validation_data = workflow.simulate(n_val_sims)

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        )

    history = workflow.fit_offline(
        data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return history
