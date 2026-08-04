"""Builds and trains the BayesFlow BasicWorkflow."""

import bayesflow as bf
from keras.callbacks import EarlyStopping

from sir_xai.simulation.adapter import build_simulator, build_adapter
from sir_xai.training.networks import GRUSummaryNetwork
from sir_xai.utils.config import CONFIG


def build_workflow():
    summary_net = GRUSummaryNetwork()
    inference_net = bf.networks.CouplingFlow(depth=2, transform="spline")

    workflow = bf.BasicWorkflow(
        simulator=build_simulator(),
        adapter=build_adapter(),
        inference_network=inference_net,
        summary_network=summary_net,
        standardize=None,
    )
    return workflow, summary_net


def train_workflow(workflow, use_early_stopping: bool = True):
    training_data = workflow.simulate(CONFIG.n_train_sims)
    validation_data = workflow.simulate(CONFIG.n_val_sims)

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        )

    history = workflow.fit_offline(
        data=training_data,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    return history
