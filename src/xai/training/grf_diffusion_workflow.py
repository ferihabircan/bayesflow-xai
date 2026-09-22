"""Builds and trains the BayesFlow BasicWorkflow for the Gaussian Random
Field (GRF) image-generation example (Spatial_Data_and_Parameters.html,
section 3.2): a diffusion model that generates fields conditioned on
(log_std, alpha), instead of grf_workflow.py's coupling-flow parameter
inference. Adapter and network configs are kept verbatim from the tutorial
(only the "ResidualUViT" config is actually used, per the tutorial's final
training cell; "UNet"/"UViT" are kept for parity with the example)."""

import bayesflow as bf

from xai.simulation.grf.grf_generative_model import build_grf_generative_simulator
from xai.utils.config import set_seed

CONFIGS = {
    "UNet": {
        "widths": (64, 128, 256, 512),
        "res_blocks": 2,
        "attn_stage": None,
    },
    "UViT": {
        "widths": (64, 128, 256),
        "res_blocks": 3,
        "transformer_blocks": 2,
        "transformer_dropout": 0.2,
        "transformer_width": 512,
    },
    "ResidualUViT": {
        "widths": (64, 128, 256),
        "res_blocks_up": 2,
        "res_blocks_down": 3,
        "transformer_blocks": 2,
        "transformer_dropout": 0.2,
        "transformer_width": 512,
    },
}


def build_grf_diffusion_adapter():
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("params_expanded", "inference_conditions")
        .rename("field", "inference_variables")
    )


def build_grf_diffusion_workflow(seed: int = 42):
    set_seed(seed)
    simulator = build_grf_generative_simulator()
    adapter = build_grf_diffusion_adapter()

    diffusion = bf.networks.DiffusionModel(
        subnet=bf.networks.ResidualUViT,
        subnet_kwargs=CONFIGS["ResidualUViT"],
        prediction_type="velocity",
        noise_schedule="cosine",
    )

    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        inference_network=diffusion,
        adapter=adapter,
        standardize="inference_conditions",
        initial_learning_rate=1e-4,
    )
    return workflow


def train_grf_diffusion_workflow(
    workflow,
    epochs: int = 20,
    batch_size: int = 32,
    num_batches_per_epoch: int = 1000,
    validation_data=100,
    seed: int = 42,
):
    set_seed(seed)
    history = workflow.fit_online(
        epochs=epochs,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        validation_data=validation_data,
    )
    return history
