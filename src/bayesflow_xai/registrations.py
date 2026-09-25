"""Built-in plugin registrations for bayesflow_xai.registry.

Importing this module (scripts/run_workflow.py does this once, at the
top) registers the project's 4 built-in simulators (sir, lotka_volterra,
grf, gravitational_wave), 4 summary-network builders (gru, conv, transformer,
gw_paper_cnn), 1 inference
network (coupling_flow), and 4 XAI methods (integrated_gradients,
saliency_map, attention_rollout, masking).

To add your own simulator/network/xai method, write a module that imports
`bayesflow_xai.registry` and calls its `register_*` decorators (see
`bayesflow_xai.registry`'s module docstring for a worked example), then import
that module before running scripts/run_workflow.py -- or just add the
import to this file if it's meant to be a project-wide built-in.
"""

from bayesflow_xai.registry import (
    register_simulator,
    register_summary_network,
    register_inference_network,
    register_xai_method,
    SimulatorSpec,
)


# =====================================================================
# 1) SIMULATORS
# =====================================================================
@register_simulator("sir", targets=["lambd", "mu", "D", "I0", "psi"])
def _build_sir_spec() -> SimulatorSpec:
    from bayesflow_xai.simulation.dataset import build_sir_tensor_dataset

    return SimulatorSpec(
        name="sir",
        param_names=["lambd", "mu", "D", "I0", "psi"],
        input_kind="timeseries",
        build_tensor_dataset=build_sir_tensor_dataset,
        channel_names=["S", "I", "R"],
        in_channels=3,
    )


@register_simulator("lotka_volterra", targets=["theta0", "theta1", "theta2"])
def _build_lotka_volterra_spec() -> SimulatorSpec:
    from bayesflow_xai.simulation.dataset import build_lv_tensor_dataset

    return SimulatorSpec(
        name="lotka_volterra",
        param_names=["theta0", "theta1", "theta2"],
        input_kind="timeseries",
        build_tensor_dataset=build_lv_tensor_dataset,
        channel_names=["X1", "X2"],
        in_channels=2,
    )


# Target order matches grf_model.PARAM_NAMES (= the y-column order that
# build_grf_tensor_dataset actually produces) so target_idx lookups stay
# correct; log_std, alpha is equivalent to the alpha/log_std pairing
# requested in the project brief, just listed in dataset-column order.
@register_simulator("grf", targets=["log_std", "alpha"])
def _build_grf_spec() -> SimulatorSpec:
    from bayesflow_xai.methods.grf_xai import build_grf_tensor_dataset
    from bayesflow_xai.simulation.grf.grf_model import FIELD_SHAPE

    return SimulatorSpec(
        name="grf",
        param_names=["log_std", "alpha"],
        input_kind="image",
        build_tensor_dataset=build_grf_tensor_dataset,
        field_shape=FIELD_SHAPE,
    )


# PyCBC IMRPhenomPv2 alternative to the notebook's (unavailable)
# GravitationalWaveBenchmarkSimulator -- see simulation/gravitational_wave/README.md.
@register_simulator("gravitational_wave", targets=["mass1", "mass_ratio"])
def _build_gravitational_wave_spec() -> SimulatorSpec:
    from bayesflow_xai.simulation.gravitational_wave.simulator import (
        build_gw_tensor_dataset,
        PARAM_NAMES,
        CHANNEL_NAMES,
    )

    return SimulatorSpec(
        name="gravitational_wave",
        param_names=PARAM_NAMES,
        input_kind="timeseries",
        build_tensor_dataset=build_gw_tensor_dataset,
        channel_names=CHANNEL_NAMES,
        in_channels=2,
    )


# =====================================================================
# 2) SUMMARY NETWORKS
# =====================================================================
@register_summary_network("gru")
def _build_gru_summary_network(spec: SimulatorSpec, hidden: int = 64, summary_dim: int = 8):
    from bayesflow_xai.methods.generic_surrogate import GenericGRUSummaryNet

    if spec.input_kind != "timeseries":
        raise ValueError(
            f"summary_network 'gru' requires a timeseries simulator, "
            f"got '{spec.name}' (input_kind={spec.input_kind})"
        )
    return GenericGRUSummaryNet(in_channels=spec.in_channels, hidden=hidden, summary_dim=summary_dim)


@register_summary_network("conv")
def _build_conv_summary_network(spec: SimulatorSpec, widths=(8, 16, 32, 64), summary_dim: int = 6):
    from bayesflow_xai.methods.generic_surrogate import GenericCNNSummaryNet

    if spec.input_kind != "image":
        raise ValueError(
            f"summary_network 'conv' requires an image simulator, "
            f"got '{spec.name}' (input_kind={spec.input_kind})"
        )
    return GenericCNNSummaryNet(field_shape=spec.field_shape, widths=widths, summary_dim=summary_dim)


@register_summary_network("transformer")
def _build_transformer_summary_network(
    spec: SimulatorSpec, d_model: int = 32, n_heads: int = 4, n_layers: int = 2, summary_dim: int = 8
):
    from bayesflow_xai.methods.attention_rollout import TransformerSummaryNet

    if spec.input_kind != "timeseries":
        raise ValueError(
            f"summary_network 'transformer' requires a timeseries simulator, "
            f"got '{spec.name}' (input_kind={spec.input_kind})"
        )
    return TransformerSummaryNet(
        in_dim=spec.in_channels, d_model=d_model, n_heads=n_heads, n_layers=n_layers, summary_dim=summary_dim
    )


# PaperEmbedding from notebooks/4_1_grav_waves.ipynb (dilated Conv1d + SELU).
# Needs >= 2**nlayers timepoints; the gravitational_wave dataset gives 8192.
@register_summary_network("gw_paper_cnn")
def _build_gw_paper_cnn_summary_network(
    spec: SimulatorSpec, n_timepoints: int = 8192, nlayers: int = 13,
    intermediate_channels: int = 16, nfinal_channels: int = 16,
):
    from bayesflow_xai.simulation.gravitational_wave.embedding import GWPaperCNNSummaryNet

    if spec.input_kind != "timeseries":
        raise ValueError(
            f"summary_network 'gw_paper_cnn' requires a timeseries simulator, "
            f"got '{spec.name}' (input_kind={spec.input_kind})"
        )
    return GWPaperCNNSummaryNet(
        in_channels=spec.in_channels, n_timepoints=n_timepoints, nlayers=nlayers,
        intermediate_channels=intermediate_channels, nfinal_channels=nfinal_channels,
    )


# =====================================================================
# 3) INFERENCE NETWORKS  (BayesFlow-side; used by the full BasicWorkflow
#    path, e.g. training/workflow.py-style latent-space analysis)
# =====================================================================
@register_inference_network("coupling_flow")
def _build_coupling_flow(depth: int = 2, transform: str = "spline", **kwargs):
    import bayesflow as bf

    return bf.networks.CouplingFlow(depth=depth, transform=transform, **kwargs)


# =====================================================================
# 4) XAI METHODS
# =====================================================================
@register_xai_method("integrated_gradients")
def _xai_integrated_gradients(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs):
    from bayesflow_xai.methods.integrated_gradients import integrated_gradients_generic

    return integrated_gradients_generic(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs)


@register_xai_method("saliency_map")
def _xai_saliency_map(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs):
    from bayesflow_xai.methods.saliency_map import saliency_map_generic

    return saliency_map_generic(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs)


@register_xai_method("attention_rollout")
def _xai_attention_rollout(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs):
    from bayesflow_xai.methods.attention_rollout import attention_rollout_generic

    return attention_rollout_generic(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs)


@register_xai_method("masking")
def _xai_masking(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs):
    from bayesflow_xai.methods.masking import masking_importance

    return masking_importance(spec, body, head, X_val, y_val, target_idx, target_name, fig_prefix, **kwargs)
