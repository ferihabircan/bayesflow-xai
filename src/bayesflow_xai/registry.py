"""Plugin/registry system for simulators, summary networks, inference
networks, and XAI methods.

Anything registered here becomes selectable purely by name from a YAML
config (see configs/workflow_example.yaml + scripts/run_workflow.py) --
no changes to the workflow runner are needed to add a new option.

The registries are populated by *importing* the module that calls the
decorators below -- `bayesflow_xai.registrations` does this for the project's
4 built-in simulators, 4 summary networks, 1 inference network, and 4 XAI
methods. `scripts/run_workflow.py` imports it before doing any lookup.

Adding your own simulator
--------------------------
    # my_lab/my_simulator.py
    from bayesflow_xai.registry import register_simulator, SimulatorSpec

    @register_simulator("my_sim", targets=["theta"])
    def build_my_sim_spec() -> SimulatorSpec:
        from my_lab.my_dataset import build_my_tensor_dataset
        return SimulatorSpec(
            name="my_sim",
            param_names=["theta"],           # must match build_tensor_dataset's y columns
            input_kind="timeseries",         # or "image"
            build_tensor_dataset=build_my_tensor_dataset,  # (n_sims) -> (X, y) torch tensors
            channel_names=["x"],             # timeseries only
            in_channels=1,                   # timeseries only
        )

Then import `my_lab.my_simulator` once (e.g. at the top of
scripts/run_workflow.py, or in your own copy of it) before running with
`simulator: my_sim` in a config -- the decorator only needs to execute
once for the name to be found.

Adding your own summary network / inference network / XAI method follows
the same pattern with `register_summary_network`, `register_inference_network`,
`register_xai_method` -- see `bayesflow_xai.registrations` for worked examples of
each.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SimulatorSpec:
    """Everything the config-driven workflow needs to know about a
    registered simulator, independent of which summary network / XAI
    method ends up applied to it."""

    name: str
    param_names: list[str]
    input_kind: str  # "timeseries" (T, channels) or "image" (1, H, W)
    build_tensor_dataset: Callable[[int], tuple]  # (n_sims) -> (X, y) torch tensors

    # timeseries-only
    channel_names: Optional[list[str]] = None
    in_channels: Optional[int] = None

    # image-only
    field_shape: Optional[tuple[int, int]] = None


SIMULATORS: dict[str, Callable[[], SimulatorSpec]] = {}
SIMULATOR_TARGETS: dict[str, list[str]] = {}
SUMMARY_NETWORKS: dict[str, Callable] = {}
INFERENCE_NETWORKS: dict[str, Callable] = {}
XAI_METHODS: dict[str, Callable] = {}


def register_simulator(name: str, targets: Optional[list[str]] = None):
    """Registers a zero-arg `SimulatorSpec` builder under `name`.

    `targets` is the simulator's inferable parameter names (e.g.
    ["lambd", "mu", "D", "I0", "psi"] for SIR), stored separately in
    SIMULATOR_TARGETS so callers (like scripts/run_workflow.py) can
    validate a config's `target:` field and show valid choices on error
    without having to build the (possibly expensive) full spec first.
    """

    def decorator(fn):
        SIMULATORS[name] = fn
        if targets is not None:
            SIMULATOR_TARGETS[name] = list(targets)
        return fn

    return decorator


def register_summary_network(name: str):
    """Registers a summary-network builder under `name`. Called as
    `builder(spec, **kwargs) -> nn.Module`, where `spec` is the
    `SimulatorSpec` of whichever simulator it's being attached to (so the
    builder knows the input shape)."""

    def decorator(fn):
        SUMMARY_NETWORKS[name] = fn
        return fn

    return decorator


def register_inference_network(name: str):
    """Registers a BayesFlow inference-network builder under `name`.
    Called as `builder(**kwargs) -> bf.networks.InferenceNetwork`."""

    def decorator(fn):
        INFERENCE_NETWORKS[name] = fn
        return fn

    return decorator


def register_xai_method(name: str):
    """Registers an XAI method under `name`. Called as
    `method(spec, body, head, X_val, y_val, target_idx, target_name,
    fig_prefix, **kwargs)`."""

    def decorator(fn):
        XAI_METHODS[name] = fn
        return fn

    return decorator
