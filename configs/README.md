# configs

`workflow_example.yaml` -- drives `scripts/run_workflow.py`: pick a
`simulator`, `summary_network`, `inference_network`, `target`, and
`xai_method` by name (see the registry in `../src/xai/registry.py`).

`default.yaml` -- human-readable mirror of `xai.utils.config.Config`,
for reference (not currently loaded by any script).
