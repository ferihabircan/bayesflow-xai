# scripts

Thin CLI entry points. `run_workflow.py --config <file.yaml>` is the
general, config-driven one (see `../configs/workflow_example.yaml`) --
start there for anything new. The rest are the original per-pipeline
scripts, kept for their specific figure sets:

| script | what it runs |
|---|---|
| `run_workflow.py` | config-driven: any registered simulator x summary network x XAI method |
| `run_diagnostics.py` | SIR BasicWorkflow training + loss/recovery/calibration plots |
| `run_xai.py` | SIR latent space + Integrated Gradients + Attention Rollout |
| `run_lv_xai.py` | Lotka-Volterra channel/time importance stats |
| `run_grf_xai.py` | GRF pixel-level Integrated Gradients |
| `run_grf_tutorial_31.py` / `run_grf_tutorial_32.py` | full GRF tutorial reproductions (sections 3.1 / 3.2) |
| `run_all.py` | diagnostics + all 3 SIR XAI analyses in one run |
