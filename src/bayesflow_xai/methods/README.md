# methods

Explainability methods and the surrogate models they attribute against.

| file | method | plain-language summary |
|---|---|---|
| `integrated_gradients.py` | Integrated Gradients | Walks a straight path from a zero baseline to the real input, accumulating gradients along the way, to attribute the output to each input value. |
| `saliency_map.py` | Saliency Map | Raw gradient of the output w.r.t. the input -- the simplest, cheapest XAI method, no baseline/path. |
| `attention_rollout.py` | Attention Rollout | Multiplies a transformer's per-layer attention weights (mixed with a residual/identity term) to trace how information flows from each input step to the output. |
| `masking.py` | Masking / Occlusion | Zeros out part of the input (a time step, or an image patch) and measures how much the prediction moves -- no gradients needed at all. |

`integrated_gradients.py` and `attention_rollout.py` each contain both the
original SIR-specific analysis (used by `scripts/run_xai.py`) and a
simulator-agnostic `*_generic()` function used by the registry-driven
`scripts/run_workflow.py`. `surrogate_models.py` / `grf_xai.py` hold the
per-simulator surrogate nets used by the original scripts;
`generic_surrogate.py` holds the shared, registry-driven equivalents.
`latent_space.py` is a 5th, BayesFlow-specific analysis (UMAP/t-SNE over
the summary network's latent space), not part of the 4-method XAI registry
since it doesn't produce a per-input attribution.
