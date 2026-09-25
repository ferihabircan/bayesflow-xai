# GRF (Gaussian Random Field)

Two scenarios from BayesFlow's `Spatial_Data_and_Parameters.html` tutorial,
kept verbatim from the example. `grf_model.py` (section 3.1) treats
`(log_std, alpha)` as the inference target given an observed field.
`grf_generative_model.py` (section 3.2) flips this: the field itself is the
inference target, conditioned pixel-wise on `(log_std, alpha)`, for the
diffusion-model workflow.

Target parameters: `log_std`, `alpha`.
