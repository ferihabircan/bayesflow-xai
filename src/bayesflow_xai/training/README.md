# training

Builds and trains BayesFlow `BasicWorkflow` instances: `workflow.py` (SIR),
`grf_workflow.py` (GRF parameter inference), `grf_diffusion_workflow.py`
(GRF field generation via diffusion). `networks.py` holds the
BayesFlow-native `GRUSummaryNetwork` used by the SIR workflow.

These are the "full BayesFlow pipeline" path (posterior inference,
latent-space analysis); the XAI attribution methods in `../methods/` instead
train small plain-PyTorch surrogates (via `../methods/generic_surrogate.py`)
that Captum can differentiate through directly.
