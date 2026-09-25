"""Simulator plugins, one subpackage per model:

    simulation/sir/              SIR epidemic model (sir_model.py, adapter.py)
    simulation/lotka_volterra/   Lotka-Volterra-style example (lotka_volterra_model.py)
    simulation/grf/              Gaussian Random Field example (grf_model.py, grf_generative_model.py)
    simulation/gravitational_wave/  original GravitationalWaveBenchmarkSimulator (sbi-practical-guide) + PaperEmbedding

`dataset.py` (this package, top level) builds the tensor datasets shared by
the XAI surrogate models across simulators.

Backward compatibility: the old flat import paths
(`bayesflow_xai.simulation.sir_model`, `bayesflow_xai.simulation.adapter`,
`bayesflow_xai.simulation.lotka_volterra_model`, `bayesflow_xai.simulation.grf_model`,
`bayesflow_xai.simulation.grf_generative_model`) still work -- each is a thin
shim re-exporting from its new subpackage location.

To register a new simulator, see `bayesflow_xai.registry` and add a subpackage
here following the same pattern (own folder, `__init__.py`, README.md).
"""
