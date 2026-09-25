"""Gaussian Random Field (GRF) simulator subpackage. See README.md.

Both modules import bayesflow + FyeldGenerator at load time, so unlike
sir/ and lotka_volterra/, nothing is re-exported eagerly here -- import
`grf_model` or `grf_generative_model` directly so callers that never touch
GRF don't pay for those imports.
"""
