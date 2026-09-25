# simulation

Simulator plugins, one subfolder per model: `sir/`, `lotka_volterra/`,
`grf/`. Each is self-contained (own README, own `__init__.py`) and
registered by name in `bayesflow_xai.registrations` (see `../registry.py`).
`dataset.py` here builds the (X, y) tensor datasets the XAI surrogates
train on, shared across models.

To add a new simulator, copy one of the existing subfolders as a template
and register it -- see `bayesflow_xai/registry.py`'s docstring.
