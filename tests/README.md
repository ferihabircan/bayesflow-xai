# tests

Unit tests, one file per module under test (`test_sir_model.py` covers
`simulation/sir/sir_model.py`, etc.). Run with `pytest` from the project
root. Pure-numpy modules (simulators) are tested without needing
bayesflow/torch to load.
