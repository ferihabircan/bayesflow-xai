# SIR

Stationary SIR (Susceptible-Infected-Recovered) epidemic model from the
BayesFlow tutorial. `sir_model.py` samples a prior over
`(lambd, mu, D, I0, psi)`, forward-simulates the outbreak, and observes
daily case counts through a negative-binomial noise model. `adapter.py`
wires this into a BayesFlow simulator + adapter for training/inference.

Target parameters: `lambd`, `mu`, `D`, `I0`, `psi`.
