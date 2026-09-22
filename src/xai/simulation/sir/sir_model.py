"""Prior, ODE simulator, and negative-binomial observation model for the
stationary SIR-like process (from the BayesFlow tutorial). No ML
dependencies here on purpose: this module should be importable and testable
in isolation from bayesflow/keras/torch."""

import numpy as np

from xai.utils.config import CONFIG

RNG = np.random.default_rng(CONFIG.seed)


def prior() -> dict:
    return {
        "lambd": RNG.lognormal(mean=np.log(0.4), sigma=0.5),
        "mu": RNG.lognormal(mean=np.log(1 / 8), sigma=0.2),
        "D": RNG.lognormal(mean=np.log(8), sigma=0.2),
        "I0": RNG.gamma(shape=2, scale=20),
        "psi": RNG.exponential(5),
    }


def _convert_params(mu: np.ndarray, phi: float):
    r = phi
    var = mu + 1 / r * mu ** 2
    p = (var - mu) / var
    return r, 1 - p


def stationary_SIR(
    lambd: float,
    mu: float,
    D: float,
    I0: float,
    psi: float,
    N: float = CONFIG.population_size,
    T: int = CONFIG.horizon_days,
    eps: float = 1e-5,
    return_full: bool = False,
) -> dict:
    """Forward-simulates the stationary SIR model. Set return_full=True to
    also get the normalized S(t)/I(t)/R(t) trajectories (used by the XAI
    surrogate models), not just the reported case counts."""
    I0c = np.ceil(I0)
    Dc = int(round(D))

    S, I, R = [N - I0c], [I0c], [0.0]
    C = [I0c]

    for _ in range(1, T + Dc):
        I_new = lambd * (I[-1] * S[-1] / N)
        S.append(S[-1] - I_new)
        I.append(np.clip(I[-1] + I_new - mu * I[-1], 0.0, N))
        R.append(np.clip(R[-1] + mu * I[-1], 0.0, N))
        C.append(I_new)

    reparam = _convert_params(np.clip(np.array(C[Dc:]), 0, N) + eps, psi)
    C_obs = RNG.negative_binomial(reparam[0], reparam[1])

    out = {"cases": C_obs}
    if return_full:
        out["S"] = np.array(S[Dc:]) / N
        out["I"] = np.array(I[Dc:]) / N
        out["R"] = np.array(R[Dc:]) / N
    return out
