# Lotka-Volterra

Despite the name, `sample_fn` is not the predator-prey ODE system -- it's a
theta-parametrized sinusoidal signal with additive Gaussian noise, kept
verbatim from the professor's Slack example. `build_lv_simulator()` wraps
it as a BayesFlow `LambdaSimulator` for workflow-style use.

Target parameters: `theta0`, `theta1`, `theta2`.
