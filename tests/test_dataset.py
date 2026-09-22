from xai.simulation.dataset import build_sir_tensor_dataset
from xai.utils.config import CONFIG


def test_dataset_shapes():
    X, y = build_sir_tensor_dataset(n_sims=5)
    assert X.shape == (5, CONFIG.horizon_days, 3)
    assert y.shape == (5, 5)
