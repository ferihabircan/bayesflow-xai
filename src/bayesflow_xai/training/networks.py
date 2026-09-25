"""BayesFlow-compatible summary network(s)."""

import keras
import bayesflow as bf


@bf.utils.serialization.serializable("custom")
class GRUSummaryNetwork(bf.networks.SummaryNetwork):
    """Compresses (batch, T, 1) case-count series into an 8-dim summary."""

    def __init__(self, hidden: int = 64, summary_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.gru = keras.layers.GRU(hidden)
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series, **kwargs):
        summary = self.gru(time_series, **kwargs)
        return self.summary_stats(summary)
