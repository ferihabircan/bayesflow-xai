"""Summary network for the gravitational-wave simulator.

`PaperEmbedding` and `init_weights` are copied VERBATIM from cell 7b3b7d22
of notebooks/4_1_grav_waves.ipynb (itself reproducing
the trust-crisis-in-simulation-based-inference GW ratio estimator).

With kernel_size=2 and dilations 1, 2, ..., 2**(nlayers-1), the unpadded
dilated stack shrinks the sequence by 2**nlayers - 1 samples, so the
default nlayers=13 needs at least 8192 = 2**13 input samples and maps
exactly 8192 samples to 1 -> output (batch, nfinal_channels) = (batch, 16).
simulator.py's dataset window (3.5 s + 0.5 s at 2048 Hz) is sized for this.

`GWPaperCNNSummaryNet` (below the verbatim block) is the project-side
adapter only: it transposes the registry's (batch, T, channels) layout to
the (batch, channels, T) Conv1d layout, applies `init_weights` as the
notebook does, and exposes `summary_dim` for train_generic_surrogate.
"""

import torch


# ---------------------------------------------------------------------
# Verbatim from notebook cell 7b3b7d22 -- do not edit.
# ---------------------------------------------------------------------
class PaperEmbedding(torch.nn.Sequential):
    def __init__(
        self,
        input_dim=(1, 2, 8192),
        nlayers=13,
        kernel_size=2,
        intermediate_channels=16,
        nfinal_channels=16
    ):
        """Reproducing https://github.com/montefiore-ai/trust-crisis-in-simulation-based-inference/blob/9dca4a508a19c514422a9b177d4fcf1dad6ea693/workflows/coverage_simulations_and_bias_reduction/gw/ratio_estimation.py#L129"""
        super(PaperEmbedding, self).__init__()
        self.add_module(
                f"conv1d::00-1x1",
                torch.nn.Conv1d(
                   input_dim[1], intermediate_channels, kernel_size=1
                ),
            )
        layer_idx = 1
        for i in range(nlayers):
            dilation = 2**i
            self.add_module(
                f"conv1d::{layer_idx:02.0f}-dil{dilation:02.0f}",
                torch.nn.Conv1d(
                    intermediate_channels, intermediate_channels, kernel_size=2, dilation=dilation
                ),
            )
            layer_idx += 1
            self.add_module(f"selu::{layer_idx:02.0f}", torch.nn.SELU())
            layer_idx += 1

        if intermediate_channels != nfinal_channels:
            self.add_module(
                f"conv1d::{layer_idx:02.0f}-1x1",
                torch.nn.Conv1d(
                    intermediate_channels, nfinal_channels, kernel_size=1
                ),
            )
        self.add_module(f"flatten::{layer_idx:02.0f}", torch.nn.Flatten())

    def forward(self, x):
        return super().forward(x)


def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
# ---------------------------------------------------------------------
# End of verbatim block.
# ---------------------------------------------------------------------


class GWPaperCNNSummaryNet(torch.nn.Module):
    """(batch, T, channels) -> (batch, summary_dim) wrapper around
    PaperEmbedding for the project's registry/surrogate workflow."""

    def __init__(self, in_channels=2, n_timepoints=8192, nlayers=13, intermediate_channels=16, nfinal_channels=16):
        super().__init__()
        out_len = n_timepoints - (2**nlayers - 1)
        if out_len < 1:
            raise ValueError(
                f"PaperEmbedding with nlayers={nlayers} needs >= {2**nlayers} timepoints, got {n_timepoints}"
            )
        self.embedding = PaperEmbedding(
            input_dim=(1, in_channels, n_timepoints),
            nlayers=nlayers,
            intermediate_channels=intermediate_channels,
            nfinal_channels=nfinal_channels,
        )
        self.embedding.apply(init_weights)
        self.summary_dim = nfinal_channels * out_len

    def forward(self, x):
        return self.embedding(x.transpose(1, 2))
