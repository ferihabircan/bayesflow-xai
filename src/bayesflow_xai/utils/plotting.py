"""Shared plotting helpers: consistent figure saving + non-overlapping windows."""

import os
import matplotlib.pyplot as plt

from bayesflow_xai.utils.config import CONFIG


def show_and_save(fig, name: str, window_title: str | None = None, block: bool = False):
    """Titles the figure window, saves a PNG to outputs/figures/, and shows
    it without blocking (so multiple figures can stay open side by side)."""
    if window_title and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    os.makedirs(CONFIG.figures_dir, exist_ok=True)
    fig.savefig(os.path.join(CONFIG.figures_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.show(block=block)
    return fig
