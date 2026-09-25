"""XAI Part 1: UMAP / t-SNE projection of the summary network's latent space."""

import keras
import numpy as np
import matplotlib.pyplot as plt

from bayesflow_xai.simulation.sir.adapter import build_simulator, build_adapter
from bayesflow_xai.utils.plotting import show_and_save


def latent_space_analysis(summary_net, n_sims: int, color_by: str = "lambd"):
    import umap
    from sklearn.manifold import TSNE

    simulator = build_simulator()
    adapter = build_adapter()

    sims = simulator.sample(n_sims)
    adapted = adapter(sims)
    summary_vars = adapted["summary_variables"]

    embeddings = summary_net(keras.ops.convert_to_tensor(summary_vars))
    embeddings = keras.ops.convert_to_numpy(embeddings)
    color_vals = np.asarray(sims[color_by]).squeeze()

    emb_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=2026).fit_transform(embeddings)
    emb_tsne = TSNE(n_components=2, perplexity=30, random_state=2026).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, emb, title in zip(axes, [emb_umap, emb_tsne], ["UMAP", "t-SNE"]):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_vals, cmap="viridis", s=12, alpha=0.8)
        ax.set_title(f"{title} of summary-network latent space\n(colored by {color_by})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=axes, label=color_by)

    show_and_save(fig, "xai_01_latent_space", "XAI 1: Latent Space (UMAP/t-SNE)")
    return fig, embeddings, color_vals
