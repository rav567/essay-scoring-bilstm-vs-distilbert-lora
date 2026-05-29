"""Representation analysis for fine-tuned DistilBERT essay embeddings.

Produces three plots using the last hidden layer of a fine-tuned encoder:
  - tsne_embeddings.png: 2-D t-SNE projection coloured by mean essay score.
  - pca_traits.png: PCA scree plot of the six-dimensional label space.
  - cross_trait_transfer.png: Heatmap of cross-trait probe transfer.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from src.data import TARGETS
from src.transformer_models import MODEL_NAME, _encode, train_for_probing
from src.probe import _layer_embeddings


def _pearson(vec_a, vec_b):
    """Compute Pearson correlation, returning 0.0 for NaN results.

    Returns:
        Pearson r as a float; 0.0 if the result is NaN.
    """
    corr = np.corrcoef(vec_a, vec_b)[0, 1]
    return 0.0 if np.isnan(corr) else corr


def _plot_tsne(emb, scores, output_dir):
    """Save a 2-D t-SNE scatter plot of essay embeddings coloured by score.
    """
    coords = TSNE(
        n_components=2, random_state=42, init="pca",
        perplexity=min(30, len(emb) - 1),
    ).fit_transform(emb)
    fig, axis = plt.subplots(figsize=(6, 5))
    dots = axis.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis", s=12)
    fig.colorbar(dots, ax=axis, label="mean score")
    axis.set_title("essay embeddings (t-SNE)")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{output_dir}/tsne_embeddings.png", dpi=150)
    plt.close(fig)


def _plot_pca(labels, output_dir):
    """Save a PCA scree plot of the six-dimensional trait label space.
    """
    var = PCA().fit(labels).explained_variance_ratio_
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(range(1, len(var) + 1), var, color="#1f3b57")
    axis.set_xlabel("principal component")
    axis.set_ylabel("variance explained")
    axis.set_title(f"PC1 explains {var[0] * 100:.0f}% of trait variance")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/pca_traits.png", dpi=150)
    plt.close(fig)


def _plot_transfer(train_emb, test_emb, y_train, y_test, output_dir):
    """Save a cross-trait probe-transfer heatmap.

    For each source trait, a Ridge probe is trained on last-layer embeddings
    and its predictions are correlated against every target trait.
    """
    size = len(TARGETS)
    matrix = np.zeros((size, size))
    for src in range(size):
        ridge = Ridge(alpha=1.0).fit(train_emb, y_train[:, src])
        pred = ridge.predict(test_emb)
        for tgt in range(size):
            matrix[src, tgt] = _pearson(pred, y_test[:, tgt])
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1,
                xticklabels=TARGETS, yticklabels=TARGETS, ax=axis)
    axis.set_xlabel("scored on trait")
    axis.set_ylabel("probe trained on trait")
    axis.set_title("cross-trait probe transfer")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/cross_trait_transfer.png", dpi=150)
    plt.close(fig)


def run_representation(train_df, val_df, test_df, output_dir, seed=42, max_len=256):
    """Run the full representation analysis and save all plots.

    Fine-tunes a DistilBERT encoder, extracts last-layer embeddings, then
    produces t-SNE, PCA and cross-trait transfer plots. On CPU, data is
    subsampled to keep runtime manageable.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        train_df = train_df.sample(n=min(400, len(train_df)), random_state=seed)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, train_mask = _encode(tokenizer, train_df["full_text"], max_len)
    test_ids, test_mask = _encode(tokenizer, test_df["full_text"], max_len)
    y_train = train_df[TARGETS].values
    y_test = test_df[TARGETS].values

    encoder = train_for_probing(train_df, val_df, seed=seed).encoder.to(device)
    last_train = _layer_embeddings(encoder, train_ids, train_mask, device)[-1]
    last_test = _layer_embeddings(encoder, test_ids, test_mask, device)[-1]

    _plot_tsne(last_test, y_test.mean(axis=1), output_dir)
    _plot_pca(np.vstack([y_train, y_test]), output_dir)
    _plot_transfer(last_train, last_test, y_train, y_test, output_dir)