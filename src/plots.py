import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _panel(axis, runs, color, ylabel, mark_best=False):
    curves = np.array(runs)
    epochs = np.arange(1, curves.shape[1] + 1)
    for row in curves:
        axis.plot(epochs, row, color=color, alpha=0.15, linewidth=0.8)
    mean, sd = curves.mean(axis=0), curves.std(axis=0)
    axis.plot(epochs, mean, color=color, linewidth=2, label="mean")
    axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.2, label="±1 SD")
    if mark_best:
        best = int(np.argmin(mean)) + 1
        axis.axvline(best, color="black", linestyle=":", linewidth=1)
        axis.annotate(f"best (epoch {best})", xy=(best, mean.min()),
                      xytext=(6, 6), textcoords="offset points", fontsize=8)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.legend()


def plot_training_curves(histories, output_dir):
    rows = len(histories)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
    for idx, (name, runs) in enumerate(histories.items()):
        _panel(axes[idx][0], [r["train_loss"] for r in runs], "#1f3b57", "training loss")
        _panel(axes[idx][1], [r["val_loss"] for r in runs], "#8c1d1d",
               "validation loss", mark_best=True)
        axes[idx][0].set_title(f"{name}: training loss")
        axes[idx][1].set_title(f"{name}: validation loss")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close(fig)


def plot_probing_heatmap(results, output_dir, targets):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False)
    layers = next(iter(results.values())).shape[0]
    for axis, (name, scores) in zip(axes[0], results.items()):
        sns.heatmap(scores, ax=axis, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
                    xticklabels=targets, yticklabels=[f"L{i}" for i in range(layers)])
        axis.set_title(f"{name}: probe correlation")
        axis.set_xlabel("trait")
        axis.set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/probing_heatmap.png", dpi=150)
    plt.close(fig)