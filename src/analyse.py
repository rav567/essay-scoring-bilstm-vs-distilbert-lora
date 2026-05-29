"""Post-training analysis and visualisation utilities.

Provides a horizontal bar chart for model comparison and a qualitative
error analysis that extracts the highest-error test examples for inspection.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import TARGETS


def plot_results_bar(results_df, output_dir):
    """Save a horizontal bar chart comparing MCRMSE across all models.

    Models are sorted worst-to-best with error bars showing standard deviation
    over seeds. Saved to output_dir/results_bar.png.
    """
    order = results_df.sort_values("mcrmse", ascending=False)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.barh(order["model"], order["mcrmse"], xerr=order["mcrmse_std"],
              color="#1f3b57", capsize=4)
    axis.set_xlabel("MCRMSE (lower is better)")
    axis.set_title("model comparison (mean +/- SD over seeds)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/results_bar.png", dpi=150)
    plt.close(fig)


def error_analysis(test_df, preds_list, output_dir, top_k=5):
    """Save a CSV of the highest-error test examples for qualitative inspection.

    Identifies the top_k essays with the largest mean absolute error and
    records a 200-character excerpt alongside true and predicted scores.
    Saved to output_dir/error_examples.csv.
    """
    preds = np.mean(preds_list, axis=0)
    y_true = test_df[TARGETS].values
    errors = np.abs(preds - y_true).mean(axis=1)
    worst = np.argsort(errors)[-top_k:][::-1]
    rows = []
    for idx in worst:
        row = {
            "error": round(float(errors[idx]), 3),
            "excerpt": test_df["full_text"].iloc[idx][:200].replace("\n", " "),
        }
        for trait_idx, name in enumerate(TARGETS):
            row[f"{name}_true"] = float(y_true[idx, trait_idx])
            row[f"{name}_pred"] = round(float(preds[idx, trait_idx]), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{output_dir}/error_examples.csv", index=False)