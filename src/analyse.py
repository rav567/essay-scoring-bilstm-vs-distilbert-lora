import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import TARGETS


def plot_results_bar(results_df, output_dir):
    order = results_df.sort_values("mcrmse", ascending=False)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.barh(order["model"], order["mcrmse"], xerr=order["mcrmse_std"],
              color="#1f3b57", capsize=4)
    axis.set_xlabel("MCRMSE (lower is better)")
    axis.set_title("model comparison (mean ± SD over seeds)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/results_bar.png", dpi=150)
    plt.close(fig)


def error_analysis(test_df, preds_list, output_dir, k=5):
    preds = np.mean(preds_list, axis=0)
    y_true = test_df[TARGETS].values
    errors = np.abs(preds - y_true).mean(axis=1)
    worst = np.argsort(errors)[-k:][::-1]
    rows = []
    for i in worst:
        row = {
            "error": round(float(errors[i]), 3),
            "excerpt": test_df["full_text"].iloc[i][:200].replace("\n", " "),
        }
        for t, name in enumerate(TARGETS):
            row[f"{name}_true"] = float(y_true[i, t])
            row[f"{name}_pred"] = round(float(preds[i, t]), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{output_dir}/error_examples.csv", index=False)