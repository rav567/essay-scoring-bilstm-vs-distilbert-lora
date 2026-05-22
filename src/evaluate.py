import os

import numpy as np
import pandas as pd

from src.data import TARGETS


def per_trait_rmse(y_true, y_pred):
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean(axis=0))


def score(name, y_true, preds_list):
    mcrmses = [float(per_trait_rmse(y_true, p).mean()) for p in preds_list]
    per_trait = np.mean([per_trait_rmse(y_true, p) for p in preds_list], axis=0)
    row = {
        "model": name,
        "mcrmse": float(np.mean(mcrmses)),
        "mcrmse_std": float(np.std(mcrmses)),
    }
    for target, value in zip(TARGETS, per_trait):
        row[target] = float(value)
    return row


def save_results(rows, output_dir):
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    return results