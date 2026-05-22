import os

import numpy as np
import pandas as pd

from src.data import TARGETS


def per_trait_rmse(y_true, y_pred):
    return np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean(axis=0))


def score(name, y_true, y_pred):
    rmses = per_trait_rmse(y_true, y_pred)
    row = {"model": name, "mcrmse": float(rmses.mean())}
    for target, value in zip(TARGETS, rmses):
        row[target] = float(value)
    return row


def save_results(rows, output_dir):
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    return results