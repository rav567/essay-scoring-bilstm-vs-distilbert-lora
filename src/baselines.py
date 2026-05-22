import numpy as np

from src.data import TARGETS


def mean_baseline(train_df, eval_df):
    means = train_df[TARGETS].mean().values
    y_true = eval_df[TARGETS].values
    y_pred = np.tile(means, (len(eval_df), 1))
    return y_true, y_pred