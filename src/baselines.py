"""Baseline models for the ELL automated essay scoring task.

Provides two non-neural baselines:
  - Mean baseline: predicts the training-set mean for every essay.
  - TF-IDF + Ridge: bag-of-words bigram features with ridge regression.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.data import TARGETS


def mean_baseline(train_df, eval_df):
    """Predict the training-set mean score for every essay (lower bound).

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays of shape (n, 6).
    """
    means = train_df[TARGETS].mean().values
    y_true = eval_df[TARGETS].values
    y_pred = np.tile(means, (len(eval_df), 1))
    return y_true, y_pred


def tfidf_ridge(train_df, eval_df, seed):
    """TF-IDF bigram features with multi-output Ridge regression.

    Uses unigrams and bigrams (up to 20k features, sublinear TF) fitted
    independently per target via scikit-learn's multi-output wrapper.

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays of shape (n, 6),
        predictions clipped to [1.0, 5.0].
    """
    model = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
        Ridge(alpha=1.0, random_state=seed),
    )
    model.fit(train_df["full_text"], train_df[TARGETS].values)
    y_pred = np.clip(model.predict(eval_df["full_text"]), 1.0, 5.0)
    y_true = eval_df[TARGETS].values
    return y_true, y_pred