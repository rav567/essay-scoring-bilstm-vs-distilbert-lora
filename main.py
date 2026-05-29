"""Entry point for the ELEC0141 DLNLP assignment.

Task: Automated Essay Scoring (AES) for English Language Learners (ELLs).
Dataset: Feedback Prize - English Language Learning (Kaggle, 2022).

Pipeline (fully automated, no user interaction required):
  1. Load and validate the ELLIPSE corpus; produce EDA plots.
  2. Stratified 80/10/10 train/val/test split.
  3. Train and evaluate models in order of complexity:
       - Mean baseline
       - TF-IDF + Ridge regression
       - Bidirectional LSTM with learned embeddings
       - DistilBERT (frozen encoder, full fine-tune, LoRA r=4/8/16)
  4. Layer-wise probing study: pretrained vs fine-tuned DistilBERT.
  5. Representation analysis: t-SNE, PCA of trait labels, cross-trait transfer.
  6. Save all metrics to outputs/results.csv and all plots to outputs/.

All experiments use fixed random seeds for reproducibility.
"""
from __future__ import annotations

import os
import random

import numpy as np

from src.analyse import error_analysis, plot_results_bar
from src.baselines import mean_baseline, tfidf_ridge
from src.data import prepare_data
from src.evaluate import save_results, score
from src.plots import plot_training_curves
from src.probe import run_probing
from src.represent import run_representation
from src.train import train_bilstm
from src.transformer_models import train_transformers

SEED = 42
SEEDS = (42, 1, 2)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"


def set_seed(seed):
    """Fix all random seeds for fully deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # pylint: disable=import-outside-toplevel
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main():
    """Run the complete AES experimental pipeline end-to-end.

    Loads data, trains all models (baselines through transformers), runs
    probing and representation studies, saves all plots and metrics to disk.
    No user interaction required; all outputs written to OUTPUT_DIR.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    train_df, val_df, test_df = prepare_data(DATA_DIR, OUTPUT_DIR, SEED)

    results = []
    histories = {}

    y_true, y_pred = mean_baseline(train_df, test_df)
    results.append(score("mean", y_true, [y_pred]))

    y_true, y_pred = tfidf_ridge(train_df, test_df, SEED)
    results.append(score("tfidf_ridge", y_true, [y_pred]))

    y_true, preds, hist = train_bilstm(train_df, val_df, test_df, SEEDS)
    results.append(score("bilstm", y_true, preds))
    histories["bilstm"] = hist

    y_true, tpreds, thist = train_transformers(train_df, val_df, test_df, SEEDS)
    for name in tpreds:
        results.append(score(name, y_true, tpreds[name]))
        histories[name] = thist[name]

    run_probing(train_df, val_df, test_df, OUTPUT_DIR)
    run_representation(train_df, val_df, test_df, OUTPUT_DIR)

    plot_training_curves(histories, OUTPUT_DIR)
    results_df = save_results(results, OUTPUT_DIR)
    plot_results_bar(results_df, OUTPUT_DIR)
    error_analysis(test_df, tpreds["finetuned"], OUTPUT_DIR)

    for row in results:
        print(row["model"], round(row["mcrmse"], 4), "+/-", round(row["mcrmse_std"], 4))


if __name__ == "__main__":
    main()
