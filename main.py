from __future__ import annotations
from src.data import prepare_data
from src.baselines import mean_baseline, tfidf_ridge
from src.evaluate import score, save_results
from src.plots import plot_training_curves
from src.train import train_bilstm

import os
import random

import numpy as np

SEED = 42
SEEDS = (42, 1, 2)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    
    """
    This function must execute the complete experimental workflow developed
    for the selected computer vision competition and research hypothesis.

    The automated grading system will call this function. Therefore:
    - The function signature must not be changed.
    - It must not require any user input.
    - It must run deterministically (fixed random seeds).
    - All outputs (metrics, logs, plots) must be saved to disk.

    The workflow should include:

        1. Dataset loading and preparation
           - Download or load the competition dataset
           - Apply preprocessing and data augmentation (if applicable)
           - Create training / validation / test splits

        2. Model construction
           - Build the baseline model
           - Build the proposed model(s) used to test the hypothesis

        3. Training
           - Train model(s) using defined hyperparameters
           - Log training and validation performance

        4. Evaluation
           - Evaluate on validation/test data
           - Compute relevant metrics (e.g., accuracy, F1, etc.)
           - Compare models if testing a hypothesis

        5. Analysis and visualisation
           - Generate and save plots used in the report
           - Save final metrics to disk (e.g., JSON/CSV)

    The purpose of this function is to reproduce all experimental evidence
    presented in the report in a fully automated and reproducible manner.

    After implementing this method, replace this docstring with a concise
    description of what your main function does.
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

    plot_training_curves(histories, OUTPUT_DIR)
    save_results(results, OUTPUT_DIR)
    for row in results:
        print(row["model"], round(row["mcrmse"], 4), "+/-", round(row["mcrmse_std"], 4))



if __name__ == "__main__":
    main()