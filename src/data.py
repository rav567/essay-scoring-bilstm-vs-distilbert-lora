"""Data loading, validation, splitting and exploratory analysis for the ELLIPSE corpus.

Downloads the Feedback Prize ELL dataset from Kaggle if not already present,
validates schema and score ranges, produces a stratified 80/10/10 split, and
saves exploratory plots (score distributions, trait correlation, essay lengths).
"""
import os
import subprocess
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

TARGETS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
COMPETITION = "feedback-prize-english-language-learning"


def _download_kaggle(data_dir):
    """Download the competition dataset via the Kaggle CLI and extract the zip.

    Raises:
        RuntimeError: If the Kaggle CLI is not found or the download fails.
    """
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
            f.write(token)
        os.chmod(os.path.join(kaggle_dir, "access_token"), 0o600)
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", data_dir],
            check=True,
        )
        with zipfile.ZipFile(os.path.join(data_dir, COMPETITION + ".zip")) as archive:
            archive.extractall(data_dir)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "train.csv missing and Kaggle download failed; "
            "place train.csv in the data/ folder manually."
        ) from exc


def load_raw(data_dir):
    """Load the raw training CSV, downloading it first if absent.

    Returns:
        pandas.DataFrame with all essays and target scores.
    """
    path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(path):
        _download_kaggle(data_dir)
    return pd.read_csv(path)


def validate(df):
    """Validate schema and score ranges; drop rows with missing values.

    Returns:
        Cleaned DataFrame with guaranteed schema and valid score ranges.

    Raises:
        ValueError: If required columns are missing or scores are out of [1, 5].
    """
    required = ["text_id", "full_text"] + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=["full_text"] + TARGETS).reset_index(drop=True)
    for target in TARGETS:
        if not df[target].between(1.0, 5.0).all():
            raise ValueError(f"{target} scores out of range")
    return df


def make_splits(df, seed):
    """Create stratified 80/10/10 train/val/test splits with no data leakage.

    Stratification is based on binned mean score across all six traits to
    ensure each split has a representative score distribution.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    strat = pd.cut(df[TARGETS].mean(axis=1), bins=5, labels=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=strat
    )
    temp_strat = pd.cut(temp_df[TARGETS].mean(axis=1), bins=5, labels=False)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_strat
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def analyse(df, output_dir):
    """Produce and save EDA plots for the full dataset.

    Saves three plots to output_dir:
      - score_distributions.png: histogram per trait
      - trait_correlation.png: Pearson correlation heatmap across traits
      - essay_lengths.png: distribution of word counts per essay
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for axis, target in zip(axes.ravel(), TARGETS):
        sns.histplot(df[target], bins=9, ax=axis)
        axis.set_title(target)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[TARGETS].corr(), annot=True, fmt=".2f", cmap="viridis", ax=axis)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trait_correlation.png"), dpi=150)
    plt.close(fig)

    lengths = df["full_text"].str.split().map(len)
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.histplot(lengths, bins=40, ax=axis)
    axis.set_xlabel("words per essay")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "essay_lengths.png"), dpi=150)
    plt.close(fig)


def prepare_data(data_dir, output_dir, seed):
    """Load, validate, analyse and split the dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = validate(load_raw(data_dir))
    analyse(df, output_dir)
    return make_splits(df, seed)
