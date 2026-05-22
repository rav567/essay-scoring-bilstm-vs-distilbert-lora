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
    path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(path):
        _download_kaggle(data_dir)
    return pd.read_csv(path)


def validate(df):
    required = ["text_id", "full_text"] + TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=["full_text"] + TARGETS).reset_index(drop=True)
    for target in TARGETS:
        if not df[target].between(1.0, 5.0).all():
            raise ValueError(f"{target} scores out of range")
    return df


def make_splits(df, seed):
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
    df = validate(load_raw(data_dir))
    analyse(df, output_dir)
    return make_splits(df, seed)