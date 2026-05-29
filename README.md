# ELEC0141 DLNLP Assignment: Automated Essay Scoring for English Language Learners

**Task:** Predict six analytic writing proficiency scores (cohesion, syntax, vocabulary, phraseology, grammar, conventions) for argumentative essays written by 8th–12th grade English Language Learners.

**Dataset:** [Feedback Prize - English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) (Kaggle, 2022). Scores range from 1.0 to 5.0. Metric: MCRMSE (mean column-wise root mean squared error).

---

## Repository Structure

```
.
├── main.py                   # Single entry point, runs the full pipeline
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip dependencies (mirrors environment.yml)
├── .pylintrc                 # Pylint configuration
├── data/
│   └── train.csv             # ELLIPSE corpus (auto-downloaded via Kaggle API if absent)
├── outputs/                  # All plots and metrics written here by main.py
│   ├── results.csv           # MCRMSE scores for all models
│   ├── score_distributions.png
│   ├── trait_correlation.png
│   ├── essay_lengths.png
│   ├── training_curves.png
│   ├── results_bar.png
│   ├── probing_heatmap.png
│   ├── tsne_embeddings.png
│   ├── pca_traits.png
│   ├── cross_trait_transfer.png
│   └── error_examples.csv
└── src/
    ├── data.py               # Loading, validation, EDA plots, stratified splits
    ├── baselines.py          # Mean baseline and TF-IDF + Ridge
    ├── models.py             # BiLSTMRegressor architecture
    ├── train.py              # BiLSTM training loop (multi-seed)
    ├── evaluate.py           # MCRMSE scoring and results persistence
    ├── transformer_models.py # DistilBERT frozen / full fine-tune / LoRA variants
    ├── probe.py              # Layer-wise probing study (pretrained vs fine-tuned)
    ├── represent.py          # t-SNE, PCA, cross-trait transfer analysis
    ├── plots.py              # Shared plotting utilities
    └── analyse.py            # Results bar chart and qualitative error analysis
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate nlp-final
```

---

## Running

```bash
python main.py
```

The script is fully automated. It will:

1. Load `data/train.csv` (auto-downloaded via the Kaggle API if absent; requires `~/.kaggle/kaggle.json`).
2. Validate the data and save EDA plots to `outputs/`.
3. Create a stratified 80/10/10 train/val/test split (no data leakage).
4. Train and evaluate all models in order of complexity:
   - Mean baseline
   - TF-IDF + Ridge regression
   - Bidirectional LSTM with learned embeddings (3 seeds)
   - DistilBERT: frozen encoder, full fine-tune, LoRA (r=4, 8, 16)
5. Run a layer-wise probing study comparing pretrained vs fine-tuned representations.
6. Produce t-SNE, PCA and cross-trait transfer plots.
7. Save all metrics to `outputs/results.csv` and all plots to `outputs/`.

All random seeds are fixed for fully deterministic execution. On CPU, transformer
training is automatically subsampled and reduced to one seed to keep runtime manageable.

---

## Results Summary

| Model | MCRMSE ↓ |
|---|---|
| Mean baseline | 0.638 |
| TF-IDF + Ridge | 0.535 |
| DistilBERT (frozen) | 0.595 |
| DistilBERT (fine-tuned) | **0.535** |
| LoRA r=4 | 0.598 |
| LoRA r=8 | 0.561 |
| LoRA r=16 | 0.601 |

Results are saved to `outputs/results.csv` after running.
