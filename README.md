# Automated Essay Scoring for English Language Learners: BiLSTM vs DistilBERT + LoRA

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%2B%20PEFT-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20Ridge-F7931E?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Feedback%20Prize%20ELL-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> **How much does a pretrained transformer really add over classical NLP when scoring student writing, and can parameter-efficient fine-tuning close the gap?**
>
> This project predicts **six analytic writing scores** (cohesion, syntax, vocabulary, phraseology, grammar, conventions) for argumentative essays written by 8th–12th grade English Language Learners. It climbs a ladder of models from a mean baseline through **TF-IDF + Ridge** and a **BiLSTM** to **DistilBERT** (frozen, fully fine-tuned and **LoRA** at ranks 4, 8 and 16), then looks inside the transformer with layer-wise probing and representation analysis.

**TL;DR:** fully fine-tuned DistilBERT reaches the best MCRMSE (**0.535**), but a simple **TF-IDF + Ridge** baseline matches it exactly. LoRA r=8 gets within 0.026 of full fine-tuning while training only a small fraction of the weights.

Built for the Deep Learning for Natural Language Processing module (ELEC0141) at UCL.

---

## Results

MCRMSE on the held-out test split (lower is better):

| Model | MCRMSE ↓ |
|---|:---:|
| Mean baseline | 0.638 |
| TF-IDF + Ridge | **0.535** |
| DistilBERT (frozen encoder) | 0.595 |
| DistilBERT (full fine-tune) | **0.535** |
| DistilBERT + LoRA r=4 | 0.598 |
| DistilBERT + LoRA r=8 | 0.561 |
| DistilBERT + LoRA r=16 | 0.601 |

Running `main.py` writes the full results, including per-trait RMSE and standard deviation across seeds, to `outputs/results.csv`.

## Key findings

1. **Classical NLP is a seriously strong baseline.** TF-IDF bigrams with Ridge regression tie fully fine-tuned DistilBERT at 0.535 MCRMSE, at a tiny fraction of the compute.
2. **A frozen transformer isn't enough.** With the encoder frozen, DistilBERT (0.595) trails TF-IDF + Ridge. The pretrained features only pay off once the encoder adapts to the task.
3. **LoRA rank isn't "more is better".** r=8 (0.561) clearly beats both r=4 (0.598) and r=16 (0.601), so rank is a hyperparameter to tune rather than maximise.
4. **Parameter-efficient fine-tuning gets close.** LoRA r=8 lands within 0.026 MCRMSE of full fine-tuning while only training low-rank adapters on the attention query and value projections.
5. **Every model beats the mean baseline** (0.638), confirming they learn real signal from the essay text rather than the score distribution.

---

## Approach

### Data
- **Corpus:** [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) (ELLIPSE corpus, Kaggle 2022)
- **Targets:** six traits, each scored from 1.0 to 5.0
- **Validation:** schema and score-range checks; rows with missing text or scores are dropped
- **Split:** stratified 80 / 10 / 10 train / validation / test, stratified on the binned mean score so every split has a representative distribution
- **Metric:** **MCRMSE**, the mean of the per-trait root mean squared errors (the official competition metric)

### Model ladder

| Model | Details |
|---|---|
| **Mean baseline** | Predicts the training-set mean for every trait |
| **TF-IDF + Ridge** | Unigrams + bigrams, 20k features, sublinear TF, multi-output Ridge (α = 1.0) |
| **BiLSTM** | Learned 64-d word embeddings (10k vocab), single-layer bidirectional LSTM (96 units per direction), masked mean pooling, linear head; Adam + MSE; trained over 3 seeds |
| **DistilBERT · frozen** | `distilbert-base-uncased` with the encoder frozen; only the regression head is trained |
| **DistilBERT · full fine-tune** | All encoder weights updated end to end (AdamW, lr 5e-5) |
| **DistilBERT · LoRA** | Hugging Face PEFT adapters on `q_lin` and `v_lin`, ranks 4 / 8 / 16 with α = 2r and dropout 0.1 |

All neural models use attention-masked mean pooling, a regression head initialised to the training-set mean, predictions clipped to [1, 5], and the checkpoint with the lowest validation loss.

### Interpretability studies
- **Layer-wise probing:** a Ridge probe is fitted on every DistilBERT layer for every trait, comparing pretrained and fine-tuned encoders to see where trait information lives and how fine-tuning moves it.
- **Representation analysis:** t-SNE of fine-tuned essay embeddings coloured by score, PCA of the six-trait label space, and a cross-trait transfer matrix showing how well a probe trained on one trait predicts the others.
- **Error analysis:** the essays with the highest prediction error are saved with excerpts for qualitative review.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/rav567/essay-scoring-bilstm-vs-distilbert-lora.git
cd essay-scoring-bilstm-vs-distilbert-lora
conda env create -f environment.yml
conda activate nlp-final
```

Prefer pip? Use Python 3.11 and run `pip install -r requirements.txt` instead.

### 2. Get the data

The dataset downloads automatically through the Kaggle API on the first run. You need to:

1. Join the [competition](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) on Kaggle and accept its rules.
2. Create an API token under **Kaggle → Settings → API → Create New Token**, and save `kaggle.json` to `~/.kaggle/`. Alternatively, set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

Or download `train.csv` manually from the competition page and place it at `data/train.csv`.

### 3. Run

```bash
python main.py
```

One command runs the whole pipeline: data validation and EDA plots, the stratified split, every model, the probing and representation studies, and the final results table and charts.

> **GPU recommended.** On a CPU the pipeline automatically shrinks itself so it finishes in reasonable time: fewer epochs, one seed for the transformers, and a subsampled training set. CPU results will therefore differ from the table above.

## Outputs

Everything is written to `outputs/`:

| File | What it shows |
|---|---|
| `results.csv` | MCRMSE (mean ± SD over seeds) and per-trait RMSE for every model |
| `results_bar.png` | Model comparison bar chart |
| `training_curves.png` | Training and validation loss for the neural models |
| `score_distributions.png` · `trait_correlation.png` · `essay_lengths.png` | Exploratory data analysis |
| `probing_heatmap.png` | Layer-wise probe correlations, pretrained vs fine-tuned |
| `tsne_embeddings.png` · `pca_traits.png` · `cross_trait_transfer.png` | Representation analysis |
| `error_examples.csv` | Highest-error test essays for qualitative review |

## Project structure

```
├── main.py                   # Single entry point: runs the full pipeline
├── environment.yml           # Conda environment (pinned versions)
├── requirements.txt          # Pip dependencies
├── .pylintrc                 # Lint configuration
└── src/
    ├── data.py               # Kaggle download, validation, EDA plots, stratified split
    ├── baselines.py          # Mean baseline and TF-IDF + Ridge
    ├── models.py             # BiLSTMRegressor architecture
    ├── train.py              # BiLSTM vocabulary, encoding and multi-seed training
    ├── transformer_models.py # DistilBERT frozen / full fine-tune / LoRA variants
    ├── probe.py              # Layer-wise probing study
    ├── represent.py          # t-SNE, PCA and cross-trait transfer analysis
    ├── evaluate.py           # Per-trait RMSE, MCRMSE and results CSV
    ├── analyse.py            # Results bar chart and error analysis
    └── plots.py              # Shared plotting utilities
```

`data/` and `outputs/` are created at run time and are git-ignored.

## Reproducibility

Seeds are fixed for Python, NumPy and PyTorch (`seed = 42`; the BiLSTM and transformers run over seeds 42, 1 and 2), and cuDNN runs in deterministic mode. Dependency versions are pinned in `environment.yml`.

## Tech stack

Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

## Author

**Ravpreet Gill** · [@rav567](https://github.com/rav567)

## Acknowledgements

Data from the [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) competition, hosted by Vanderbilt University and The Learning Agency Lab, using the ELLIPSE corpus.
