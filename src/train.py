from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import TARGETS
from src.models import BiLSTMRegressor

PAD, UNK = 0, 1


def build_vocab(texts, max_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, _ in counter.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab


def encode(texts, vocab, max_len):
    encoded = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for i, text in enumerate(texts):
        for j, word in enumerate(text.lower().split()[:max_len]):
            encoded[i, j] = vocab.get(word, UNK)
    return torch.tensor(encoded)


def _forward(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def _train_single(tensors, vocab_size, seed):
    x_train, y_train, x_val, y_val, x_eval = tensors
    torch.manual_seed(seed)
    epochs = 20 if torch.cuda.is_available() else 12

    model = BiLSTMRegressor(vocab_size)
    with torch.no_grad():
        model.head.bias.copy_(y_train.mean(dim=0))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    train_losses, val_losses = [], []
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(loss_fn(_forward(model, x_val), y_val).item())
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    history = {"train_loss": train_losses, "val_loss": val_losses}
    return _forward(model, x_eval).clamp(1.0, 5.0).numpy(), history


def train_bilstm(train_df, val_df, eval_df, seeds, max_len=256):
    vocab = build_vocab(train_df["full_text"])
    tensors = (
        encode(train_df["full_text"].tolist(), vocab, max_len),
        torch.tensor(train_df[TARGETS].values, dtype=torch.float32),
        encode(val_df["full_text"].tolist(), vocab, max_len),
        torch.tensor(val_df[TARGETS].values, dtype=torch.float32),
        encode(eval_df["full_text"].tolist(), vocab, max_len),
    )
    preds, histories = [], []
    for seed in seeds:
        pred, history = _train_single(tensors, len(vocab), seed)
        preds.append(pred)
        histories.append(history)
    return eval_df[TARGETS].values, preds, histories