import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=96, num_targets=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, num_targets)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))