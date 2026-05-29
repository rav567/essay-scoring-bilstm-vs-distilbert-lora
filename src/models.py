"""PyTorch model definitions for the ELL essay scoring task.

Contains the BiLSTMRegressor, a bidirectional LSTM with a learned word
embedding layer and a linear regression head over six scoring traits.
"""
import torch  # noqa: F401
from torch import nn


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM regressor for multi-trait essay scoring.

    Architecture:
      1. Token embedding with dropout.
      2. Single-layer bidirectional LSTM.
      3. Masked mean-pooling over token dimension.
      4. Linear head projecting to num_targets scores.

    Args:
        vocab_size: Number of tokens in the vocabulary (including PAD/UNK).
        embed_dim: Dimensionality of token embeddings (default 64).
        hidden_dim: Hidden units per LSTM direction (default 96).
        num_targets: Number of regression outputs (default 6).
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=96, num_targets=6):
        """Initialise layers."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, num_targets)

    def forward(self, tokens):  # tokens: LongTensor (batch, seq_len)
        """Forward pass with masked mean-pooling.

        Args:
            x: LongTensor of token ids, shape (batch, seq_len).

        Returns:
            FloatTensor of predicted scores, shape (batch, num_targets).
        """
        mask = (tokens != 0).unsqueeze(-1).float()
        emb = self.dropout(self.embedding(tokens))
        out, _ = self.lstm(emb)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))
