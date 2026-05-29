"""DistilBERT-based regressor variants for the ELL essay scoring task.

Implements three fine-tuning strategies for DistilBERT-base-uncased:
  - frozen: encoder weights fixed, only the regression head is trained.
  - full: all encoder parameters fine-tuned end-to-end.
  - lora: parameter-efficient fine-tuning via LoRA adapters (ranks 4, 8, 16).

"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.data import TARGETS

MODEL_NAME = "distilbert-base-uncased"


class DistilBertRegressor(nn.Module):
    """DistilBERT encoder with an attention-masked mean-pool regression head.
    """

    def __init__(self, encoder, hidden_dim, num_targets=6):
        """Initialise encoder and regression head."""
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, num_targets)

    def forward(self, input_ids, attention_mask):
        """Forward pass with attention-masked mean pooling.

        Returns:
            FloatTensor of predicted scores, shape (batch, num_targets).
        """
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


def _build(kind, rank):
    """Construct a DistilBertRegressor with the requested fine-tuning strategy.

    Returns:
        DistilBertRegressor ready for training.
    """
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    dim = encoder.config.dim
    if kind == "frozen":
        for param in encoder.parameters():
            param.requires_grad = False
    elif kind == "lora":
        config = LoraConfig(
            r=rank, lora_alpha=rank * 2, target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1, bias="none",
        )
        encoder = get_peft_model(encoder, config)
    return DistilBertRegressor(encoder, dim)


def _encode(tokenizer, texts, max_len):
    """Tokenise a list of texts and return padded id/mask tensors.

    Returns:
        Tuple of (input_ids, attention_mask) as LongTensors of shape (n, max_len).
    """
    enc = tokenizer(
        list(texts), truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _predict(model, ids, mask, device, batch_size=32):
    """Run batched inference and return predictions as a numpy array.

    Returns:
        numpy array of shape (n, 6) with scores clipped to [1.0, 5.0].
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            out = model(ids[i:i + batch_size].to(device), mask[i:i + batch_size].to(device))
            outs.append(out.cpu())
    return torch.cat(outs).clamp(1.0, 5.0).numpy()


def _train_single(kind, rank, data, device, seed):
    """Train one DistilBERT variant for a single seed and return the model.

    Returns:
        Tuple of (model, predictions, history) where predictions is a numpy
        array of shape (n_eval, 6) and history has train_loss/val_loss lists.
    """
    torch.manual_seed(seed)
    train_ids, train_mask, y_train, val_ids, val_mask, y_val, eval_ids, eval_mask = data
    epochs = 4 if torch.cuda.is_available() else 1
    lr = {"frozen": 1e-3, "full": 5e-5, "lora": 5e-4}[kind]

    model = _build(kind, rank).to(device)
    with torch.no_grad():
        model.head.bias.copy_(y_train.mean(dim=0).to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(train_ids, train_mask, y_train), batch_size=16, shuffle=True
    )

    train_losses, val_losses = [], []
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for ids, mask, target in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(ids.to(device), mask.to(device)), target.to(device))
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))
        val_pred = _predict(model, val_ids, val_mask, device)
        val_losses.append(float(((val_pred - y_val.numpy()) ** 2).mean()))
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    preds = _predict(model, eval_ids, eval_mask, device)
    return model, preds, {"train_loss": train_losses, "val_loss": val_losses}


def train_transformers(train_df, val_df, eval_df, seeds, lora_ranks=(4, 8, 16), max_len=256):
    """Train all DistilBERT variants (frozen, full, LoRA) over multiple seeds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        train_df = train_df.sample(n=min(50, len(train_df)), random_state=42)
        seeds = seeds[:1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, train_mask = _encode(tokenizer, train_df["full_text"], max_len)
    val_ids, val_mask = _encode(tokenizer, val_df["full_text"], max_len)
    eval_ids, eval_mask = _encode(tokenizer, eval_df["full_text"], max_len)
    data = (
        train_ids, train_mask, torch.tensor(train_df[TARGETS].values, dtype=torch.float32),
        val_ids, val_mask, torch.tensor(val_df[TARGETS].values, dtype=torch.float32),
        eval_ids, eval_mask,
    )

    variants = [("frozen", "frozen", None), ("finetuned", "full", None)]
    variants += [(f"lora_r{r}", "lora", r) for r in lora_ranks]

    preds_by_model, histories = {}, {}
    for label, kind, rank in variants:
        preds, hists = [], []
        for seed in seeds:
            _, pred, history = _train_single(kind, rank, data, device, seed)
            preds.append(pred)
            hists.append(history)
        preds_by_model[label] = preds
        histories[label] = hists
    return eval_df[TARGETS].values, preds_by_model, histories


def train_for_probing(train_df, val_df, seed=42, max_len=256):
    """Train a fully fine-tuned DistilBERT for use in the probing study.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, train_mask = _encode(tokenizer, train_df["full_text"], max_len)
    val_ids, val_mask = _encode(tokenizer, val_df["full_text"], max_len)
    data = (
        train_ids, train_mask, torch.tensor(train_df[TARGETS].values, dtype=torch.float32),
        val_ids, val_mask, torch.tensor(val_df[TARGETS].values, dtype=torch.float32),
        val_ids, val_mask,
    )
    model, _, _ = _train_single("full", None, data, device, seed)
    return model
