import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.data import TARGETS

MODEL_NAME = "distilbert-base-uncased"


class DistilBertRegressor(nn.Module):
    def __init__(self, encoder, hidden_dim, num_targets=6):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, num_targets)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


def _build(kind, rank):
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
    enc = tokenizer(
        list(texts), truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _predict(model, ids, mask, device, batch_size=32):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            out = model(ids[i:i + batch_size].to(device), mask[i:i + batch_size].to(device))
            outs.append(out.cpu())
    return torch.cat(outs).clamp(1.0, 5.0).numpy()


def _train_single(kind, rank, data, device, seed):
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
    return preds, {"train_loss": train_losses, "val_loss": val_losses}


def train_transformers(train_df, val_df, eval_df, seeds, lora_ranks=(4, 8, 16), max_len=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        train_df = train_df.sample(n=min(400, len(train_df)), random_state=42)
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
            pred, history = _train_single(kind, rank, data, device, seed)
            preds.append(pred)
            hists.append(history)
        preds_by_model[label] = preds
        histories[label] = hists
    return eval_df[TARGETS].values, preds_by_model, histories