"""Layer probing study for pretrained vs fine-tuned DistilBERT.

Extracts mean-pooled hidden states from every transformer layer and fits a
linear Ridge probe per layer per trait. Pearson correlation between probe
predictions and ground-truth scores is reported as a heatmap, showing which
layers encode linguistically relevant information before and after fine-tuning.
"""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

from src.data import TARGETS
from src.transformer_models import MODEL_NAME, _encode, train_for_probing
from src.plots import plot_probing_heatmap


def _layer_embeddings(encoder, ids, mask, device, batch_size=32):
    """Extract mean-pooled embeddings from every hidden layer of the encoder.

    Returns:
        numpy array of shape (num_layers, n, hidden_dim).
    """
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            b_ids = ids[i:i + batch_size].to(device)
            b_mask = mask[i:i + batch_size].to(device)
            out = encoder(input_ids=b_ids, attention_mask=b_mask, output_hidden_states=True)
            weight = b_mask.unsqueeze(-1).float()
            pooled = [
                (h * weight).sum(1) / weight.sum(1).clamp(min=1.0)
                for h in out.hidden_states
            ]
            chunks.append(torch.stack(pooled, dim=0).cpu())
    return torch.cat(chunks, dim=1).numpy()


def _probe(train_emb, test_emb, y_train, y_test):
    """Fit a Ridge probe per layer per trait and return Pearson correlations.

    Returns:
        numpy array of shape (num_layers, 6) with Pearson r values in [0, 1].
    """
    scores = np.zeros((train_emb.shape[0], len(TARGETS)))
    for layer in range(train_emb.shape[0]):
        for trait in range(len(TARGETS)):
            ridge = Ridge(alpha=1.0)
            ridge.fit(train_emb[layer], y_train[:, trait])
            pred = ridge.predict(test_emb[layer])
            corr = np.corrcoef(pred, y_test[:, trait])[0, 1]
            scores[layer, trait] = 0.0 if np.isnan(corr) else corr
    return scores


def run_probing(train_df, val_df, test_df, output_dir, seed=42, max_len=256):
    """Run the full probing study and save a heatmap to output_dir.

    Compares layer-wise probe correlations for a pretrained DistilBERT encoder
    against one fine-tuned on the ELL task, revealing how fine-tuning reshapes
    internal representations.

    On CPU, training and test sets are subsampled to keep runtime manageable.

    Returns:
        Dict mapping encoder name to a numpy array of shape (num_layers, 6).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        train_df = train_df.sample(n=min(400, len(train_df)), random_state=seed)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, train_mask = _encode(tokenizer, train_df["full_text"], max_len)
    test_ids, test_mask = _encode(tokenizer, test_df["full_text"], max_len)
    y_train = train_df[TARGETS].values
    y_test = test_df[TARGETS].values

    pretrained = AutoModel.from_pretrained(MODEL_NAME).to(device)
    finetuned = train_for_probing(train_df, val_df, seed=seed).encoder.to(device)

    results = {}
    for name, encoder in [("pretrained", pretrained), ("fine-tuned", finetuned)]:
        train_emb = _layer_embeddings(encoder, train_ids, train_mask, device)
        test_emb = _layer_embeddings(encoder, test_ids, test_mask, device)
        results[name] = _probe(train_emb, test_emb, y_train, y_test)

    plot_probing_heatmap(results, output_dir, list(TARGETS))
    return results