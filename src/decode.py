import torch
from src.config import Config as cfg

def greedy_decode(logits):
    """Greedy decoder for a single batch (batch_size=1)."""
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    preds = torch.argmax(logits, dim=-1)
    # collapse repeats and blanks
    decoded = []
    prev = cfg.BLANK_IDX
    for idx in preds:
        if idx != prev and idx != cfg.BLANK_IDX:
            decoded.append(idx)
        prev = idx
    return ''.join(cfg.VOCAB[i] for i in decoded)