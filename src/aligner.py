import torch
from src.config import Config as cfg

def ctc_alignment(logits):
    """Simple CTC alignment: returns list of (char, start_frame, end_frame)."""
    logits = logits.squeeze(0)  # (T, C)
    preds = torch.argmax(logits, dim=-1)  # (T,)
    T = preds.size(0)
    segments = []
    current_char = cfg.BLANK_IDX
    start = 0
    for t, p in enumerate(preds):
        if p != current_char:
            if current_char != cfg.BLANK_IDX:
                segments.append((current_char, start, t))
            current_char = p
            start = t
    if current_char != cfg.BLANK_IDX:
        segments.append((current_char, start, T))
    # Merge consecutive same chars
    merged = []
    for char, s, e in segments:
        if char == cfg.BLANK_IDX:
            continue
        if merged and merged[-1][0] == char:
            merged[-1] = (char, merged[-1][1], e)
        else:
            merged.append((char, s, e))
    return merged