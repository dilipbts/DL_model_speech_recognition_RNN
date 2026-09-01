import torch
from src.config import Config as cfg
import jiwer

def decode_indices_to_text(indices):
    """Convert token indices to string, removing blanks and repeats."""
    chars = []
    prev = cfg.BLANK_IDX
    for idx in indices:
        if idx != prev and idx != cfg.BLANK_IDX:
            chars.append(cfg.VOCAB[idx])
        prev = idx
    return ''.join(chars)

def calculate_wer(pred_text, target_text):
    return jiwer.wer(target_text, pred_text)