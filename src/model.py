import torch
import torch.nn as nn
from src.config import Config as cfg

class CTCASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = len(cfg.VOCAB)
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_SIZE,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.HIDDEN_SIZE * 2, self.vocab_size + 1)  # +1 for blank

    def forward(self, x, input_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.fc(output)
        return logits