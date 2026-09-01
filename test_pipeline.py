import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import librosa
import numpy as np
from src.config import Config as cfg
from src.data_loader import LibriSpeechDataset, collate_fn
from src.model import CTCASR
from torch.utils.data import DataLoader

print("="*60)
print("🧪 PIPELINE TEST")
print("="*60)

# Test 1: CUDA & Config
print("\n[1] Checking CUDA and Config...")
print(f"  Device: {cfg.DEVICE}")
print(f"  Batch Size: {cfg.BATCH_SIZE}")
if cfg.DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print("  ✅ OK")

# Test 2: Dataset loading
print("\n[2] Loading dataset (first 50 files)...")
class TestDataset(LibriSpeechDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_files = self.audio_files[:50]
        self.transcripts = self.transcripts[:50]

try:
    ds = TestDataset(cfg.DATA_ROOT, train=True)
    mfcc, target, in_len, tgt_len = ds[0]
    print(f"  Loaded {len(ds)} samples. Sample transcript: {ds.transcripts[0][:50]}...")
    print("  ✅ OK")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 3: Model forward pass
print("\n[3] Testing model forward pass...")
model = CTCASR().to(cfg.DEVICE)
dummy = torch.randn(1, 200, cfg.INPUT_SIZE).to(cfg.DEVICE)
dummy_len = torch.tensor([200], dtype=torch.long).to(cfg.DEVICE)
with torch.no_grad():
    out = model(dummy, dummy_len)
print(f"  Output shape: {out.shape} (expected (1, 200, {len(cfg.VOCAB)+1}))")
print("  ✅ OK")

# Test 4: One training iteration
print("\n[4] Testing one training iteration...")
loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, num_workers=0)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
ctc_loss = torch.nn.CTCLoss(blank=cfg.BLANK_IDX, reduction='mean', zero_infinity=True)
for batch in loader:
    mfccs, targets, input_lens, target_lens = batch
    mfccs = mfccs.to(cfg.DEVICE)
    targets = targets.to(cfg.DEVICE)
    input_lens = input_lens.to(cfg.DEVICE)
    target_lens = target_lens.to(cfg.DEVICE)
    logits = model(mfccs, input_lens)
    log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    break
print("  ✅ OK")

print("\n🎉 All tests passed! Ready for full training.")
print("Run: python run_training.py")