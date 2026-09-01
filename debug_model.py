import sys
import os
import torch
import librosa
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config as cfg
from src.model import CTCASR
from src.decode import greedy_decode

def debug_model():
    """Debug script – shows raw model output for a LibriSpeech file."""
    model_path = "models/ctc_best.pt"
    if not os.path.exists(model_path):
        model_path = "models/ctc_epoch_20.pt"
        if not os.path.exists(model_path):
            print("❌ No model found in models/")
            return

    # Try GPU, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📂 Loading model: {model_path} on {device}")
    model = CTCASR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.lstm.flatten_parameters()
    model.eval()

    # Load a LibriSpeech file
    audio_path = "data/train-clean-100/19/198/19-198-0000.flac"
    if not os.path.exists(audio_path):
        print(f"❌ Test file not found: {audio_path}")
        return

    print(f"🎵 Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=cfg.SAMPLE_RATE)

    # MFCC with .T fix
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=cfg.N_MFCC,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = mfcc.T  # 🔥 Transpose!
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    input_len = torch.tensor([mfcc.size(1)], dtype=torch.long).to(device)

    print(f"MFCC shape: {mfcc.shape}")

    with torch.no_grad():
        logits = model(mfcc, input_len)
        print(f"Logits shape: {logits.shape}")

        preds = torch.argmax(logits, dim=-1)[0]
        print(f"Argmax indices (first 50): {preds[:50].tolist()}")

        blank_count = (preds == cfg.BLANK_IDX).sum().item()
        total = preds.numel()
        print(f"Blank tokens: {blank_count}/{total} ({blank_count/total*100:.1f}%)")

        text = greedy_decode(logits.cpu())
        print(f"Decoded text: '{text}'")

        # Raw chars
        chars = []
        for idx in preds:
            if idx != cfg.BLANK_IDX:
                if idx < len(cfg.VOCAB):
                    chars.append(cfg.VOCAB[idx])
                else:
                    chars.append('?')
        print(f"Raw chars (no repeat removal): {''.join(chars)}")

        # Actual transcript
        txt_path = audio_path.replace(".flac", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                transcript = f.read().strip()
            print(f"Actual transcript: {transcript}")

if __name__ == "__main__":
    debug_model()