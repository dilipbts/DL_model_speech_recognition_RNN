import sys
import os
import subprocess
import argparse
import torch
import librosa
import numpy as np
from datetime import timedelta
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config as cfg
from src.model import CTCASR
from src.decode import greedy_decode

# ============================================================
# 1. LOAD THE TRAINED MODEL (Auto GPU/CPU)
# ============================================================
def load_model(model_path="models/ctc_best.pt"):
    """Load the trained LSTM model – attempts GPU, falls back to CPU on error."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Available models:")
        for f in os.listdir("models"):
            if f.endswith(".pt"):
                print(f"     - {f}")
        sys.exit(1)

    # Try GPU first
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"⚡ Attempting to load model on GPU (RTX 3050)...")
    else:
        print(f"ℹ️  No GPU detected – using CPU.")

    model = CTCASR().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.lstm.flatten_parameters()  # Prevent LSTM shape errors
        model.eval()
        print(f"✅ Model loaded on {device.upper()} from: {model_path}")
        return model, device
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "shape" in str(e).lower():
            print(f"⚠️  GPU error: {e}")
            print("⏳ Falling back to CPU...")
            device = "cpu"
            model = CTCASR().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.eval()
            print(f"✅ Model loaded on CPU from: {model_path}")
            return model, device
        else:
            raise e

# ============================================================
# 2. EXTRACT AUDIO FROM VIDEO (if needed)
# ============================================================
def extract_audio(input_path, output_path="temp_audio.wav"):
    """Convert any media (video or audio) to 16kHz mono WAV."""
    if input_path.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.aac')):
        return input_path

    print(f"🎵 Extracting audio from: {input_path}")
    cmd = f"ffmpeg -i \"{input_path}\" -ar {cfg.SAMPLE_RATE} -ac 1 \"{output_path}\" -y -loglevel error"
    result = subprocess.call(cmd, shell=True)
    if result != 0:
        print("❌ FFmpeg error. Is FFmpeg installed?")
        print("   Install: winget install FFmpeg (Windows) or sudo apt install ffmpeg (Linux)")
        sys.exit(1)
    print(f"✅ Audio extracted to: {output_path}")
    return output_path

# ============================================================
# 3. TRANSCRIBE A SINGLE AUDIO CHUNK (WITH .T FIX)
# ============================================================
def transcribe_chunk(model, audio, sr, device):
    """Transcribe a single audio chunk (max ~10 seconds)."""
    if len(audio) == 0:
        return ""

    # Extract MFCCs
    try:
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=cfg.N_MFCC,
            n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
        )
    except Exception as e:
        print(f"   ⚠️ MFCC extraction failed: {e}")
        return ""

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # 🔥 CRITICAL FIX: Transpose to (time, features) – the LSTM expects this!
    mfcc = mfcc.T

    # Convert to tensor on the chosen device
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    # Skip if too short
    if mfcc.size(1) < 5:
        return ""

    input_len = torch.tensor([mfcc.size(1)], dtype=torch.long).to(device)

    # Model inference
    with torch.no_grad():
        try:
            logits = model(mfcc, input_len)
            text = greedy_decode(logits.cpu())
        except RuntimeError as e:
            # If GPU fails, fall back to CPU for this chunk
            if "cuda" in str(e).lower() and device == "cuda":
                print(f"   ⚠️ GPU error on chunk – falling back to CPU...")
                mfcc_cpu = mfcc.cpu()
                input_len_cpu = input_len.cpu()
                model_cpu = model.cpu()
                logits = model_cpu(mfcc_cpu, input_len_cpu)
                text = greedy_decode(logits.cpu())
                # Move model back to GPU for next chunks
                model.to(device)
            else:
                return ""
    return text

# ============================================================
# 4. TRANSCRIBE FULL AUDIO (with chunking)
# ============================================================
def transcribe_audio(model, audio_path, device):
    """Run the LSTM model on an audio file, handling long files by chunking."""
    print(f"📝 Transcribing: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=cfg.SAMPLE_RATE)
    total_duration = len(audio) / sr
    print(f"⏱️  Audio duration: {total_duration:.1f} seconds")

    chunk_seconds = 10
    chunk_samples = chunk_seconds * sr
    overlap_seconds = 2
    overlap_samples = overlap_seconds * sr
    hop_samples = chunk_samples - overlap_samples

    if len(audio) <= chunk_samples:
        return transcribe_chunk(model, audio, sr, device)

    print(f"🔄 Processing in {chunk_seconds}s chunks with {overlap_seconds}s overlap...")
    all_text = []
    total_chunks = 0

    for start in range(0, len(audio), hop_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        if len(chunk) < sr:
            continue

        text = transcribe_chunk(model, chunk, sr, device)
        if text:
            all_text.append(text)

        total_chunks += 1
        progress = (end / len(audio)) * 100
        print(f"   Chunk {total_chunks} – Progress: {progress:.1f}%", end="\r")

    print("\n✅ Transcription complete!")
    return ' '.join(all_text)

# ============================================================
# 5. GENERATE SRT SUBTITLES
# ============================================================
def generate_srt(text, audio_duration, output_path="outputs/lstm_subtitles.srt"):
    """Create a basic .srt file (one subtitle per sentence)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [text]

    segment_duration = audio_duration / len(sentences)

    with open(output_path, 'w', encoding='utf-8') as f:
        idx = 1
        current_time = 0
        for sent in sentences:
            start = current_time
            end = min(start + segment_duration, audio_duration)
            f.write(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{sent}\n\n")
            current_time = end
            idx += 1

    print(f"💾 Subtitles saved to: {output_path}")
    return output_path

# ============================================================
# 6. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video using the trained LSTM model."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to audio file (.wav, .flac, .mp3) or video file (.mp4, .avi, .mov)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ctc_best.pt",
        help="Path to model checkpoint (default: models/ctc_best.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/lstm_subtitles.srt",
        help="Path to save .srt file (default: outputs/lstm_subtitles.srt)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print("=" * 60)
    print("🎙️  LSTM SUBTITLE GENERATOR (TRAINED MODEL)")
    print("=" * 60)
    print(f"📁 Input: {args.file}")
    print(f"🧠 Model: {args.model}")
    print("=" * 60)

    # Step 1: Load model (auto GPU/CPU fallback)
    model, device = load_model(args.model)

    # Step 2: Extract audio if it's a video
    audio_path = extract_audio(args.file)

    # Step 3: Transcribe (handles long files automatically)
    text = transcribe_audio(model, audio_path, device)

    # Step 4: Print result
    print("\n" + "=" * 60)
    print("📝 TRANSCRIPTION RESULT:")
    print("=" * 60)
    print(text if text else "[No text recognized]")
    print("=" * 60)

    # Step 5: Generate SRT
    audio, sr = librosa.load(audio_path, sr=cfg.SAMPLE_RATE)
    duration = len(audio) / sr
    srt_path = generate_srt(text, duration, args.output)

    # Step 6: Clean up temp audio
    if audio_path == "temp_audio.wav" and os.path.exists(audio_path):
        os.remove(audio_path)

    print("\n🎯 HOW TO CHECK THE SUBTITLES:")
    print(f"   1. Open the .srt file: {srt_path}")
    print(f"   2. Load it in VLC with your video/audio.")
    print("   3. Or view the text above right here in the terminal!")

if __name__ == "__main__":
    main()