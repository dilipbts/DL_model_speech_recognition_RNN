import sys
import os

print("=" * 60)
print("🔍 FINAL SYSTEM CHECK")
print("=" * 60)

# --- 1. Python Version ---
print("\n[1] Python Version:")
print(f"    {sys.version.split()[0]} (✓ OK)")

# --- 2. PyTorch & CUDA ---
print("\n[2] PyTorch & CUDA:")
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    print(f"    CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("    ✓ GPU is ready!")
    else:
        print("    ❌ CUDA NOT available. PyTorch installed in CPU mode.")
        print("    → Run: pip uninstall torch torchaudio -y")
        print("    → Then: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
except ImportError:
    print("    ❌ PyTorch not installed.")
    sys.exit(1)

# --- 3. Required Packages ---
print("\n[3] Required Packages:")
packages = {
    "librosa": "audio processing",
    "numpy": "numerical",
    "whisper": "ASR (demo)",
    "jiwer": "WER evaluation",
    "tqdm": "progress bars",
    "scipy": "scientific",
}
missing = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f"    ✓ {pkg} ({desc})")
    except ImportError:
        print(f"    ❌ {pkg} ({desc}) - MISSING")
        missing.append(pkg)

if missing:
    print(f"\n    → Install missing: pip install {' '.join(missing)}")
    sys.exit(1)

# --- 4. Dataset Path & Loading ---
print("\n[4] Dataset Check:")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from src.config import Config as cfg
    from src.data_loader import LibriSpeechDataset

    if not os.path.exists(cfg.DATA_ROOT):
        print(f"    ❌ Dataset not found at: {cfg.DATA_ROOT}")
        print("    → Place your 'train-clean-100' folder inside 'data/'")
        sys.exit(1)
    else:
        # Try loading 5 samples
        ds = LibriSpeechDataset(cfg.DATA_ROOT, train=True)
        total = len(ds)
        print(f"    ✓ Dataset found at: {cfg.DATA_ROOT}")
        print(f"    ✓ Total samples: {total}")

        if total == 0:
            print("    ❌ No audio files found. Check folder structure.")
            sys.exit(1)

        # Test load first sample
        mfcc, target, in_len, tgt_len = ds[0]
        print(f"    ✓ Sample 0 - MFCC shape: {mfcc.shape}, Target length: {len(target)}")
        print(f"    ✓ Transcript preview: {ds.transcripts[0][:60]}...")
except Exception as e:
    print(f"    ❌ Error: {e}")
    print("    → Check if 'src/' folder exists and dataset is correct.")
    sys.exit(1)

# --- 5. Model Forward Pass ---
print("\n[5] Model Forward Pass:")
try:
    from src.model import CTCASR
    model = CTCASR().to(cfg.DEVICE)
    dummy = torch.randn(1, 100, cfg.INPUT_SIZE).to(cfg.DEVICE)
    dummy_len = torch.tensor([100], dtype=torch.long).to(cfg.DEVICE)
    with torch.no_grad():
        out = model(dummy, dummy_len)
    print(f"    ✓ Model loaded. Output shape: {out.shape}")
    print(f"    ✓ Device: {cfg.DEVICE}")
except Exception as e:
    print(f"    ❌ Model error: {e}")
    sys.exit(1)

# --- 6. Batch Size Recommendation ---
print("\n[6] Hardware Recommendation:")
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if gpu_mem_gb < 5:
    print(f"    ⚠️  Your GPU has {gpu_mem_gb:.1f} GB VRAM (4GB).")
    print("    → Set BATCH_SIZE = 2 in src/config.py")
    print("    → If OOM, use BATCH_SIZE = 1")
    if cfg.BATCH_SIZE > 2:
        print(f"    ⚠️  Current BATCH_SIZE = {cfg.BATCH_SIZE}. Change it to 2 NOW!")
    else:
        print(f"    ✓ Current BATCH_SIZE = {cfg.BATCH_SIZE} (good for 4GB)")
else:
    print(f"    ✓ GPU has {gpu_mem_gb:.1f} GB VRAM. BATCH_SIZE = {cfg.BATCH_SIZE} is fine.")

# --- 7. Check for script files ---
print("\n[7] Required Scripts:")
scripts = ["run_training.py", "video_to_subtitle.py", "test_pipeline.py"]
for scr in scripts:
    if os.path.exists(scr):
        print(f"    ✓ {scr}")
    else:
        print(f"    ⚠️  {scr} not found (optional)")

print("\n" + "=" * 60)
print("🎉 ALL CHECKS PASSED! YOUR SYSTEM IS READY.")
print("=" * 60)
print("\n🚀 To start training, run:")
print("   python run_training.py")
print("\n⏱️  Estimated time: ~15-18 hours for 20 epochs.")