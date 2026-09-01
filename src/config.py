import os
import torch

class Config:
    # Paths
    DATA_ROOT = "data/train-clean-100"
    MODEL_SAVE_DIR = "models"
    OUTPUT_DIR = "outputs"

    # Audio preprocessing (16kHz, MFCC)
    SAMPLE_RATE = 16000
    N_MFCC = 40
    N_FFT = 512
    HOP_LENGTH = 160          # 10ms at 16kHz

    # Model architecture
    INPUT_SIZE = N_MFCC
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    DROPOUT = 0.3

    # Training hyperparameters (RTX 3050 8GB)
    BATCH_SIZE = 2             # Reduce to 4 if OOM
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    GRAD_CLIP = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4            # Set to 0 on Windows if you get errors

    # Decoding
    BEAM_WIDTH = 10

    # Vocabulary (lowercase letters, space, apostrophe)
    VOCAB = " abcdefghijklmnopqrstuvwxyz'"
    BLANK_IDX = len(VOCAB)     # CTC blank token
    PAD_IDX = BLANK_IDX + 1

    # Create required directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)