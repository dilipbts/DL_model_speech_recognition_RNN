import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from src.config import Config as cfg

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, split_ratio=0.9, train=True):
        self.root = root_dir
        self.audio_files, self.transcripts = self._scan_files()
        # Split into train/val
        total = len(self.audio_files)
        split_idx = int(total * split_ratio)
        if train:
            self.audio_files = self.audio_files[:split_idx]
            self.transcripts = self.transcripts[:split_idx]
        else:
            self.audio_files = self.audio_files[split_idx:]
            self.transcripts = self.transcripts[split_idx:]

        # Character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(cfg.VOCAB)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

    def _scan_files(self):
        audio_paths = []
        transcripts = []
        # Walk through speaker -> chapter
        for speaker in os.listdir(self.root):
            spk_dir = os.path.join(self.root, speaker)
            if not os.path.isdir(spk_dir):
                continue
            for chapter in os.listdir(spk_dir):
                chap_dir = os.path.join(spk_dir, chapter)
                if not os.path.isdir(chap_dir):
                    continue
                # Find the single .trans.txt file for this chapter
                trans_files = glob.glob(os.path.join(chap_dir, "*.trans.txt"))
                if not trans_files:
                    continue
                trans_file = trans_files[0]
                # Parse transcript map: {filename_base: "text"}
                trans_map = {}
                with open(trans_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            trans_map[parts[0]] = parts[1].lower()
                # Find all .flac files and match with transcripts
                flac_files = glob.glob(os.path.join(chap_dir, "*.flac"))
                for flac_path in flac_files:
                    base_name = os.path.splitext(os.path.basename(flac_path))[0]
                    if base_name in trans_map:
                        audio_paths.append(flac_path)
                        transcripts.append(trans_map[base_name])
        return audio_paths, transcripts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.audio_files[idx], sr=cfg.SAMPLE_RATE)
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=cfg.N_MFCC,
            n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
        )
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).T  # (T, features)
        # Encode transcript
        text = self.transcripts[idx]
        encoded = [self.char_to_idx.get(ch, self.char_to_idx[' ']) for ch in text]
        target = torch.tensor(encoded, dtype=torch.long)
        return mfcc, target, mfcc.size(0), len(encoded)

def collate_fn(batch):
    mfccs, targets, input_lens, target_lens = zip(*batch)
    mfccs_padded = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=cfg.PAD_IDX
    )
    return (
        mfccs_padded,
        targets_padded,
        torch.tensor(input_lens, dtype=torch.long),
        torch.tensor(target_lens, dtype=torch.long),
    )