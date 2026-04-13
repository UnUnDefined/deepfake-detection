#!/usr/bin/env python3
"""
Dataset loaders for ASVspoof 2019 LA, WaveFake, and In-the-Wild.

All datasets return raw waveforms at 16kHz, truncated/padded to a fixed length.
Feature extraction (mel, wavelet) happens downstream in the feature module.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


def _load_audio(path: str) -> tuple:
    """Load audio using soundfile to avoid torchcodec/FFmpeg issues on Windows."""
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    else:
        waveform = waveform.T  # soundfile returns (T, channels) → (channels, T)
    return waveform, sr


# label convention: 0 = bonafide/real, 1 = spoof/fake


class ASVspoof2019LA(Dataset):
    """
    ASVspoof 2019 Logical Access dataset.
    
    Protocol format (space-separated):
      SPEAKER_ID AUDIO_FILE_ID - SYSTEM_ID KEY
      where KEY is 'bonafide' or 'spoof'
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",      # train, dev, eval
        max_samples: int = 64000,   # 4 sec at 16kHz
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        # Map split names to directory/protocol names
        split_map = {
            "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
            "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
            "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
        }
        
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())}")
        
        audio_dir_name, proto_name = split_map[split]
        self.audio_dir = self.root / "LA" / audio_dir_name / "flac"
        proto_file = self.root / "LA" / "ASVspoof2019_LA_cm_protocols" / proto_name
        
        if not proto_file.exists():
            raise FileNotFoundError(f"Protocol file not found: {proto_file}")
        
        # Parse protocol
        self.samples = []
        with open(proto_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                speaker_id = parts[0]
                file_id = parts[1]
                system_id = parts[3]
                key = parts[4]
                label = 0 if key == "bonafide" else 1
                self.samples.append({
                    "file_id": file_id,
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                    "label": label,
                    "label_str": key,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, dict]:
        info = self.samples[idx]
        audio_path = self.audio_dir / f"{info['file_id']}.flac"
        
        waveform, sr = _load_audio(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)  # (T,)
        
        # Pad or truncate
        waveform = self._fix_length(waveform)
        
        return waveform, info["label"], info
    
    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] >= self.max_samples:
            return waveform[:self.max_samples]
        else:
            pad = torch.zeros(self.max_samples - waveform.shape[0])
            return torch.cat([waveform, pad])


class WaveFake(Dataset):
    """
    WaveFake dataset.
    
    Directory structure:
      wavefake/
        LJSpeech-1.1_16k/   (real samples)
        ljspeech_melgan/     (fake)
        ljspeech_full_band_melgan/ (fake)
        ...
    """
    
    REAL_DIRS = {"LJSpeech-1.1_16k", "LJSpeech-1.1"}
    
    def __init__(
        self,
        root: str,
        max_samples: int = 64000,
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self.samples = []
        
        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            
            is_real = subdir.name in self.REAL_DIRS
            label = 0 if is_real else 1
            system = "real" if is_real else subdir.name
            
            for wav_file in sorted(subdir.glob("*.wav")):
                self.samples.append({
                    "path": wav_file,
                    "label": label,
                    "system": system,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, dict]:
        info = self.samples[idx]
        waveform, sr = _load_audio(info["path"])
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if waveform.shape[0] >= self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            pad = torch.zeros(self.max_samples - waveform.shape[0])
            waveform = torch.cat([waveform, pad])
        
        return waveform, info["label"], info


class InTheWild(Dataset):
    """
    In-the-Wild dataset.
    
    Expected: a directory of .wav files + meta.csv with columns:
      file, label (where label is 'bona-fide' or 'spoof')
    
    If no meta.csv, falls back to filename heuristics or
    the standard release structure.
    """
    
    def __init__(
        self,
        root: str,
        max_samples: int = 64000,
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self.samples = []
        
        # Try to find meta.csv
        meta_candidates = list(self.root.rglob("meta.csv"))
        if meta_candidates:
            meta = pd.read_csv(meta_candidates[0])
            audio_base = meta_candidates[0].parent
            
            for _, row in meta.iterrows():
                file_path = audio_base / row["file"]
                if not file_path.exists():
                    # Try looking in subdirectories
                    candidates = list(self.root.rglob(row["file"]))
                    if candidates:
                        file_path = candidates[0]
                    else:
                        continue
                
                label_str = str(row["label"]).lower().strip()
                label = 0 if "bona" in label_str or "real" in label_str else 1
                
                self.samples.append({
                    "path": file_path,
                    "label": label,
                    "label_str": label_str,
                })
        else:
            # Fallback: scan for wavs, require user to provide labels
            wavs = list(self.root.rglob("*.wav"))
            print(f"  [WARN] No meta.csv found for In-the-Wild. Found {len(wavs)} wav files.")
            print(f"         Labels cannot be determined without metadata.")
            for w in wavs:
                self.samples.append({
                    "path": w,
                    "label": -1,  # unknown
                    "label_str": "unknown",
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, dict]:
        info = self.samples[idx]
        waveform, sr = _load_audio(info["path"])
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if waveform.shape[0] >= self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            pad = torch.zeros(self.max_samples - waveform.shape[0])
            waveform = torch.cat([waveform, pad])
        
        return waveform, info["label"], info


# ── collate function ──────────────────────────────────────────────
def collate_fn(batch):
    """Custom collate that handles the metadata dict."""
    waveforms = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metadata = [b[2] for b in batch]
    return waveforms, labels, metadata


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
