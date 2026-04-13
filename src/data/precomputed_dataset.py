#!/usr/bin/env python3
"""
Dataset that loads precomputed features from disk instead of raw waveforms.

Replaces the frontend forward pass with a single torch.load() per sample.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SpecAugment(nn.Module):
    """SpecAugment-style masking for precomputed spectral features.

    Applied during training only. Zeros out random time and frequency bands
    to prevent the model from memorizing domain-specific spectral patterns.
    """

    def __init__(self, freq_masks: int = 2, freq_width: int = 4,
                 time_masks: int = 2, time_width: int = 20):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, freq, time)"""
        if not self.training:
            return x
        x = x.clone()
        _, _, F, T = x.shape
        for _ in range(self.freq_masks):
            f = torch.randint(0, max(F - self.freq_width, 1), (1,)).item()
            x[:, :, f:f + self.freq_width, :] = 0
        for _ in range(self.time_masks):
            t = torch.randint(0, max(T - self.time_width, 1), (1,)).item()
            x[:, :, :, t:t + self.time_width] = 0
        return x


class PrecomputedDataset(Dataset):
    """
    Loads precomputed features (.pt files) from disk.

    Expected layout:
        root/
            mel/ or wavelet/   (feature .pt files: 000000.pt, 000001.pt, ...)
            labels.pt          (LongTensor of labels)
    """

    def __init__(self, root: str, frontend: str):
        self.feature_dir = Path(root) / frontend
        self.labels = torch.load(Path(root) / "labels.pt", weights_only=True)

        # Count files
        self.n_samples = len(self.labels)

        # Verify first file exists
        first = self.feature_dir / "000000.pt"
        if not first.exists():
            raise FileNotFoundError(f"No precomputed features at {self.feature_dir}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        features = torch.load(
            self.feature_dir / f"{idx:06d}.pt", weights_only=True
        )
        label = int(self.labels[idx])
        meta = {"idx": idx, "label": label}
        return features, label, meta


def precomputed_collate_fn(batch):
    """Collate for precomputed features."""
    features = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metadata = [b[2] for b in batch]
    return features, labels, metadata


def get_precomputed_dataloader(
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
        collate_fn=precomputed_collate_fn,
        pin_memory=True,
    )
