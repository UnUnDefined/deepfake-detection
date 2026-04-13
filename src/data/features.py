#!/usr/bin/env python3
"""
Feature extraction for the 2x2 factorial design.

Two front-ends:
  1. Mel spectrogram (standard baseline)
  2. Wavelet scattering transform (experimental condition)

Both take raw waveforms (batch, T) and return feature tensors.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np


class MelSpectrogramFrontEnd(nn.Module):
    """
    Standard mel spectrogram extraction.
    
    Input:  (batch, T)  raw waveform at 16kHz
    Output: (batch, 1, n_mels, time_frames)  log-mel spectrogram
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.log_offset = 1e-6
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, T)
        mel = self.mel_transform(waveform)          # (batch, n_mels, time)
        log_mel = torch.log(mel + self.log_offset)
        return log_mel.unsqueeze(1)                  # (batch, 1, n_mels, time)
    
    @property
    def output_channels(self):
        return 1


class WaveletScatteringFrontEnd(nn.Module):
    """
    Wavelet scattering transform via Kymatio.
    
    The scattering transform is a cascade of wavelet convolutions
    and modulus nonlinearities. It produces a multi-resolution,
    locally translation-invariant representation.
    
    J controls the max scale (2^J samples), Q controls frequency resolution.
    
    Input:  (batch, T)  raw waveform
    Output: (batch, 1, n_coeffs, time_frames)  scattering coefficients
    
    NOTE: Kymatio import is deferred so the rest of the codebase
    doesn't hard-depend on it during development.
    """
    
    def __init__(
        self,
        shape: int = 64000,    # input length in samples
        J: int = 8,            # scattering depth (max scale = 2^J)
        Q: int = 12,           # filters per octave
    ):
        super().__init__()
        self.shape = shape
        self.J = J
        self.Q = Q
        self._scattering = None  # lazy init
    
    def _init_scattering(self, device):
        from kymatio.torch import Scattering1D
        self._scattering = Scattering1D(
            J=self.J,
            shape=(self.shape,),
            Q=self.Q,
        ).to(device)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, T)
        if self._scattering is None:
            self._init_scattering(waveform.device)
        
        # Kymatio expects (batch, T)
        Sx = self._scattering(waveform)  # (batch, n_coeffs, time_frames)
        
        # Log-compress for dynamic range
        Sx = torch.log1p(Sx)
        
        return Sx.unsqueeze(1)  # (batch, 1, n_coeffs, time_frames)
    
    @property
    def output_channels(self):
        return 1


class WaveletScatteringV2FrontEnd(nn.Module):
    """
    Improved wavelet scattering frontend.

    Changes from v1:
      1. average=False — preserves temporal dynamics per channel (the whole point)
      2. order=1 only — skip order-2 cross-frequency paths (noisy, hard to learn)
      3. Q=8 — standard for speech, less redundant than Q=12
      4. log compression (not log1p) with offset — much better dynamic range
      5. Per-channel normalization — each scattering path normalized independently
         so order-0 (loud) doesn't dominate order-1 (quiet but informative)

    Input:  (batch, T) raw waveform
    Output: (batch, 1, n_coeffs, time_frames) normalized scattering coefficients
    """

    def __init__(
        self,
        shape: int = 64000,
        J: int = 8,
        Q: int = 8,
    ):
        super().__init__()
        self.shape = shape
        self.J = J
        self.Q = Q
        self._scattering = None
        # Running stats for per-channel normalization (computed during precompute)
        self.register_buffer("channel_mean", None)
        self.register_buffer("channel_std", None)
        self._stats_accumulator = {"sum": None, "sum_sq": None, "count": 0}

    def _init_scattering(self, device):
        from kymatio.torch import Scattering1D
        self._scattering = Scattering1D(
            J=self.J,
            shape=(self.shape,),
            Q=self.Q,
            max_order=1,        # order-1 only — skip noisy cross-frequency paths
            T=0,                # no temporal averaging (preserves dynamics)
            out_type="list",    # required when T=0
        ).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._scattering is None:
            self._init_scattering(waveform.device)

        # out_type='list' returns list of dicts with 'coef' and 'j' keys
        Sx_list = self._scattering(waveform)

        # Stack all coefficient tensors and align time dimension
        # Each coef has shape (batch, time_i) where time_i varies by scale
        # Interpolate all to the same time resolution (max time)
        coefs = [entry["coef"] for entry in Sx_list]
        max_time = max(c.shape[-1] for c in coefs)
        aligned = []
        for c in coefs:
            if c.shape[-1] < max_time:
                c = torch.nn.functional.interpolate(
                    c.unsqueeze(1), size=max_time, mode="linear", align_corners=False
                ).squeeze(1)
            aligned.append(c)
        Sx = torch.stack(aligned, dim=1)  # (batch, n_coeffs, time)

        # Log compression — abs ensures positive before log (scattering can have
        # small negative values from interpolation artifacts)
        Sx = torch.log(torch.abs(Sx) + 1e-6)

        # Temporal pooling: 64000 frames → ~250 (matches mel's time resolution)
        # kernel=256 with stride=256 gives 64000/256 = 250 frames
        if Sx.shape[-1] > 1000:
            pool_size = Sx.shape[-1] // 250
            Sx = torch.nn.functional.avg_pool1d(Sx, kernel_size=pool_size, stride=pool_size)

        # Per-channel normalization if stats are available
        if self.channel_mean is not None:
            mean = self.channel_mean.unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
            std = self.channel_std.unsqueeze(0).unsqueeze(-1)
            Sx = (Sx - mean) / (std + 1e-8)

        return Sx.unsqueeze(1)  # (batch, 1, n_coeffs, time_frames)

    def update_stats(self, Sx_batch: torch.Tensor):
        """Accumulate per-channel stats from a batch (call during precompute)."""
        # Sx_batch: (batch, n_coeffs, time) — already log-compressed
        batch_mean = Sx_batch.mean(dim=(0, 2))  # (n_coeffs,)
        batch_sq = (Sx_batch ** 2).mean(dim=(0, 2))
        n = Sx_batch.shape[0] * Sx_batch.shape[2]

        if self._stats_accumulator["sum"] is None:
            self._stats_accumulator["sum"] = batch_mean * n
            self._stats_accumulator["sum_sq"] = batch_sq * n
        else:
            self._stats_accumulator["sum"] += batch_mean * n
            self._stats_accumulator["sum_sq"] += batch_sq * n
        self._stats_accumulator["count"] += n

    def finalize_stats(self):
        """Compute final mean/std from accumulated stats."""
        count = self._stats_accumulator["count"]
        if count == 0:
            return
        mean = self._stats_accumulator["sum"] / count
        var = self._stats_accumulator["sum_sq"] / count - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-8))
        self.channel_mean = mean
        self.channel_std = std

    @property
    def output_channels(self):
        return 1


class WaveletScatteringV3FrontEnd(nn.Module):
    """
    Wavelet scattering v3: WST-X-inspired configuration for deepfake detection.

    Key changes from v2 (based on WST-X series, arXiv 2602.02980):
      1. Smaller J (5 vs 8) — preserves transient synthesis artifacts that large
         averaging windows destroy. WST-X showed J=2 optimal but they combine
         with SSL features; standalone needs J=4-5 for sufficient frequency range.
      2. Higher Q (10 vs 8) — finer frequency resolution captures subtle spectral
         artifacts from neural vocoders.
      3. Second order (max_order=2 vs 1) — order-2 coefficients capture amplitude
         modulation patterns (how energy at one frequency modulates another),
         which are key synthesis artifacts that order-1 misses.

    Output: ~100 coefficients covering 250-8000 Hz with modulation dynamics.
    Speed: ~13 samples/sec (vs 19 for v2) — acceptable since we precompute.

    Input:  (batch, T) raw waveform
    Output: (batch, 1, n_coeffs, time_frames) normalized scattering coefficients
    """

    def __init__(
        self,
        shape: int = 64000,
        J: int = 5,
        Q: int = 10,
        max_order: int = 2,
    ):
        super().__init__()
        self.shape = shape
        self.J = J
        self.Q = Q
        self.max_order = max_order
        self._scattering = None
        self.register_buffer("channel_mean", None)
        self.register_buffer("channel_std", None)

    def _init_scattering(self, device):
        from kymatio.torch import Scattering1D
        self._scattering = Scattering1D(
            J=self.J,
            shape=(self.shape,),
            Q=self.Q,
            max_order=self.max_order,
            T=0,
            out_type="list",
        ).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._scattering is None:
            self._init_scattering(waveform.device)

        Sx_list = self._scattering(waveform)

        coefs = [entry["coef"] for entry in Sx_list]
        max_time = max(c.shape[-1] for c in coefs)
        aligned = []
        for c in coefs:
            if c.shape[-1] < max_time:
                c = torch.nn.functional.interpolate(
                    c.unsqueeze(1), size=max_time, mode="linear", align_corners=False
                ).squeeze(1)
            aligned.append(c)
        Sx = torch.stack(aligned, dim=1)  # (batch, n_coeffs, time)

        # Log compression
        Sx = torch.log(torch.abs(Sx) + 1e-6)

        # Temporal pooling to ~250 frames (match mel resolution)
        if Sx.shape[-1] > 1000:
            pool_size = Sx.shape[-1] // 250
            Sx = torch.nn.functional.avg_pool1d(Sx, kernel_size=pool_size, stride=pool_size)

        # Per-channel normalization
        if self.channel_mean is not None:
            mean = self.channel_mean.unsqueeze(0).unsqueeze(-1)
            std = self.channel_std.unsqueeze(0).unsqueeze(-1)
            Sx = (Sx - mean) / (std + 1e-8)

        return Sx.unsqueeze(1)  # (batch, 1, n_coeffs, time_frames)

    @property
    def output_channels(self):
        return 1


def get_frontend(name: str, config: dict) -> nn.Module:
    """Factory function for feature front-ends."""
    if name == "mel":
        return MelSpectrogramFrontEnd(**config.get("mel", {}))
    elif name == "wavelet":
        return WaveletScatteringFrontEnd(**config.get("wavelet", {}))
    elif name == "wavelet_v2":
        return WaveletScatteringV2FrontEnd(**config.get("wavelet_v2", {}))
    elif name == "wavelet_v3":
        return WaveletScatteringV3FrontEnd(**config.get("wavelet_v3", {}))
    elif name == "wavelet_v3_order1":
        return WaveletScatteringV3FrontEnd(**config.get("wavelet_v3_order1", {}))
    else:
        raise ValueError(f"Unknown frontend: {name}.")
