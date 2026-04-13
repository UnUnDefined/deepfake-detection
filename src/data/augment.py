#!/usr/bin/env python3
"""
Waveform-level audio augmentation for domain-robust training.

Two augmentation modes:
  - "simple": Gaussian noise + lowpass + gain (original pipeline)
  - "kaldi":  MUSAN noise/babble/music + RIR reverb + codec compression
              (Snyder et al. 2018, the standard Kaldi x-vector recipe)

Applied to raw audio BEFORE feature extraction (mel/wavelet) to simulate
the channel/environment variability between ASVspoof (clean lab) and
In-the-Wild (compressed, noisy, variable quality).
"""

import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-level caches: preload audio into RAM for fast augmentation
# ---------------------------------------------------------------------------
_MUSAN_AUDIO: dict[str, list[torch.Tensor]] | None = None
_RIR_AUDIO: list[torch.Tensor] | None = None
_MUSAN_FILES: dict[str, list[str]] | None = None

_MAX_RIR_CACHE = 500   # preload this many RIRs (out of 61K)
_MAX_RIR_LEN = 16000   # truncate RIRs to 1 second (early reflections carry most info)


def _load_audio_file(path: str, sr: int = 16000) -> torch.Tensor:
    """Load full audio file as a 1D float32 tensor at target sample rate."""
    data, file_sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data.astype(np.float32))
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)  # mono
    if file_sr != sr:
        wav = F.interpolate(
            wav.unsqueeze(0).unsqueeze(0),
            size=int(len(wav) * sr / file_sr),
            mode="linear", align_corners=False,
        ).squeeze()
    return wav


def _preload_musan(musan_root: str, sr: int = 16000) -> dict[str, list[torch.Tensor]]:
    """Load all MUSAN audio files into memory (one-time cost)."""
    global _MUSAN_AUDIO, _MUSAN_FILES
    if _MUSAN_AUDIO is not None:
        return _MUSAN_AUDIO

    root = Path(musan_root)
    audio = {}
    _MUSAN_FILES = {}
    for cat in ["noise", "speech", "music"]:
        cat_dir = root / cat
        if not cat_dir.exists():
            audio[cat] = []
            _MUSAN_FILES[cat] = []
            continue
        files = sorted(str(p) for p in cat_dir.rglob("*.wav"))
        _MUSAN_FILES[cat] = files
        loaded = []
        for f in files:
            try:
                loaded.append(_load_audio_file(f, sr))
            except Exception:
                continue
        audio[cat] = loaded
        print(f"  MUSAN/{cat}: {len(loaded)} clips loaded into RAM")

    _MUSAN_AUDIO = audio
    return _MUSAN_AUDIO


def _preload_rir(rir_root: str, sr: int = 16000) -> list[torch.Tensor]:
    """Load a random subset of RIR files into memory."""
    global _RIR_AUDIO
    if _RIR_AUDIO is not None:
        return _RIR_AUDIO

    root = Path(rir_root)
    files = sorted(str(p) for p in root.rglob("*.wav"))
    # Sample a subset for memory efficiency
    if len(files) > _MAX_RIR_CACHE:
        rng = random.Random(42)
        files = rng.sample(files, _MAX_RIR_CACHE)

    loaded = []
    for f in files:
        try:
            rir = _load_audio_file(f, sr)
            # Truncate long RIRs — early reflections carry most useful info
            if rir.shape[0] > _MAX_RIR_LEN:
                rir = rir[:_MAX_RIR_LEN]
            loaded.append(rir)
        except Exception:
            continue

    print(f"  RIR: {len(loaded)} impulse responses loaded into RAM (max {_MAX_RIR_LEN} samples)")
    _RIR_AUDIO = loaded
    return _RIR_AUDIO


def _get_random_segment(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """Extract a random segment of target_len from a preloaded audio tensor."""
    if wav.shape[0] <= target_len:
        return F.pad(wav, (0, target_len - wav.shape[0]))
    start = random.randint(0, wav.shape[0] - target_len)
    return wav[start:start + target_len]


# Backward compat: old functions that scan files (used by individual functions)
def _scan_musan(musan_root: str) -> dict[str, list[str]]:
    """Return MUSAN file paths (also triggers preload)."""
    _preload_musan(musan_root)
    return _MUSAN_FILES


def _scan_rir(rir_root: str) -> list[str]:
    """Return RIR file paths (unused in fast path, kept for compat)."""
    return []  # not needed when using preloaded audio


# ---------------------------------------------------------------------------
# Simple augmentations (original pipeline)
# ---------------------------------------------------------------------------

def add_noise(waveform: torch.Tensor, snr_range: tuple = (5, 25)) -> torch.Tensor:
    """Add Gaussian white noise at random SNR (dB)."""
    snr_db = random.uniform(*snr_range)
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise


def random_lowpass(waveform: torch.Tensor, sr: int = 16000,
                   cutoff_range: tuple = (3000, 7000)) -> torch.Tensor:
    """Simulate bandwidth limiting (codec/phone quality) via simple FIR lowpass."""
    cutoff = random.uniform(*cutoff_range)
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist

    n_taps = 101
    half = n_taps // 2
    n = torch.arange(n_taps, dtype=waveform.dtype) - half
    n_safe = n.clone()
    n_safe[half] = 1.0
    h = torch.sin(2 * torch.pi * normalized_cutoff * n_safe) / (torch.pi * n_safe)
    h[half] = 2 * normalized_cutoff
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(n_taps, dtype=waveform.dtype) / (n_taps - 1))
    h = h * window
    h = h / h.sum()

    h = h.to(waveform.device).unsqueeze(0).unsqueeze(0)
    x = waveform.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (half, half), mode="reflect")
    out = F.conv1d(x, h).squeeze(0).squeeze(0)
    return out


def random_gain(waveform: torch.Tensor, db_range: tuple = (-6, 6)) -> torch.Tensor:
    """Random volume change in dB."""
    gain_db = random.uniform(*db_range)
    return waveform * (10 ** (gain_db / 20))


def random_trim_pad(waveform: torch.Tensor, max_samples: int,
                    shift_range: tuple = (0, 4000)) -> torch.Tensor:
    """Randomly shift the start position, then re-pad to max_samples."""
    shift = random.randint(*shift_range)
    if shift > 0 and shift < waveform.shape[0]:
        waveform = waveform[shift:]
    if waveform.shape[0] >= max_samples:
        waveform = waveform[:max_samples]
    else:
        pad = torch.zeros(max_samples - waveform.shape[0], dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad])
    return waveform


def augment_waveform(waveform: torch.Tensor, sr: int = 16000,
                     max_samples: int = 64000) -> torch.Tensor:
    """Original simple augmentation pipeline (Gaussian noise + lowpass + gain)."""
    if random.random() < 0.5:
        waveform = random_trim_pad(waveform, max_samples)
    if random.random() < 0.7:
        waveform = add_noise(waveform, snr_range=(5, 25))
    if random.random() < 0.4:
        waveform = random_lowpass(waveform, sr=sr, cutoff_range=(3000, 7000))
    if random.random() < 0.5:
        waveform = random_gain(waveform, db_range=(-6, 6))

    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    elif waveform.shape[0] < max_samples:
        pad = torch.zeros(max_samples - waveform.shape[0], dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad])

    return waveform


# ---------------------------------------------------------------------------
# Kaldi-style augmentations (MUSAN + RIR + codec)
# ---------------------------------------------------------------------------

def _mix_at_snr(signal: torch.Tensor, noise: torch.Tensor,
                snr_db: float) -> torch.Tensor:
    """Mix signal and noise at a target SNR (dB)."""
    sig_power = signal.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    scale = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return signal + noise * scale


def add_musan_noise(waveform: torch.Tensor, musan_root: str,
                    sr: int = 16000) -> torch.Tensor:
    """Add MUSAN noise (noise/music subset) at Kaldi recipe SNRs.

    SNR ranges (Snyder et al. 2018):
      - noise: 0-15 dB
      - music: 5-15 dB
    """
    audio = _preload_musan(musan_root, sr)
    target_len = waveform.shape[0]

    cat = random.choice(["noise", "music"])
    if not audio.get(cat):
        return waveform

    snr_ranges = {"noise": (0, 15), "music": (5, 15)}
    snr_db = random.uniform(*snr_ranges[cat])

    clip = random.choice(audio[cat])
    noise = _get_random_segment(clip, target_len).to(waveform.device)

    return _mix_at_snr(waveform, noise, snr_db)


def add_musan_babble(waveform: torch.Tensor, musan_root: str,
                     sr: int = 16000,
                     n_speakers: tuple = (3, 7),
                     snr_range: tuple = (13, 20)) -> torch.Tensor:
    """Add MUSAN babble noise (mix of 3-7 speakers) at 13-20 dB SNR."""
    audio = _preload_musan(musan_root, sr)
    speech_clips = audio.get("speech", [])
    if not speech_clips:
        return waveform

    target_len = waveform.shape[0]
    n = random.randint(*n_speakers)
    n = min(n, len(speech_clips))

    selected = random.sample(speech_clips, n)
    babble = torch.zeros(target_len, dtype=waveform.dtype, device=waveform.device)
    for clip in selected:
        babble = babble + _get_random_segment(clip, target_len).to(waveform.device)

    snr_db = random.uniform(*snr_range)
    return _mix_at_snr(waveform, babble, snr_db)


def add_rir_reverb(waveform: torch.Tensor, rir_root: str,
                   sr: int = 16000) -> torch.Tensor:
    """Convolve waveform with a random Room Impulse Response (from preloaded cache)."""
    rir_cache = _preload_rir(rir_root, sr)
    if not rir_cache:
        return waveform

    rir = random.choice(rir_cache)

    # Normalize RIR (peak at 1.0)
    rir = rir / (rir.abs().max() + 1e-10)
    rir = rir.to(waveform.device)

    # FFT-based convolution (much faster for long signals)
    n = waveform.shape[0] + rir.shape[0] - 1
    fft_size = 1 << (n - 1).bit_length()  # next power of 2
    X = torch.fft.rfft(waveform, n=fft_size)
    H = torch.fft.rfft(rir, n=fft_size)
    out = torch.fft.irfft(X * H, n=fft_size)[:waveform.shape[0]]

    # Truncate to original length
    out = out[:waveform.shape[0]]

    # Normalize energy to match original
    orig_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-10)
    out_rms = out.pow(2).mean().sqrt().clamp(min=1e-10)
    out = out * (orig_rms / out_rms)

    return out


def codec_compress(waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Apply lossy codec compression via ffmpeg (MP3/OGG at random quality).

    Introduces quantization artifacts common in real-world audio.
    Returns original waveform if ffmpeg is unavailable.
    """
    # Pick a codec and quality
    codec_configs = [
        # (format, extension, quality_args)
        ("mp3", ".mp3", ["-b:a", str(random.choice([32, 48, 64, 96, 128])) + "k"]),
        ("ogg", ".ogg", ["-q:a", str(random.choice([0, 1, 2, 3, 4]))]),
    ]
    fmt, ext, quality_args = random.choice(codec_configs)

    wav_np = waveform.numpy().astype(np.float32)

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            in_path = f_in.name
            sf.write(in_path, wav_np, sr, subtype="FLOAT")

        out_path = in_path.replace(".wav", ext)
        back_path = in_path.replace(".wav", "_back.wav")

        # Encode to lossy codec
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", str(sr)] + quality_args + [out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
        # Decode back to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path, "-ar", str(sr), back_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )

        data, _ = sf.read(back_path, dtype="float32")
        result = torch.from_numpy(data.astype(np.float32))
        if result.ndim > 1:
            result = result.mean(dim=-1)

        # Match length
        target_len = waveform.shape[0]
        if result.shape[0] >= target_len:
            result = result[:target_len]
        else:
            result = F.pad(result, (0, target_len - result.shape[0]))

        return result

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return waveform
    finally:
        for p in [in_path, out_path, back_path]:
            try:
                os.unlink(p)
            except (OSError, UnboundLocalError):
                pass


# ---------------------------------------------------------------------------
# Kaldi-style augmentation pipeline
# ---------------------------------------------------------------------------

# Kaldi x-vector recipe: 5 augmented copies per sample
# Copy 0: original (clean)
# Copy 1: RIR reverb
# Copy 2: RIR reverb + MUSAN noise
# Copy 3: MUSAN music
# Copy 4: MUSAN babble
# Copy 5: MUSAN noise
#
# We add codec compression as a 6th option since it's a primary domain
# shift in In-the-Wild audio.

# Original Kaldi x-vector recipe: 5 augmentation types (no codec)
# Codec is available but slow (ffmpeg subprocess per sample) — enable explicitly
KALDI_AUG_TYPES = ["reverb", "reverb_noise", "music", "babble", "noise"]


def augment_waveform_kaldi(
    waveform: torch.Tensor,
    sr: int = 16000,
    max_samples: int = 64000,
    musan_root: str | None = None,
    rir_root: str | None = None,
    aug_type: str | None = None,
) -> torch.Tensor:
    """Kaldi-style augmentation with MUSAN, RIR, and codec compression.

    Args:
        waveform: (T,) raw audio tensor
        sr: sample rate
        max_samples: target length
        musan_root: path to MUSAN corpus (required for noise/babble/music)
        rir_root: path to RIR dataset (required for reverb)
        aug_type: specific augmentation type, or None for random selection.
                  One of: reverb, reverb_noise, music, babble, noise, codec

    Returns:
        augmented waveform of shape (max_samples,)
    """
    # Determine which augmentation to apply
    if aug_type is None:
        # Filter to available augmentations
        available = []
        if rir_root:
            available.append("reverb")
        if rir_root and musan_root:
            available.append("reverb_noise")
        if musan_root:
            available.extend(["music", "babble", "noise"])
        # codec is available but slow (ffmpeg per sample); enable via aug_type="codec"
        if not available:
            # Fallback to simple augmentation
            return augment_waveform(waveform, sr, max_samples)

        aug_type = random.choice(available)

    # Random trim/pad (50% chance — same as simple pipeline)
    if random.random() < 0.5:
        waveform = random_trim_pad(waveform, max_samples)

    # Apply the selected augmentation
    if aug_type == "reverb":
        if rir_root:
            waveform = add_rir_reverb(waveform, rir_root, sr)

    elif aug_type == "reverb_noise":
        if rir_root:
            waveform = add_rir_reverb(waveform, rir_root, sr)
        if musan_root:
            waveform = add_musan_noise(waveform, musan_root, sr)

    elif aug_type == "music":
        if musan_root:
            audio = _preload_musan(musan_root, sr)
            if audio.get("music"):
                target_len = waveform.shape[0]
                snr_db = random.uniform(5, 15)
                clip = random.choice(audio["music"])
                noise = _get_random_segment(clip, target_len).to(waveform.device)
                waveform = _mix_at_snr(waveform, noise, snr_db)

    elif aug_type == "babble":
        if musan_root:
            waveform = add_musan_babble(waveform, musan_root, sr)

    elif aug_type == "noise":
        if musan_root:
            waveform = add_musan_noise(waveform, musan_root, sr)

    elif aug_type == "codec":
        waveform = codec_compress(waveform, sr)

    # Random gain (50% chance)
    if random.random() < 0.5:
        waveform = random_gain(waveform, db_range=(-6, 6))

    # Ensure correct length
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    elif waveform.shape[0] < max_samples:
        pad = torch.zeros(max_samples - waveform.shape[0], dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad])

    return waveform
