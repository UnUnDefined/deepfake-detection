#!/usr/bin/env python3
"""
Precompute features (mel + wavelet) to disk as .pt files.

Both frontends are deterministic — same waveform always produces the same
features. Precomputing eliminates redundant Kymatio scattering on every
epoch, cutting wavelet training time from ~10 min/epoch to ~2 min/epoch.

Usage:
    python -m src.data.precompute --config configs/default.yaml

Output structure:
    data/processed/
        asvspoof_train/
            mel/        0000.pt, 0001.pt, ...
            wavelet/    0000.pt, 0001.pt, ...
            labels.pt   (int tensor)
            metadata.json
        asvspoof_dev/
            mel/        ...
            wavelet/    ...
            labels.pt
            metadata.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.datasets import ASVspoof2019LA, InTheWild
from src.data.features import get_frontend
from src.data.augment import augment_waveform, augment_waveform_kaldi


def precompute_split(
    dataset,
    frontend: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Precompute features for one dataset split with one frontend."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend.eval()
    labels = []
    metadata_list = []
    n = len(dataset)

    with torch.no_grad():
        for i in tqdm(range(n), desc=f"  {output_dir.name}"):
            waveform, label, meta = dataset[i]
            waveform = waveform.unsqueeze(0).to(device)  # (1, T)
            features = frontend(waveform)  # (1, 1, freq, time)
            features = features.squeeze(0).cpu()  # (1, freq, time)

            torch.save(features, output_dir / f"{i:06d}.pt")
            labels.append(label)
            metadata_list.append({
                k: str(v) if isinstance(v, Path) else v
                for k, v in meta.items()
            })

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(labels_tensor, output_dir.parent / "labels.pt")

    # Save shape info
    sample = torch.load(output_dir / "000000.pt", weights_only=True)
    info = {
        "n_samples": n,
        "feature_shape": list(sample.shape),
        "freq_dim": sample.shape[1],
        "time_dim": sample.shape[2],
    }
    return info


def precompute_augmented(
    dataset,
    frontend: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    n_copies: int = 5,
    sample_rate: int = 16000,
    max_samples: int = 64000,
    aug_mode: str = "simple",
    musan_root: str | None = None,
    rir_root: str | None = None,
) -> dict:
    """Precompute features for original + N augmented copies per sample.

    Output: copy 0 = original, copies 1..N = augmented.
    File naming: {original_idx * (n_copies+1) + copy_idx}:06d.pt
    Total samples: len(dataset) * (n_copies + 1)

    aug_mode: "simple" (Gaussian noise) or "kaldi" (MUSAN/RIR/codec)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend.eval()

    labels = []
    n = len(dataset)
    total = n * (n_copies + 1)
    out_idx = 0

    print(f"  Augmentation mode: {aug_mode}")
    if aug_mode == "kaldi":
        if musan_root:
            print(f"  MUSAN root: {musan_root}")
        if rir_root:
            print(f"  RIR root: {rir_root}")

    with torch.no_grad():
        for i in tqdm(range(n), desc=f"  {output_dir.name} (aug x{n_copies})"):
            waveform, label, meta = dataset[i]

            # Copy 0: original (unaugmented)
            wav = waveform.unsqueeze(0).to(device)  # (1, T)
            features = frontend(wav).squeeze(0).cpu()  # (1, freq, time)
            torch.save(features, output_dir / f"{out_idx:06d}.pt")
            labels.append(label)
            out_idx += 1

            # Copies 1..N: augmented
            for _ in range(n_copies):
                if aug_mode == "kaldi":
                    aug_wav = augment_waveform_kaldi(
                        waveform.clone(), sr=sample_rate,
                        max_samples=max_samples,
                        musan_root=musan_root, rir_root=rir_root,
                    )
                else:
                    aug_wav = augment_waveform(
                        waveform.clone(), sr=sample_rate,
                        max_samples=max_samples,
                    )
                aug_wav = aug_wav.unsqueeze(0).to(device)
                features = frontend(aug_wav).squeeze(0).cpu()
                torch.save(features, output_dir / f"{out_idx:06d}.pt")
                labels.append(label)
                out_idx += 1

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(labels_tensor, output_dir.parent / "labels.pt")

    sample = torch.load(output_dir / "000000.pt", weights_only=True)
    info = {
        "n_samples": total,
        "n_original": n,
        "n_copies": n_copies,
        "feature_shape": list(sample.shape),
        "freq_dim": sample.shape[1],
        "time_dim": sample.shape[2],
    }
    return info


def compute_channel_stats(feature_dir: Path, n_samples: int) -> dict:
    """
    Compute per-channel mean and std from precomputed feature files.

    Streams through all files to compute stats in a single pass using
    Welford's online algorithm (numerically stable).

    Returns dict with 'mean' and 'std' tensors of shape (n_channels,).
    """
    print(f"  Computing per-channel stats from {feature_dir} ({n_samples} files)...")

    running_sum = None
    running_sum_sq = None
    total_frames = 0

    for i in tqdm(range(n_samples), desc="  stats"):
        feat = torch.load(feature_dir / f"{i:06d}.pt", weights_only=True)
        # feat: (1, freq_dim, time_dim)
        feat = feat.squeeze(0)  # (freq_dim, time)
        n_frames = feat.shape[1]

        if running_sum is None:
            running_sum = feat.sum(dim=1).double()       # (freq_dim,)
            running_sum_sq = (feat ** 2).sum(dim=1).double()
        else:
            running_sum += feat.sum(dim=1).double()
            running_sum_sq += (feat ** 2).sum(dim=1).double()
        total_frames += n_frames

    mean = (running_sum / total_frames).float()
    var = (running_sum_sq / total_frames - mean.double() ** 2).float().clamp(min=1e-8)
    std = torch.sqrt(var)

    stats = {"mean": mean, "std": std, "n_samples": n_samples, "n_frames": total_frames}

    stats_path = feature_dir / "channel_stats.pt"
    torch.save(stats, stats_path)
    print(f"  Saved: {stats_path}")
    print(f"  Channels: {mean.shape[0]}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range:  [{std.min():.3f}, {std.max():.3f}]")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Precompute features to disk")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--frontends", nargs="+", default=["mel", "wavelet"],
        help="Which frontends to precompute",
    )
    parser.add_argument(
        "--splits", nargs="*", default=["train", "dev"],
        help="Which ASVspoof splits to precompute (omit for none)",
    )
    parser.add_argument(
        "--compute-stats", action="store_true",
        help="Compute per-channel normalization stats from existing precomputed files",
    )
    parser.add_argument(
        "--probe-inthewild", type=int, default=0, metavar="N",
        help="Precompute N random In-the-Wild samples as a cross-domain probe set",
    )
    parser.add_argument(
        "--augment", type=int, default=0, metavar="N",
        help="Generate N augmented copies per sample (0=no augmentation). "
             "Output goes to asvspoof_train_aug/ instead of asvspoof_train/.",
    )
    parser.add_argument(
        "--augment-mode", choices=["simple", "kaldi"], default="simple",
        help="Augmentation mode: 'simple' (Gaussian noise/lowpass/gain) "
             "or 'kaldi' (MUSAN/RIR/codec, Snyder et al. 2018)",
    )
    parser.add_argument(
        "--musan-root", type=str, default=None,
        help="Path to MUSAN corpus (required for --augment-mode kaldi)",
    )
    parser.add_argument(
        "--rir-root", type=str, default=None,
        help="Path to RIR dataset (required for --augment-mode kaldi reverb)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = config["data"]
    processed_root = Path(data_cfg["processed_root"])

    # --- Compute stats mode: scan existing files, save channel_stats.pt ---
    if args.compute_stats:
        for split in args.splits:
            split_dir = processed_root / f"asvspoof_{split}"
            for frontend_name in args.frontends:
                feature_dir = split_dir / frontend_name
                if not feature_dir.exists():
                    print(f"  Skipping {feature_dir} (not found)")
                    continue
                labels_path = split_dir / "labels.pt"
                n_samples = len(torch.load(labels_path, weights_only=True))
                compute_channel_stats(feature_dir, n_samples)
        print("\nDone computing stats.")
        return

    # --- Probe set: precompute a small In-the-Wild subset ---
    if args.probe_inthewild > 0:
        from torch.utils.data import Subset

        print(f"\n{'='*50}")
        print(f"  In-the-Wild probe set ({args.probe_inthewild} samples)")
        print(f"{'='*50}")

        itw = InTheWild(
            root=data_cfg["inthewild_root"],
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        # Deterministic subsample
        rng = np.random.RandomState(42)
        indices = rng.choice(len(itw), size=min(args.probe_inthewild, len(itw)), replace=False)
        indices.sort()
        probe = Subset(itw, indices.tolist())
        print(f"  Full dataset: {len(itw)} | Probe: {len(probe)}")

        probe_dir = processed_root / "inthewild_probe"
        for frontend_name in args.frontends:
            print(f"\n  Frontend: {frontend_name}")
            t0 = time.time()
            frontend = get_frontend(frontend_name, config["features"]).to(device)
            output_dir = probe_dir / frontend_name
            info = precompute_split(probe, frontend, output_dir, device)
            elapsed = time.time() - t0
            print(f"  Shape: {info['feature_shape']} | Time: {elapsed:.1f}s")
            del frontend
            torch.cuda.empty_cache()

        print(f"\nProbe set saved to {probe_dir}")
        if not args.splits:
            return

    # --- Augmented precompute mode ---
    if args.augment > 0 and args.splits and "train" in args.splits:
        print(f"\n{'='*50}")
        print(f"  Augmented precompute ({args.augment} copies per sample)")
        print(f"{'='*50}")

        dataset = ASVspoof2019LA(
            root=data_cfg["asvspoof_root"],
            split="train",
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        print(f"  Original samples: {len(dataset)}")
        print(f"  Total after augmentation: {len(dataset) * (args.augment + 1)}")

        aug_suffix = "kaldi" if args.augment_mode == "kaldi" else "aug"
        aug_dir = processed_root / f"asvspoof_train_{aug_suffix}"
        all_info = {}

        for frontend_name in args.frontends:
            print(f"\n  Frontend: {frontend_name}")
            t0 = time.time()

            frontend = get_frontend(frontend_name, config["features"]).to(device)
            output_dir = aug_dir / frontend_name

            info = precompute_augmented(
                dataset, frontend, output_dir, device,
                n_copies=args.augment,
                sample_rate=data_cfg["sample_rate"],
                max_samples=data_cfg["max_samples"],
                aug_mode=args.augment_mode,
                musan_root=args.musan_root,
                rir_root=args.rir_root,
            )
            elapsed = time.time() - t0

            info["frontend"] = frontend_name
            info["elapsed_sec"] = round(elapsed, 1)
            all_info[f"train_aug_{frontend_name}"] = info

            print(f"  Shape: {info['feature_shape']}")
            print(f"  Total samples: {info['n_samples']}")
            print(f"  Time: {elapsed:.1f}s")

            del frontend
            torch.cuda.empty_cache()

        # Compute channel stats on augmented set
        for frontend_name in args.frontends:
            feature_dir = aug_dir / frontend_name
            n_samples = len(torch.load(aug_dir / "labels.pt", weights_only=True))
            compute_channel_stats(feature_dir, n_samples)

        # Also precompute dev (unaugmented) if not already done
        remaining_splits = [s for s in args.splits if s != "train"]
        if remaining_splits:
            for split in remaining_splits:
                dataset_dev = ASVspoof2019LA(
                    root=data_cfg["asvspoof_root"], split=split,
                    max_samples=data_cfg["max_samples"],
                    sample_rate=data_cfg["sample_rate"],
                )
                split_dir = processed_root / f"asvspoof_{split}"
                for frontend_name in args.frontends:
                    feat_dir = split_dir / frontend_name
                    if feat_dir.exists():
                        print(f"\n  [SKIP] {feat_dir} already exists")
                        continue
                    print(f"\n  Dev split: {frontend_name}")
                    frontend = get_frontend(frontend_name, config["features"]).to(device)
                    precompute_split(dataset_dev, frontend, feat_dir, device)
                    del frontend
                    torch.cuda.empty_cache()

        meta_path = processed_root / "precompute_info.json"
        with open(meta_path, "w") as f:
            json.dump(all_info, f, indent=2)
        print(f"\nDone. Augmented data at {aug_dir}")
        return

    # --- Normal precompute mode ---
    all_info = {}

    for split in args.splits:
        print(f"\n{'='*50}")
        print(f"  Split: {split}")
        print(f"{'='*50}")

        dataset = ASVspoof2019LA(
            root=data_cfg["asvspoof_root"],
            split=split,
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        print(f"  Samples: {len(dataset)}")

        split_dir = processed_root / f"asvspoof_{split}"

        for frontend_name in args.frontends:
            print(f"\n  Frontend: {frontend_name}")
            t0 = time.time()

            frontend = get_frontend(frontend_name, config["features"]).to(device)
            output_dir = split_dir / frontend_name

            info = precompute_split(dataset, frontend, output_dir, device)
            elapsed = time.time() - t0

            info["frontend"] = frontend_name
            info["split"] = split
            info["elapsed_sec"] = round(elapsed, 1)
            all_info[f"{split}_{frontend_name}"] = info

            print(f"  Shape: {info['feature_shape']}")
            print(f"  Time: {elapsed:.1f}s")

            del frontend
            torch.cuda.empty_cache()

    # Save metadata
    meta_path = processed_root / "precompute_info.json"
    with open(meta_path, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f"\nMetadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
