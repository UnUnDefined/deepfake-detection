#!/usr/bin/env python3
"""
Download MUSAN corpus and RIRS_NOISES for Kaldi-style augmentation.

Sources:
  - MUSAN: https://www.openslr.org/17/  (~10 GB, noise/speech/music)
  - RIRS_NOISES: https://www.openslr.org/28/  (~600 MB, simulated RIRs)

Usage:
    python -m src.data.download_augmentation_data
    python -m src.data.download_augmentation_data --musan-only
    python -m src.data.download_augmentation_data --rir-only
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def download_file(url: str, dest: Path):
    """Download a file using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    # Try wget first, then curl
    for cmd in [
        ["wget", "-O", str(dest), url],
        ["curl", "-L", "-o", str(dest), url],
    ]:
        try:
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print("ERROR: Neither wget nor curl available. Please install one.")
    sys.exit(1)


def setup_musan():
    """Download and extract MUSAN corpus."""
    musan_dir = RAW / "musan"
    if musan_dir.exists() and any(musan_dir.iterdir()):
        n_files = sum(1 for _ in musan_dir.rglob("*.wav"))
        print(f"  MUSAN already exists: {musan_dir} ({n_files} wav files)")
        return

    archive = RAW / "musan.tar.gz"
    download_file("https://www.openslr.org/resources/17/musan.tar.gz", archive)

    print(f"  Extracting to {RAW}...")
    subprocess.run(["tar", "xzf", str(archive), "-C", str(RAW)], check=True)

    n_files = sum(1 for _ in musan_dir.rglob("*.wav"))
    print(f"  MUSAN ready: {n_files} wav files")

    # Optionally remove archive
    archive_size = archive.stat().st_size / (1024 ** 3)
    print(f"  Archive size: {archive_size:.1f} GB (delete manually if needed: {archive})")


def setup_rir():
    """Download and extract RIRS_NOISES dataset."""
    rir_dir = RAW / "RIRS_NOISES"
    if rir_dir.exists() and any(rir_dir.iterdir()):
        n_files = sum(1 for _ in rir_dir.rglob("*.wav"))
        print(f"  RIRS_NOISES already exists: {rir_dir} ({n_files} wav files)")
        return

    archive = RAW / "rirs_noises.zip"
    download_file(
        "https://www.openslr.org/resources/28/rirs_noises.zip", archive
    )

    print(f"  Extracting to {RAW}...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(RAW)

    n_files = sum(1 for _ in rir_dir.rglob("*.wav"))
    print(f"  RIRS_NOISES ready: {n_files} wav files")

    archive_size = archive.stat().st_size / (1024 ** 3)
    print(f"  Archive size: {archive_size:.1f} GB (delete manually if needed: {archive})")


def main():
    parser = argparse.ArgumentParser(
        description="Download MUSAN and RIRS_NOISES for Kaldi-style augmentation"
    )
    parser.add_argument("--musan-only", action="store_true")
    parser.add_argument("--rir-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Augmentation Data Download")
    print("=" * 60)

    if not args.rir_only:
        print("\n[1/2] MUSAN corpus (~10 GB)")
        setup_musan()

    if not args.musan_only:
        print("\n[2/2] RIRS_NOISES dataset (~600 MB)")
        setup_rir()

    print("\n" + "=" * 60)
    print("  Done! Configure paths in configs/default.yaml:")
    print(f"    musan_root: data/raw/musan")
    print(f"    rir_root: data/raw/RIRS_NOISES")
    print("=" * 60)


if __name__ == "__main__":
    main()
