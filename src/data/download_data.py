#!/usr/bin/env python3
"""
Download and organize datasets for deepfake detection experiments.

Datasets:
  1. ASVspoof 2019 LA  -- primary training/eval set
  2. WaveFake          -- cross-domain generalization test
  3. In-the-Wild       -- cross-domain generalization test

Some datasets require manual download due to licensing.
This script checks what's present and handles organization.
"""

import os
import sys
import subprocess
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ASVSPOOF_DIR = RAW / "asvspoof2019"
WAVEFAKE_DIR = RAW / "wavefake"
INTHEWILD_DIR = RAW / "in_the_wild"


def check_tool(name):
    """Check if a command-line tool is available."""
    return shutil.which(name) is not None


# ── ASVspoof 2019 LA ──────────────────────────────────────────────
def setup_asvspoof():
    """
    ASVspoof 2019 LA dataset.
    
    Source: https://datashare.ed.ac.uk/handle/10283/3336
    
    You need to download these files manually:
      - LA.zip (or LA_train.zip, LA_dev.zip, LA_eval.zip)
      - LA protocol files
    
    Expected structure after setup:
      data/raw/asvspoof2019/
        LA/
          ASVspoof2019_LA_train/flac/
          ASVspoof2019_LA_dev/flac/
          ASVspoof2019_LA_eval/flac/
          ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
            ASVspoof2019.LA.cm.eval.trl.txt
    """
    print("\n" + "="*60)
    print("ASVspoof 2019 LA")
    print("="*60)
    
    proto_path = ASVSPOOF_DIR / "LA" / "ASVspoof2019_LA_cm_protocols"
    train_path = ASVSPOOF_DIR / "LA" / "ASVspoof2019_LA_train" / "flac"
    
    if proto_path.exists() and train_path.exists():
        n_train = len(list(train_path.glob("*.flac")))
        print(f"  [OK] Found ASVspoof 2019 LA ({n_train} train files)")
        return True
    
    # Check if zip was dropped in raw/
    zips = list(RAW.glob("LA*.zip")) + list(RAW.glob("asvspoof*.zip"))
    if zips:
        print(f"  Found archive(s): {[z.name for z in zips]}")
        ASVSPOOF_DIR.mkdir(parents=True, exist_ok=True)
        for z in zips:
            print(f"  Extracting {z.name}...")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(ASVSPOOF_DIR)
        return True
    
    print("  [MISSING] ASVspoof 2019 LA not found.")
    print()
    print("  Download instructions:")
    print("    1. Go to: https://datashare.ed.ac.uk/handle/10283/3336")
    print("    2. Download the LA partition files")
    print("    3. Place the zip(s) in: data/raw/")
    print("    4. Re-run this script")
    print()
    print("  Alternative (wget):")
    print("    The dataset is also available via Zenodo and various mirrors.")
    print("    Check the ASVspoof 2019 website for current download links.")
    return False


# ── WaveFake ──────────────────────────────────────────────────────
def setup_wavefake():
    """
    WaveFake dataset (Frank & Schonherr, 2021).
    
    Source: https://zenodo.org/record/5642694
    
    Contains real LJSpeech samples + generated samples from:
      MelGAN, FB-MelGAN, MB-MelGAN, PWG, WaveGlow, HiFi-GAN
    
    Expected structure after setup:
      data/raw/wavefake/
        LJSpeech-1.1_16k/          (real)
        ljspeech_melgan/            (fake)
        ljspeech_full_band_melgan/  (fake)
        ljspeech_multi_band_melgan/ (fake)
        ljspeech_parallel_wavegan/  (fake)
        ljspeech_waveglow/          (fake)
        ljspeech_hifiGAN/           (fake)
    """
    print("\n" + "="*60)
    print("WaveFake")
    print("="*60)
    
    if WAVEFAKE_DIR.exists() and any(WAVEFAKE_DIR.iterdir()):
        subdirs = [d.name for d in WAVEFAKE_DIR.iterdir() if d.is_dir()]
        print(f"  [OK] Found WaveFake ({len(subdirs)} subdirectories)")
        for sd in sorted(subdirs):
            n = len(list((WAVEFAKE_DIR / sd).glob("*.wav")))
            print(f"       {sd}: {n} files")
        return True
    
    print("  [MISSING] WaveFake not found.")
    print()
    print("  Download instructions:")
    print("    1. Go to: https://zenodo.org/record/5642694")
    print("    2. Download all .zip files")
    print("    3. Extract into: data/raw/wavefake/")
    print("    4. Re-run this script")
    print()
    print("  Or use zenodo_get:")
    print("    pip install zenodo_get")
    print("    cd data/raw/wavefake && zenodo_get 5642694")
    return False


# ── In-the-Wild ───────────────────────────────────────────────────
def setup_in_the_wild():
    """
    In-the-Wild dataset (Muller et al., 2022).
    
    Source: https://inthewild.aimlab.ca/ 
    or GitHub: https://github.com/piotrkawa/deepfake-whisper-features
    (commonly redistributed via release_in_the_wild.zip)
    
    Expected structure:
      data/raw/in_the_wild/
        release_in_the_wild/
          *.wav
        meta.csv
    """
    print("\n" + "="*60)
    print("In-the-Wild")
    print("="*60)
    
    if INTHEWILD_DIR.exists() and any(INTHEWILD_DIR.rglob("*.wav")):
        n = len(list(INTHEWILD_DIR.rglob("*.wav")))
        print(f"  [OK] Found In-the-Wild ({n} wav files)")
        return True
    
    # Check for zip
    zips = list(RAW.glob("*in_the_wild*.zip")) + list(RAW.glob("*inthewild*.zip"))
    if zips:
        INTHEWILD_DIR.mkdir(parents=True, exist_ok=True)
        for z in zips:
            print(f"  Extracting {z.name}...")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(INTHEWILD_DIR)
        return True
    
    print("  [MISSING] In-the-Wild not found.")
    print()
    print("  Download instructions:")
    print("    1. Visit: https://inthewild.aimlab.ca/")
    print("    2. Download release_in_the_wild.zip")
    print("    3. Place in: data/raw/")
    print("    4. Re-run this script")
    return False


# ── main ──────────────────────────────────────────────────────────
def main():
    print("Deepfake Detection -- Dataset Setup")
    print("="*60)
    print(f"Data root: {RAW}")
    
    RAW.mkdir(parents=True, exist_ok=True)
    ASVSPOOF_DIR.mkdir(parents=True, exist_ok=True)
    WAVEFAKE_DIR.mkdir(parents=True, exist_ok=True)
    INTHEWILD_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    results["asvspoof"] = setup_asvspoof()
    results["wavefake"] = setup_wavefake()
    results["in_the_wild"] = setup_in_the_wild()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, ok in results.items():
        status = "READY" if ok else "NEEDS DOWNLOAD"
        print(f"  {name:20s} [{status}]")
    
    if all(results.values()):
        print("\nAll datasets ready! Run preprocessing next.")
    else:
        print("\nSome datasets missing. Follow instructions above.")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
