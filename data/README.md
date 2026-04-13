# Data

This directory is intentionally empty. Run the download scripts to populate it:

```bash
# ASVspoof 2019 LA + In-the-Wild
python -m src.data.download_data

# MUSAN + RIR for Kaldi augmentation
python -m src.data.download_augmentation_data
```

After downloading, the expected structure is:

```
data/raw/
  asvspoof2019/LA/
    ASVspoof2019_LA_train/...
    ASVspoof2019_LA_dev/...
    ASVspoof2019_LA_eval/...
  in_the_wild/
    release_in_the_wild/...
  musan/
    noise/...
    music/...
    speech/...
  RIRS_NOISES/
    simulated_rirs/...
```

Precomputed features will be saved under `data/processed/` after running `src.data.precompute`.
