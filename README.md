# Wavelet Scattering + Spiking Neural Networks for Audio Deepfake Detection

An exploratory study of wavelet scattering transforms and spiking neural networks for cross-domain audio deepfake detection. Trained on ASVspoof 2019 LA, evaluated on In-the-Wild (Muller et al., 2022). This represents the first SNN evaluation on the In-the-Wild benchmark.

**Headline result:** 25.65% In-the-Wild EER with a 136K-parameter wavelet+SNN model, competitive with AASIST (23.55%, 297K params; Jung et al., 2022; Schafer et al., 2024) and within 6pp of WavLM-Large SSL (19.61%, 317M params).

## Results

| Model | Params | ITW EER | Source |
|-------|--------|---------|--------|
| RawGAT-ST + aug | 437K | 18.08% | Schafer et al. |
| WavLM-Large + probe | 317M + 25K | 19.61% | this work |
| AASIST + aug | 297K | 23.55% | Schafer et al. |
| **Wavelet + SNN + Kaldi** | **136K** | **25.65%** | **this work** |
| AASIST-L + aug | 85K | 26.86% | Schafer et al. |
| RawNet2, no aug | 17.6M | 33.94% | Muller et al. |

Parameter counts for AASIST/AASIST-L/RawGAT-ST from Jung et al. (2022). ITW EER numbers for Schafer et al. are from their preparatory tests (Table 1), trained on ASVspoof 2019 LA with Gaussian noise + MP3 compression augmentation.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the datasets and place them under `data/raw/`:

```bash
# Check dataset status and get download instructions
python -m src.data.download_data

# Download MUSAN + RIR for Kaldi-style augmentation
python -m src.data.download_augmentation_data
```

Expected structure:
```
data/raw/
  asvspoof2019/LA/...
  in_the_wild/...
  musan/...
  RIRS_NOISES/...
```

## Precompute Features

Feature extraction (especially wavelet scattering) is slow. Precompute once, then train fast:

```bash
# Precompute mel + wavelet features for train/dev splits
python -m src.data.precompute --config configs/default.yaml \
  --frontends mel wavelet_v3_order1 --splits train dev

# Precompute with Kaldi augmentation (5 copies per sample)
python -m src.data.precompute --config configs/default.yaml \
  --frontends wavelet_v3_order1 --splits train dev \
  --augment 5 --augment-mode kaldi \
  --musan-root data/raw/musan --rir-root data/raw/RIRS_NOISES

# Compute per-channel normalization stats
python -m src.data.precompute --config configs/default.yaml \
  --frontends wavelet_v3_order1 --splits train --compute-stats
```

## Training

```bash
# Wavelet + SNN (best config)
python -m src.training.train \
  --frontend wavelet_v3_order1 --model snn \
  --precomputed --augmented --augment-mode kaldi --cross-eval

# Mel + ResNet baseline
python -m src.training.train \
  --frontend mel --model resnet \
  --precomputed --augmented --augment-mode kaldi
```

## Evaluation

```bash
# Precompute eval features
python -m src.data.precompute --config configs/default.yaml \
  --frontends wavelet_v3_order1 --splits eval

# Evaluate checkpoint on ASVspoof eval + In-the-Wild
python -m src.evaluation.evaluate_precomputed \
  --checkpoint results/checkpoints/wavelet_v3_order1_snn_kaldi/best.pt \
  --frontend wavelet_v3_order1 \
  --output-name wavelet_order1_snn_kaldi \
  --raw-datasets in_the_wild
```

## SSL Baseline

```bash
# Extract WavLM-Large features (requires GPU, ~317M params frozen)
python -m src.models.ssl_baseline extract --config configs/default.yaml

# Train linear probe
python -m src.models.ssl_baseline train --config configs/default.yaml

# Evaluate
python -m src.models.ssl_baseline eval --config configs/default.yaml
```

## Paper

The paper is in `paper/draft.md` (source) and `paper/draft.pdf` (rendered). Key findings:

1. Wavelet scattering regularizes against domain memorization (2.9x smaller generalization gap than mel)
2. Kaldi augmentation activates latent cross-domain features in the wavelet representation
3. The wavelet-augmentation interaction is amplified within the SNN architecture relative to ResNet

All results use a single random seed; multi-seed replication is needed for statistical significance.

## Citation

```bibtex
@misc{collins2026wavelet,
  title={Wavelet Scattering and Spiking Neural Networks for Cross-Domain Audio Deepfake Detection: An Exploratory Study},
  author={Collins, Micah},
  year={2026},
  note={DATASCI 266, UC Berkeley MIDS}
}
```

## References

- Muller et al. (2022). "Does audio deepfake detection generalize?" Proc. Interspeech.
- Jung et al. (2022). "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks." Proc. ICASSP.
- Schafer et al. (2024). "Robust audio deepfake detection." Proc. ASVspoof Workshop.
- Xuan et al. (2026). "WST-X series." arXiv:2602.02980.
