# Wavelet Scattering + Spiking Neural Networks for Audio Deepfake Detection

Factorial study of wavelet scattering transforms and spiking neural networks for cross-domain audio deepfake detection. Trained on ASVspoof 2019 LA, evaluated on In-the-Wild (Muller et al., 2022).

**Headline result:** 25.65% In-the-Wild EER with a 136K-parameter wavelet+SNN model, competitive with AASIST (23.55%, 297K params) and within 6pp of WavLM-Large SSL (19.61%, 315M params).

## Results

| Model | Params | ITW EER |
|-------|--------|---------|
| RawGAT-ST + aug (Schafer et al., 2024) | 437K | 18.08% |
| WavLM-Large + linear probe (this work) | 315M + 25K | 19.61% |
| AASIST + aug (Schafer et al., 2024) | 297K | 23.55% |
| **Wavelet order-1 + SNN + Kaldi (ours)** | **136K** | **25.65%** |
| AASIST-L + aug (Schafer et al., 2024) | 85K | 26.86% |
| RawNet2, no aug (Muller et al., 2022) | 17.6M | 33.94% |

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
  wavefake/...
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
  --precomputed --augmented --augment-mode kaldi

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

# Evaluate checkpoint
python -m src.evaluation.evaluate \
  --checkpoint results/checkpoints/wavelet_v3_order1_snn_kaldi/best.pt
```

## SSL Baseline

```bash
# Extract WavLM-Large features (requires GPU, ~315M params frozen)
python -m src.models.ssl_baseline extract --config configs/default.yaml

# Train linear probe
python -m src.models.ssl_baseline train --config configs/default.yaml

# Evaluate
python -m src.models.ssl_baseline eval --config configs/default.yaml
```

## Citation

```bibtex
@misc{collins2026wavelet,
  title={Frontend Choice Dominates Cross-Domain Generalization in Audio Deepfake Detection},
  author={Collins, Micah},
  year={2026},
  note={DATASCI 266, UC Berkeley MIDS}
}
```

## References

- Muller et al. (2022). "Does audio deepfake detection generalize?" Proc. Interspeech.
- Schafer et al. (2024). "Robust audio deepfake detection." Proc. ASVspoof Workshop.
- Xuan et al. (2026). "WST-X series: Wavelet scattering transform for interpretable speech deepfake detection." arXiv:2602.02980.
