# Wavelet Scattering + Spiking Neural Networks for Audio Deepfake Detection

An exploratory study of wavelet scattering transforms paired with spiking neural networks (SNNs) for cross-domain audio deepfake detection. Trained on ASVspoof 2019 LA, evaluated on the In-the-Wild benchmark (Müller et al., 2022).

**Headline result:** 25.65% In-the-Wild EER with a 136K-parameter wavelet+SNN model, competitive with augmented non-SSL systems in the same parameter class (AASIST-L: 26.86% at 85K params; AASIST: 23.55% at 297K params; Schäfer et al., 2024).

## Key findings

The study was designed around three structural predictions, made before running the experiments:

1. **SNN backends should benefit from temporally-preserving frontends more than CNN backends do**, because SNN dynamics are natively temporal and spectrograms discard the fine temporal structure that SNNs are built to exploit.
2. **Wavelet scattering is the right temporally-preserving frontend for this task**, because wavelet decompositions capture transient artifacts that spectral averaging smears — the same property that makes them useful for industrial vibration analysis, where synthesis-artifact-like transients are the signal of interest.
3. **Kaldi-style augmentation is the activating condition**, because cross-domain generalization requires corrupting dataset-specific shortcuts before the wavelet features can be selected for.

The 2x2 exploratory design (frontend × backend, with and without augmentation) was designed to test these predictions. All three landed:

- Wavelet scattering yields a **24pp** improvement over mel with the SNN backend, vs. **17pp** with the ResNet backend — the asymmetric interaction predicted by (1).
- Wavelet+SNN achieves a **2.9x smaller generalization gap** (ASVspoof eval → ITW) than mel+SNN at matched training conditions, consistent with (2).
- Without Kaldi augmentation, all SNN frontends collapse to ~51% ITW EER (chance); with it, wavelet improves by 21-26pp while mel *worsens*, consistent with (3).

This is the first SNN evaluation on the In-the-Wild benchmark. The only prior SNN work in audio deepfake detection (SAFE, withdrawn from ICLR 2025) used spectrograms and was not evaluated on cross-domain real-world audio.

## Results

| Model | Params | ITW EER | Source |
|-------|--------|---------|--------|
| RawGAT-ST + aug | 437K | 18.08% | Schäfer et al. (2024) |
| WavLM-Large + linear probe | 317M + 25K | 19.61% | this work |
| AASIST + aug | 297K | 23.55% | Schäfer et al. (2024) |
| **Wavelet + SNN + Kaldi** | **136K** | **25.65%** | **this work** |
| AASIST-L + aug | 85K | 26.86% | Schäfer et al. (2024) |
| RawNet2, no aug | 17.6M | 33.94% | Müller et al. (2022) |

Parameter counts for AASIST, AASIST-L, and RawGAT-ST are from Jung et al. (2022). Schäfer et al. numbers are from their preparatory tests (Table 1), trained on ASVspoof 2019 LA with Gaussian noise + MP3 compression augmentation. The WavLM-Large reference is a frozen linear probe over layer-weighted hidden states, not a fine-tuned system — fine-tuned SSL approaches currently reach substantially lower ITW EER (<10%) and are the state of the art; the comparison here is between non-SSL architectures in the 85K-437K parameter class.

## What this is and isn't

This is a **course project** (DATASCI 266, UC Berkeley MIDS) written up with more methodological structure than course projects typically receive. It is not a peer-reviewed publication. Specifically:

- **All results use a single random seed.** Multi-seed replication is needed to make statistical significance claims about the interaction effects.
- **The SNN topology, LIF dynamics, augmentation strategy, and training configuration were chosen from first-principles reasoning**, not empirical search. The 25.65% ITW EER should be interpreted as a lower bound on what this architecture family achieves, not a tuned result.
- **No neuromorphic hardware deployment.** The SNN is simulated on GPU via snnTorch. Energy claims in the paper are theoretical and would require Loihi 2 or Akida deployment to validate.

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

## Precompute features

Feature extraction (especially wavelet scattering) is slow. Precompute once, then train fast:

```bash
# Mel + wavelet features for train/dev
python -m src.data.precompute --config configs/default.yaml \
  --frontends mel wavelet_v3_order1 --splits train dev

# With Kaldi augmentation (5 copies per sample)
python -m src.data.precompute --config configs/default.yaml \
  --frontends wavelet_v3_order1 --splits train dev \
  --augment 5 --augment-mode kaldi \
  --musan-root data/raw/musan --rir-root data/raw/RIRS_NOISES

# Per-channel normalization stats
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

# Evaluate on ASVspoof eval + In-the-Wild
python -m src.evaluation.evaluate_precomputed \
  --checkpoint results/checkpoints/wavelet_v3_order1_snn_kaldi/best.pt \
  --frontend wavelet_v3_order1 \
  --output-name wavelet_order1_snn_kaldi \
  --raw-datasets in_the_wild
```

## SSL baseline

```bash
# Extract frozen WavLM-Large features (~317M params, GPU required)
python -m src.models.ssl_baseline extract --config configs/default.yaml

# Train layer-weighted linear probe
python -m src.models.ssl_baseline train --config configs/default.yaml

# Evaluate
python -m src.models.ssl_baseline eval --config configs/default.yaml
```

## Paper

Full writeup in `paper/draft.md` (source) and `paper/draft.pdf` (rendered).

## Citation

```bibtex
@misc{collins2026wavelet,
  title={Wavelet Scattering and Spiking Neural Networks for Cross-Domain Audio Deepfake Detection},
  author={Collins, Micah},
  year={2026},
  note={DATASCI 266, UC Berkeley MIDS}
}
```

## References

- Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., & Böttinger, K. (2022). Does audio deepfake detection generalize? *Proc. Interspeech*.
- Jung, J., Heo, H., Tak, H., et al. (2022). AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. *Proc. ICASSP*.
- Schäfer, K., Neu, M., & Choi, J.-E. (2024). Robust audio deepfake detection: Exploring front-/back-end combinations and data augmentation strategies for the ASVspoof5 challenge. *Proc. ASVspoof Workshop*.
- Xuan, X., Carbone, D., Pandey, R., Zhang, W., & Kinnunen, T. H. (2026). WST-X series: Wavelet scattering transform for interpretable speech deepfake detection. arXiv:2602.02980.
- Anonymous (2024). SAFE: Spiking neural network-based audio fidelity evaluation. ICLR 2025 submission (withdrawn). OpenReview: QWDZE2mYIe.
