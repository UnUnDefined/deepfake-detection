# Wavelet Scattering and Spiking Neural Networks for Cross-Domain Audio Deepfake Detection: An Exploratory Study

**Author:** Micah Collins
**Course:** DATASCI 266, UC Berkeley MIDS

## Abstract

Audio deepfake detectors trained on laboratory data suffer severe generalization failure on real-world audio. We investigate whether the acoustic frontend, not the classifier, determines cross-domain robustness, using a 2×2 exploratory design crossing frontend (mel spectrogram vs. wavelet scattering) with backend (ResNet-18 vs. spiking neural network, a biologically-inspired architecture that processes information through sparse, event-driven spikes rather than dense activations), trained on ASVspoof 2019 LA and evaluated on the In-the-Wild dataset under the Müller et al. (2022) protocol. We find three interacting effects. First, wavelet scattering regularizes against domain memorization: the mel+SNN generalization gap (ASVspoof eval to ITW EER) is 52pp, while wavelet+SNN achieves 18pp at matched training conditions, a 2.9x reduction consistent with wavelet features transferring across domains rather than encoding dataset-specific artifacts. Second, this regularization is latent: without Kaldi-style augmentation (MUSAN noise, room impulse responses), all SNN frontends converge to ~51% ITW EER (chance). Kaldi augmentation activates the wavelet advantage, improving cross-domain EER by 21-26 percentage points while mel worsens. Third, the wavelet-augmentation interaction is amplified within the SNN architecture: the 2x2 design shows a 24pp wavelet advantage with SNN versus 17pp with ResNet. Our best configuration (first-order wavelet scattering, J=5, Q=10, with a 136K-parameter SNN) achieves 25.65% ITW EER, competitive with augmented non-SSL models such as AASIST (23.55%, 297K params, Schäfer et al. 2024) at fewer parameters, and within 6pp of a frozen WavLM-Large SSL reference (19.61%, 317M parameters) with no pretraining.

## 1. Introduction

Audio deepfake detection systems trained on laboratory datasets achieve near-perfect in-domain accuracy yet degrade dramatically on real-world audio. Müller et al. (2022) quantified this generalization collapse using their In-the-Wild (ITW) dataset, showing EER increases of 200-1000% across all tested architectures. Prior work has addressed this primarily through SSL frontends requiring hundreds of millions of parameters. Meanwhile, spiking neural networks (SNNs) remain entirely unexplored for cross-domain deepfake detection: the only prior work (SAFE, withdrawn from ICLR 2025) never evaluated on real-world data. SNNs are of interest primarily for their energy profile: on neuromorphic hardware (Intel Loihi 2, BrainChip Akida), sparse event-driven computation translates to substantially lower energy per inference than equivalent dense networks on conventional accelerators. This makes SNNs particularly relevant for always-on deepfake detection scenarios such as call screening or on-device authentication. We do not measure energy in this work; our contribution is establishing that cross-domain accuracy is competitive enough to make such deployment worth pursuing.

The intuition for this pairing originated in industrial vibration analysis, where wavelet decompositions are standard for detecting transient events that spectral averaging smears; synthesis artifacts in vocoded speech share this transient character. There is also reason to expect a structural fit between wavelet scattering and SNN computation. Mel spectrograms average signal energy into fixed time-frequency bins, discarding fine temporal structure within each bin. Wavelet scattering preserves this structure through its cascade of wavelet convolutions and modulus operations, producing representations where amplitude modulations at different timescales remain temporally resolved. SNNs, through their leaky integrate-and-fire dynamics and membrane potential evolution, are natively temporal processors: information propagates through spike timing and membrane state, not through static activations. A CNN processes its input as a static 2D array and cannot exploit this temporal alignment. This suggests wavelet scattering should yield a larger improvement over mel when paired with an SNN than with a CNN.

We present the first SNN evaluation on the In-the-Wild benchmark, using a 2x2 design crossing frontend (mel spectrogram vs. wavelet scattering) with backend (ResNet-18 vs. SNN) to test this prediction and explore what drives cross-domain generalization. Our contributions are:

1. The first SNN result on ITW, competitive with augmented non-SSL baselines at 136K parameters.

2. Evidence that wavelet scattering regularizes against domain memorization (2.9x smaller generalization gap than mel at matched conditions), and that Kaldi-style augmentation activates latent cross-domain features in the wavelet representation.

3. An observation that the wavelet-augmentation interaction is amplified within the SNN architecture relative to ResNet.

## 2. Related Work

**Audio deepfake detection and generalization.** The ASVspoof challenge series (Wang et al., 2020; Yamagishi et al., 2021) established standardized benchmarks for spoofing countermeasures. Müller et al. (2022) exposed the generalization gap by evaluating ASVspoof-trained models on their In-the-Wild dataset, finding that mel-spectrogram models achieve 58-78% ITW EER and raw-waveform models 34-53% without augmentation. In preparatory experiments for the ASVspoof5 challenge, Schäfer et al. (2024) showed that augmentation substantially improves these results, with SincNet + RawGAT-ST achieving 18.08% and SincNet + AASIST achieving 23.55% ITW EER using Gaussian noise and MP3 compression augmentation. SSL-based approaches currently achieve the best results but require orders of magnitude more parameters.

**Wavelet scattering for audio.** The scattering transform (Mallat, 2012; Andén & Mallat, 2014) cascades wavelet convolutions with modulus nonlinearities to produce translation-invariant, deformation-stable representations. Xuan et al. (2026) demonstrated that wavelet scattering outperforms mel spectrograms for deepfake detection on the Deepfake-Eval-2024 benchmark, with small averaging scale and high frequency resolution being critical for capturing synthesis artifacts.

**Spiking neural networks for audio.** The only prior SNN work on audio deepfake detection, SAFE (Anonymous, 2024), applied SNNs with MFCC features to ASVspoof 2019 LA and Fake-or-Real datasets, achieving 31.18% cross-dataset EER (CSNN trained on FoR, tested on ASVspoof), a lab-to-lab transfer task considerably easier than the lab-to-wild generalization that ITW requires. SAFE was withdrawn from ICLR 2025 review and never evaluated on In-the-Wild. No other SNN-based audio deepfake detector exists in the literature.

**Data augmentation for anti-spoofing.** The Kaldi x-vector recipe (Snyder et al., 2018) applies MUSAN noise and room impulse response convolution. Schäfer et al. (2024) showed that even simple augmentation (Gaussian noise, MP3 compression) substantially improves cross-domain robustness for non-SSL models.

## 3. Experimental Setup

### 3.1 Datasets

**Training:** ASVspoof 2019 Logical Access (LA) train split (25,380 utterances: 2,580 bonafide, 22,800 spoofed from 6 TTS/VC systems). All audio at 16 kHz, truncated/padded to 4 seconds (64,000 samples).

**Validation:** ASVspoof 2019 LA dev split (24,844 utterances). Used for model selection and early stopping.

**Evaluation:** ASVspoof 2019 LA eval split (71,933 utterances, 13 attack systems including 7 unseen). In-the-Wild dataset (31,779 utterances: 19,963 bonafide, 11,816 deepfake from 58 public figures). We follow the Müller et al. (2022) protocol: train on ASVspoof, select checkpoint by ASVspoof dev EER, evaluate on ITW without any ITW-based selection.

### 3.2 Acoustic Frontends

**Mel spectrogram (baseline):** 80 mel bands, 1024-point FFT, 256-sample hop, log compression. Output: (1, 80, 250).

**Wavelet scattering (proposed):** Kymatio (Andreux et al., 2020) Scattering1D with J=5 (averaging scale), Q=10 (wavelets per octave). No temporal averaging (T=0), per-coefficient time alignment via interpolation, log compression, per-channel normalization from training statistics. We evaluate two order configurations: **order-2** (max_order=2, 100 coefficients) includes second-order scattering paths that capture amplitude modulation dynamics; **order-1** (max_order=1, ~55 coefficients) uses only first-order paths. Configuration informed by Xuan et al. (2026), who showed that small J preserves transient synthesis artifacts and high Q provides fine spectral resolution for vocoder artifact detection.

### 3.3 SNN Architecture

Both frontends feed a shared SNN architecture (136K parameters) consisting of a Conv2d encoder with frequency-axis mean pooling, three Leaky Integrate-and-Fire (LIF) layers implemented via snnTorch (Eshraghian et al., 2023) with learnable thresholds and membrane dynamics, and a membrane potential readout. Full architectural details are in Appendix A.

### 3.4 Data Augmentation

Kaldi-style augmentation produces five augmented copies per training sample (152,280 total) using MUSAN noise, music, and babble at calibrated SNRs, plus room impulse response convolution. Augmentation is applied to raw waveforms before feature extraction. Full details in Appendix B.

### 3.5 Training

SNN models use Adam (lr=1e-4) with cosine annealing for up to 50 epochs; ResNet-18 uses lr=4e-3 for up to 30 epochs. Both use early stopping on dev EER. All features precomputed to disk. All experiments conducted on a single NVIDIA RTX 4090; results should be interpreted as a case study rather than a comprehensive benchmark. Full hyperparameters in Appendix C.

### 3.6 SSL Reference Baseline

We include a frozen WavLM-Large baseline (317M parameters) with a layer-weighted linear probe (~25K trainable parameters) as an SSL reference. Details in Appendix D.

## 4. Results

All results follow the Müller et al. (2022) protocol: best ASVspoof dev checkpoint, single-shot ITW evaluation.

### 4.1 Comparison with Published Results

**Table 1: ITW EER comparison (all trained on ASVspoof 2019 LA)**

| Model | Params | ITW EER | Source |
|-------|--------|---------|--------|
| RawGAT-ST + aug | 437K | 18.08% | Schäfer et al. |
| WavLM-Large + probe | 317M + 25K | 19.61% | this work |
| AASIST + aug | 297K | 23.55% | Schäfer et al. |
| **Wavelet + SNN + Kaldi** | **136K** | **25.65%** | **this work** |
| AASIST-L + aug | 85K | 26.86% | Schäfer et al. |
| RawNet2, no aug | 17.6M | 33.94% | Müller et al. |

This is the first SNN evaluation on In-the-Wild. Our 25.65% is competitive with AASIST (297K params; Jung et al., 2022) and AASIST-L (85K; Jung et al., 2022), whose ITW EER numbers are from Schäfer et al. (2024). Our model occupies a middle point in both EER and parameter count (136K), and substantially improves on the only prior SNN result (SAFE CSNN: 31.18% EER trained on Fake-or-Real and tested on ASVspoof 2019 LA, a lab-to-lab evaluation considered less challenging than ITW, which additionally introduces real-world recording conditions, unknown post-processing, and out-of-distribution synthesis methods).

### 4.2 Experimental Results

**Table 2: Full experimental results, ITW EER (%)**

| Frontend | Backend | Augmentation | ITW EER |
|----------|---------|-------------|---------|
| mel | ResNet-18 (11.2M) | Kaldi | 47.52% |
| mel | SNN (136K) | Kaldi | 55.13% |
| mel | SNN (136K) | none | 50.63% |
| wavelet order-2 | ResNet-18 (11.2M) | Kaldi | 30.72% |
| wavelet order-2 | SNN (136K) | Kaldi | 31.05% |
| wavelet order-2 | SNN (136K) | none | 52.55% |
| wavelet order-1 | ResNet-18 (11.2M) | Kaldi | 34.55% |
| **wavelet order-1** | **SNN (136K)** | **Kaldi** | **25.65%** |
| wavelet order-1 | SNN (136K) | none | 51.78% |

Three effects are visible in Table 2:

**Frontend effect.** With Kaldi augmentation, switching from mel to wavelet order-2 reduces ITW EER by 17pp with ResNet (47.52% to 30.72%) and 24pp with SNN (55.13% to 31.05%). With the right frontend, a 136K-parameter SNN matches an 11.2M-parameter ResNet.

**Augmentation as activation.** Without augmentation, all SNN frontends converge to ~51% ITW EER (chance). Kaldi augmentation activates the wavelet advantage (21-26pp improvement) while mel worsens (50.63% to 55.13%). This is consistent with augmentation functioning as a selection mechanism: Kaldi perturbations corrupt dataset-specific artifacts, forcing the model toward discriminative features that exist in the wavelet representation (deformation-stable coefficients) but may not survive mel's spectral averaging. In preliminary runs, ResNet without augmentation also collapsed to chance ITW performance with both frontends; the no-augmentation analysis is reported only for the SNN backend due to time constraints.

**Backend-dependent order interaction.** Order-1 helps SNN (31.05% to 25.65%) but hurts ResNet (30.72% to 34.55%), attributable to the SNN's frequency-axis mean pooling discarding cross-frequency information that second-order coefficients capture. The J=5, Q=10 configuration (informed by Xuan et al., 2026) was chosen to preserve transient synthesis artifacts (small J) with fine spectral resolution (high Q).

### 4.3 Generalization Gap

We define the generalization gap as the difference between ASVspoof eval EER and ITW EER for the best-dev checkpoint. For mel + SNN + Kaldi, this gap is 52.24 percentage points (eval 2.96%, ITW 55.13%). For wavelet order-1 + SNN + Kaldi, the gap is 18.01 percentage points (eval 7.64%, ITW 25.65%). The wavelet model achieves a 2.9x smaller gap despite using the same backend, augmentation, and training regime, consistent with wavelet scattering encoding features that transfer across domains rather than dataset-specific artifacts. Notably, mel's lower in-domain EER (2.96% vs 7.64%) is a symptom of overfitting rather than superior representation quality.

## 5. Discussion

This work presents the first SNN evaluation on the In-the-Wild benchmark. For context, ITW is considered a difficult cross-domain benchmark: Müller et al. (2022) found most architectures in the 33-90% EER range, and the published baselines in Table 1 reflect multi-year iteration by their respective research groups. The SNN configuration reported here used no systematic hyperparameter search beyond the wavelet scattering axis, making 25.65% a first-pass result on an unoptimized architecture family. The 2x2 design (Table 2) suggests three interacting mechanisms: wavelet scattering regularizes against domain memorization (Section 4.3), Kaldi augmentation activates latent cross-domain features in the wavelet representation, and the SNN architecture amplifies this interaction relative to ResNet. The convergence of all SNN frontends to ~51% ITW EER without augmentation is consistent with the hypothesis that cross-domain information is present but latent in the wavelet representation, and that augmentation selects for it by corrupting dataset-specific artifacts.

**Limitations and future work.** This work establishes an initial SNN baseline on ITW. While we explored scattering order (first vs. second) and wavelet hyperparameters (J, Q), the SNN topology, LIF dynamics, augmentation strategy, and training configuration were chosen based on first-principles reasoning rather than empirical search. Each represents an axis along which the architecture has not been optimized. The 25.65% EER should be interpreted as a lower bound on what this architecture family can achieve. In particular, the SNN's frequency pooling destroys the multi-resolution band structure that wavelet scattering preserves (as demonstrated by the backend-dependent order effect), and an architecture that maintains frequency structure could potentially leverage order-2 features. Our comparison to Schäfer et al. (2024) is approximate: they used different augmentation (Gaussian noise + MP3 compression vs. our MUSAN + RIR) and a learned frontend (SincNet) rather than a fixed transform. All results use a single random seed per configuration; the observed differences between cells (e.g., the 24pp frontend effect with SNN, the 2.9x generalization gap ratio) have not been tested for statistical significance and could be influenced by seed variance. Multi-seed replication with significance testing is needed to confirm the interaction effects we report. All experiments use snnTorch on GPU, which simulates spiking dynamics through temporal unrolling; the energy advantages discussed in Section 1 require deployment on neuromorphic hardware, which we leave to future work.

## 6. Conclusion

We present the first SNN evaluation on the In-the-Wild audio deepfake benchmark, achieving 25.65% ITW EER with a 136K-parameter wavelet scattering + SNN model. This is competitive with augmented non-SSL models (AASIST: 23.55% at 297K params, Schäfer et al. 2024), and substantially improves on the only prior SNN result in audio deepfake detection (SAFE: 31.18% cross-dataset). A 2x2 exploratory experiment suggests three interacting mechanisms: wavelet scattering regularizes against domain memorization, Kaldi augmentation activates latent cross-domain features in the wavelet representation, and the SNN architecture amplifies this interaction. Without augmentation all SNN frontends converge to chance (~51% ITW EER); with it, wavelet improves by 21-26pp while mel worsens. Our results suggest that for audio deepfake detection, the interaction between frontend, augmentation, and architecture may matter more than any single component, and that SNNs are a viable architecture for this task when paired with appropriate representations.

## References

Andén, J., & Mallat, S. (2014). Deep scattering spectrum. *IEEE Transactions on Signal Processing*, 62(16), 4114-4128.

Andreux, M., Angles, T., Exarchakis, G., Leonarduzzi, R., Rochette, G., Thiry, L., Zarka, J., Mallat, S., Andén, J., Belilovsky, E., Bruna, J., Lostanlen, V., Chaudhary, M., Hirn, M. J., Oyallon, E., Zhang, S., Cella, C., & Eickenberg, M. (2020). Kymatio: Scattering transforms in Python. *Journal of Machine Learning Research*, 21(60), 1-6.

Anonymous. (2024). SAFE: Spiking neural network-based audio fidelity evaluation. *ICLR 2025 submission* (withdrawn Nov 2024). OpenReview ID: QWDZE2mYIe.

Eshraghian, J. K., Ward, M., Neftci, E. O., Wang, X., Lenz, G., Dwivedi, G., Bennamoun, M., Jeong, D. S., & Lu, W. D. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016-1054.

Jung, J.-W., Heo, H.-S., Tak, H., Shim, H.-J., Chung, J. S., Lee, B.-J., Yu, H.-J., & Evans, N. (2022). AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. *Proc. ICASSP*, 6367-6371.

Mallat, S. (2012). Group invariant scattering. *Communications on Pure and Applied Mathematics*, 65(10), 1331-1398.

Müller, N. M., Czempin, P., Dieckmann, F., Froghyar, A., & Böttinger, K. (2022). Does audio deepfake detection generalize? *Proc. Interspeech*.

Schäfer, K., Neu, M., & Choi, J.-E. (2024). Robust audio deepfake detection: Exploring front-/back-end combinations and data augmentation strategies for the ASVspoof5 challenge. *Proc. ASVspoof Workshop*.

Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-vectors: Robust DNN embeddings for speaker recognition. *Proc. ICASSP*, 5329-5333.


Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N., & Larcher, A. (2021). End-to-end anti-spoofing with RawNet2. *Proc. ICASSP*, 6369-6373.


Wang, X., Yamagishi, J., Todisco, M., et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language*, 64, 101114.

Xuan, X., Carbone, D., Pandey, R., Zhang, W., & Kinnunen, T. H. (2026). WST-X series: Wavelet scattering transform for interpretable speech deepfake detection. *arXiv:2602.02980*.

Yamagishi, J., Wang, X., Todisco, M., et al. (2021). ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection. *Proc. ASVspoof Workshop*.

## Appendix A: SNN Architecture Details

The SNN architecture (136K parameters) consists of:

- **Encoder:** Two Conv2d layers (1 to 32 to 64 channels) with BatchNorm, ReLU, and MaxPool2d, followed by frequency-axis mean pooling and linear projection to hidden dimension (256).
- **SNN layers:** Three Leaky Integrate-and-Fire (LIF) layers implemented via snnTorch. Per-neuron learnable thresholds (initialized at 0.3), learnable membrane decay (beta initialized log-uniform in [0.7, 0.9]), soft reset mechanism. 50 SNN timesteps via temporal interpolation.
- **Readout:** Sum of output membrane potentials across timesteps.
- **Regularization:** Firing rate regularization targeting 0.2 spikes/timestep (lambda=0.05), dropout (0.2).

The Conv2d encoder performs frequency-axis mean pooling early in processing, which has implications for scattering order selection (see Section 4.2). This pooling destroys the multi-resolution band structure that wavelet scattering preserves, explaining why second-order coefficients help ResNet (which preserves full 2D structure) but not the SNN.

## Appendix B: Data Augmentation Details

**Kaldi-style augmentation** produces five augmented copies per training sample (152,280 total from 25,380 originals). Each copy receives one of:

| Augmentation | Parameters |
|-------------|------------|
| MUSAN noise | SNR 0-15 dB |
| MUSAN music | SNR 5-15 dB |
| MUSAN babble | 3-7 speakers, SNR 13-20 dB |
| Room impulse response | Convolution with real RIRs |
| RIR + noise combination | RIR convolution followed by noise |

Augmentation is applied to raw waveforms before feature extraction. The MUSAN corpus (2,016 clips) and RIRS_NOISES (500 impulse responses) are preloaded into memory for efficiency.

## Appendix C: Training Hyperparameters

| Parameter | SNN | ResNet-18 |
|-----------|-----|-----------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 4e-3 |
| Warmup | 2 epochs linear | none |
| Schedule | cosine annealing | cosine annealing |
| Max epochs | 50 | 30 |
| Batch size | 256 | 256 |
| Early stopping patience | 15 epochs | 5 epochs |
| Gradient clipping | 1.0 | 1.0 |
| Loss | CE + firing rate reg | CE |
| Hardware | NVIDIA RTX 4090 | NVIDIA RTX 4090 |

All features precomputed to disk before training. Model selection by best ASVspoof dev EER.

## Appendix D: SSL Reference Baseline

We include a frozen WavLM-Large baseline (317M parameters) as a reference for the current SSL frontier. Following standard SSL probing protocol, we extract all 25 hidden states from WavLM-Large for each utterance, mean-pool over time, and train a layer-weighted linear probe (learned softmax weights over layers + linear classifier, ~25K trainable parameters). No augmentation is applied, as SSL pretraining on 94,000 hours of diverse speech provides implicit channel robustness. The probe is trained with Adam (lr=1e-3) for up to 100 epochs with early stopping on dev EER.
