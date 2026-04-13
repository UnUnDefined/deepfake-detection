#!/usr/bin/env python3
"""
Spiking Neural Network classifier for binary deepfake detection.

Key design choices:
  1. Input normalization (LayerNorm) before LIF loop -- scale-invariant
  2. Learnable per-neuron threshold -- adapts to input magnitudes
  3. Soft reset (subtract) -- preserves residual membrane potential
  4. Diverse beta initialization -- log-uniform [0.7, 0.9]
  5. Firing rate regularization -- penalizes dead/saturated neurons

Temporal modes (controlled by temporal_mode parameter):
  - "conv" (default): Conv2d encoder -> freq pool -> interpolate to num_steps.
  - "aligned": Per-frame frequency projection, num_steps = time_frames.
    Each SNN timestep = one audio frame. Preserves wavelet freq structure.

forward() returns (logits, spike_record).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class SpikingClassifierV2(nn.Module):
    """
    SNN classifier for binary deepfake detection.

    LayerNorm on input currents + learnable thresholds ensure neurons fire
    regardless of frontend output scale.
    """

    def __init__(
        self,
        freq_dim: int,
        time_dim: int,
        hidden_dim: int = 256,
        num_steps: int = 50,
        beta_init: float = 0.8,
        threshold_init: float = 1.0,
        num_classes: int = 2,
        surrogate_fn: str = "fast_sigmoid",
        firing_rate_target: float = 0.2,
        firing_rate_lambda: float = 0.01,
        temporal_mode: str = "conv",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.firing_rate_target = firing_rate_target
        self.firing_rate_lambda = firing_rate_lambda
        self.temporal_mode = temporal_mode

        if temporal_mode == "aligned":
            # --- Time-aligned encoder ---
            # Each audio frame's freq vector is projected to hidden_dim.
            # num_steps = time_dim (one SNN step per audio frame).
            # Preserves wavelet band structure — no frequency pooling.
            self.num_steps = time_dim
            self.freq_proj = nn.Sequential(
                nn.Linear(freq_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            # --- Conv encoder (original, works for mel) ---
            self.num_steps = num_steps
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            encoder_channels = 64
            self.channel_proj = nn.Sequential(
                nn.Linear(encoder_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # --- FIX 1: Input normalization before SNN ---
        # This is THE critical fix. Normalizes currents to ~N(0,1) regardless
        # of whether the frontend is mel or wavelet scattering.
        self.input_norm = nn.LayerNorm(hidden_dim)

        # --- SNN layers ---
        if surrogate_fn == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        elif surrogate_fn == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        else:
            raise ValueError(f"Unknown surrogate: {surrogate_fn}")

        # FIX 2: learn_threshold=True — each neuron adapts its firing threshold
        # FIX 3: reset_mechanism="subtract" — soft reset preserves residual V_mem
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.lif1 = snn.Leaky(
            beta=beta_init,
            threshold=threshold_init,
            spike_grad=spike_grad,
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
        )

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lif2 = snn.Leaky(
            beta=beta_init,
            threshold=threshold_init,
            spike_grad=spike_grad,
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
        )

        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta_init,
            threshold=threshold_init,
            spike_grad=spike_grad,
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
            output=True,
        )

        # FIX 4: diverse beta initialization
        self._init_beta_diverse(self.lif1)
        self._init_beta_diverse(self.lif2)
        self._init_beta_diverse(self.lif_out)

    def _init_beta_diverse(
        self, lif_layer: nn.Module, low: float = 0.7, high: float = 0.9
    ):
        """Initialize beta with log-uniform distribution for temporal diversity."""
        beta = getattr(lif_layer, "beta", None)
        if beta is not None and isinstance(beta, nn.Parameter):
            with torch.no_grad():
                n = beta.numel()
                log_low, log_high = math.log(low), math.log(high)
                vals = torch.exp(torch.empty(n).uniform_(log_low, log_high))
                beta.data.copy_(vals.reshape_as(beta.data))

    def _encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (batch, 1, freq, time) features to (batch, num_steps, hidden)
        input currents, with normalization for scale invariance.
        """
        if self.temporal_mode == "aligned":
            # Time-aligned: project each frame's freq vector directly
            # (B, 1, F, T) -> (B, T, F) -> (B, T, hidden)
            h = x.squeeze(1).permute(0, 2, 1)  # (B, T, F)
            h = self.freq_proj(h)               # (B, T, hidden)
        else:
            # Conv encoder: (B, 1, F, T) -> (B, 64, F//4, T//4)
            h = self.encoder(x)

            # Pool over frequency: (B, 64, F//4, T//4) -> (B, 64, T//4)
            h = h.mean(dim=2)

            # Project: (B, T//4, 64) -> (B, T//4, hidden)
            h = h.permute(0, 2, 1)
            h = self.channel_proj(h)

            # Interpolate to SNN timesteps: (B, hidden, T//4) -> (B, hidden, num_steps)
            h = h.permute(0, 2, 1)
            h = F.interpolate(
                h, size=self.num_steps, mode="linear", align_corners=False
            )
            h = h.permute(0, 2, 1)  # (B, num_steps, hidden)

        # Normalize currents — makes SNN agnostic to frontend scale
        h = self.input_norm(h)

        return h

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, freq, time) spectrogram from frontend

        Returns:
            logits: (batch, num_classes)
            spike_record: (batch, num_steps, hidden_dim)
        """
        currents = self._encode_temporal(x)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        mem_out_rec = []

        for t in range(self.num_steps):
            cur = currents[:, t, :]

            # Layer 1
            spk1, mem1 = self.lif1(self.fc1(cur), mem1)
            spk1_drop = self.dropout(spk1)

            # Layer 2
            spk2, mem2 = self.lif2(self.fc2(spk1_drop), mem2)

            # Output (membrane accumulation, no spikes)
            _, mem_out = self.lif_out(self.fc_out(spk2), mem_out)

            spk_rec.append(spk1)
            mem_out_rec.append(mem_out)

        spike_record = torch.stack(spk_rec, dim=1)
        mem_out_all = torch.stack(mem_out_rec, dim=1)

        logits = mem_out_all.sum(dim=1)

        return logits, spike_record

    def firing_rate_loss(self, spike_record: torch.Tensor) -> torch.Tensor:
        """
        FIX 5: Regularization to keep firing rates near target.

        Penalizes per-neuron mean rate deviating from target (default 0.2).
        Added to CE loss in training loop.
        """
        # spike_record: (batch, num_steps, hidden_dim)
        rates = spike_record.mean(dim=1).mean(dim=0)  # per-neuron average
        return self.firing_rate_lambda * ((rates - self.firing_rate_target) ** 2).mean()
