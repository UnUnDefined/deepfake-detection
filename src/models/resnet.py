#!/usr/bin/env python3
"""
ResNet-18 classifier adapted for single-channel spectrograms.

Input: (batch, 1, freq, time) from either mel or wavelet frontend.
Output: (batch, num_classes) logits.

Uses torchvision's ResNet-18 with modified first conv (1 input channel)
and final FC layer (2 output classes). AdaptiveAvgPool in the original
architecture handles variable (freq, time) dimensions automatically.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 for binary deepfake detection on single-channel spectrograms.

    Works identically for mel and wavelet scattering inputs since both
    produce (batch, 1, freq, time) tensors. The ResNet treats the
    spectrogram as a single-channel image.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Replace first conv: 3 channels → 1 channel
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC: 512 → num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, freq, time) spectrogram from frontend

        Returns:
            logits: (batch, num_classes)
        """
        return self.resnet(x)
