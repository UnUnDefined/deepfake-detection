#!/usr/bin/env python3
"""
SSL Baseline: Frozen WavLM-Large + learned layer-weighted linear probe.

Standard SSL probing protocol (Pascu et al., 2024):
  1. Extract all 25 hidden states from WavLM-Large (frozen)
  2. Learn per-layer weights via softmax
  3. Weighted sum → linear classifier → binary prediction

No augmentation — SSL pretraining provides robustness.

Usage:
  # Step 1: Extract and cache features
  python -m src.models.ssl_baseline extract --config configs/default.yaml

  # Step 2: Train linear probe
  python -m src.models.ssl_baseline train --config configs/default.yaml

  # Step 3: Evaluate
  python -m src.models.ssl_baseline eval --config configs/default.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import yaml

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(config: dict, splits: list[str], output_dir: Path, device: torch.device):
    """Extract all 25 WavLM-Large hidden states, mean-pooled over time."""
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    print("Loading WavLM-Large...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    model.eval()
    print(f"WavLM-Large loaded: {sum(p.numel() for p in model.parameters()):,} params (frozen)")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]

    from src.data.datasets import ASVspoof2019LA, InTheWild

    datasets_map = {
        "train": lambda: ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="train",
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
        "dev": lambda: ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="dev",
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
        "eval": lambda: ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="eval",
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
        "itw": lambda: InTheWild(
            root=data_cfg["inthewild_root"],
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
    }

    for split_name in splits:
        split_path = output_dir / f"{split_name}.pt"
        if split_path.exists():
            print(f"  [SKIP] {split_name} already cached at {split_path}")
            continue

        print(f"\n  Extracting {split_name}...")
        dataset = datasets_map[split_name]()
        n = len(dataset)
        print(f"  Samples: {n}")

        all_features = []  # list of (25, 1024) tensors
        all_labels = []
        batch_size = 8  # WavLM is large, keep batch small

        with torch.no_grad():
            for i in tqdm(range(0, n, batch_size), desc=f"  {split_name}"):
                batch_wavs = []
                batch_labels = []
                for j in range(i, min(i + batch_size, n)):
                    waveform, label, _meta = dataset[j]
                    # waveform is (1, T) from dataset, squeeze to (T,)
                    if waveform.ndim == 2:
                        waveform = waveform.squeeze(0)
                    batch_wavs.append(waveform.numpy())
                    batch_labels.append(label)

                # Process batch
                inputs = processor(
                    batch_wavs, sampling_rate=16000, return_tensors="pt",
                    padding=True, truncation=True, max_length=64000,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True)
                # outputs.hidden_states is tuple of 25 tensors, each (batch, time, 1024)
                hidden_states = torch.stack(outputs.hidden_states, dim=1)  # (batch, 25, time, 1024)

                # Mean pool over time
                # Handle padding mask if present
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"]
                    # Downsample mask to match hidden state time dim
                    time_dim = hidden_states.shape[2]
                    if mask.shape[1] != time_dim:
                        mask = F.interpolate(
                            mask.unsqueeze(1).float(), size=time_dim, mode="nearest"
                        ).squeeze(1)
                    mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, time, 1)
                    pooled = (hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
                else:
                    pooled = hidden_states.mean(dim=2)  # (batch, 25, 1024)

                all_features.append(pooled.cpu())
                all_labels.extend(batch_labels)

        features = torch.cat(all_features, dim=0)  # (N, 25, 1024)
        labels = torch.tensor(all_labels, dtype=torch.long)

        print(f"  Shape: {features.shape}, Labels: {labels.shape}")
        print(f"  Saving to {split_path}...")
        torch.save({"features": features.half(), "labels": labels}, split_path)
        print(f"  Saved ({split_path.stat().st_size / 1e6:.0f} MB)")

    print("\nExtraction complete.")


# ---------------------------------------------------------------------------
# Linear probe model
# ---------------------------------------------------------------------------

class LayerWeightedProbe(nn.Module):
    """Learned layer weighting + linear classifier."""

    def __init__(self, n_layers: int = 25, hidden_dim: int = 1024, num_classes: int = 2):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, n_layers, hidden_dim) — cached WavLM features
        Returns:
            logits: (batch, num_classes)
        """
        weights = F.softmax(self.layer_weights, dim=0)  # (n_layers,)
        weighted = (hidden_states * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return self.classifier(weighted)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(config: dict, cache_dir: Path, device: torch.device):
    """Train the linear probe on cached features."""
    from src.evaluation.metrics import compute_eer

    print("Loading cached features...")
    train_data = torch.load(cache_dir / "train.pt", weights_only=True)
    dev_data = torch.load(cache_dir / "dev.pt", weights_only=True)

    train_features = train_data["features"].float().to(device)
    train_labels = train_data["labels"].to(device)
    dev_features = dev_data["features"].float().to(device)
    dev_labels = dev_data["labels"].to(device)

    print(f"Train: {train_features.shape[0]} samples")
    print(f"Dev:   {dev_features.shape[0]} samples")

    model = LayerWeightedProbe(
        n_layers=train_features.shape[1],
        hidden_dim=train_features.shape[2],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Probe parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_eer = 1.0
    patience = 10
    no_improve = 0
    best_state = None

    # Mini-batch training
    batch_size = 512
    n_train = train_features.shape[0]

    for epoch in range(100):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(train_features[idx])
            loss = criterion(logits, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_features)
            dev_scores = F.log_softmax(dev_logits, dim=1)[:, 1].cpu().numpy()
            dev_labs = dev_labels.cpu().numpy()
            valid = dev_labs >= 0
            eer, threshold = compute_eer(dev_labs[valid], dev_scores[valid])
            eer_pct = eer * 100

        # Layer weights
        weights = F.softmax(model.layer_weights, dim=0)
        top_layers = torch.topk(weights, 3)

        if eer < best_eer:
            best_eer = eer
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  Epoch {epoch+1:>3}: loss={epoch_loss/n_batches:.4f} "
                  f"dev_EER={eer_pct:.2f}%{marker} "
                  f"top_layers=[{top_layers.indices[0].item()},{top_layers.indices[1].item()},{top_layers.indices[2].item()}]")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Save best
    ckpt_path = cache_dir / "probe_best.pt"
    torch.save({
        "model_state": best_state,
        "best_eer": best_eer,
        "n_layers": train_features.shape[1],
        "hidden_dim": train_features.shape[2],
    }, ckpt_path)
    print(f"\nBest dev EER: {best_eer*100:.2f}%")
    print(f"Saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_probe(config: dict, cache_dir: Path, device: torch.device):
    """Evaluate trained probe on eval and ITW."""
    from src.evaluation.metrics import compute_all_metrics

    ckpt = torch.load(cache_dir / "probe_best.pt", weights_only=True)
    model = LayerWeightedProbe(
        n_layers=ckpt["n_layers"], hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded probe (dev EER: {ckpt['best_eer']*100:.2f}%)")

    # Layer weights analysis
    weights = F.softmax(model.layer_weights, dim=0)
    top5 = torch.topk(weights, 5)
    print(f"Top 5 layer weights: {[(i.item(), f'{w:.3f}') for i, w in zip(top5.indices, top5.values)]}")

    results = {}
    for split_name in ["eval", "itw"]:
        data_path = cache_dir / f"{split_name}.pt"
        if not data_path.exists():
            print(f"  [SKIP] {split_name}: not cached")
            continue

        data = torch.load(data_path, weights_only=True)
        features = data["features"].float().to(device)
        labels = data["labels"].cpu().numpy()

        with torch.no_grad():
            logits = model(features)
            scores = F.log_softmax(logits, dim=1)[:, 1].cpu().numpy()

        valid = labels >= 0
        if valid.sum() == 0:
            print(f"  [SKIP] {split_name}: no valid labels")
            continue

        metrics = compute_all_metrics(labels[valid], scores[valid])
        results[split_name] = metrics
        ds_label = "ASVspoof eval" if split_name == "eval" else "In-the-Wild"
        print(f"  {ds_label:>15}: EER={metrics['eer_pct']:.2f}%  AUC={metrics['auc']:.4f}")

    # Save
    out_path = cache_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSL Baseline: WavLM-Large + Linear Probe")
    parser.add_argument("command", choices=["extract", "train", "eval", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-dir", default="data/processed/ssl_wavlm")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    print(f"Device: {device}")
    print(f"Cache dir: {cache_dir}")

    if args.command in ("extract", "all"):
        extract_features(config, ["train", "dev", "eval", "itw"], cache_dir, device)

    if args.command in ("train", "all"):
        train_probe(config, cache_dir, device)

    if args.command in ("eval", "all"):
        evaluate_probe(config, cache_dir, device)


if __name__ == "__main__":
    main()
