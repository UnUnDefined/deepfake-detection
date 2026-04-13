#!/usr/bin/env python3
"""
Unified training loop for the 2x2 factorial experiment.

Usage:
  python -m src.training.train --frontend mel --model resnet
  python -m src.training.train --frontend wavelet --model resnet
  python -m src.training.train --frontend mel --model snn
  python -m src.training.train --frontend wavelet --model snn

All four combos use the same training logic. SNN models additionally
log spike sparsity per epoch. Checkpoints saved to results/checkpoints/.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm
import yaml

# Project imports
from src.data.datasets import ASVspoof2019LA, get_dataloader
from src.data.precomputed_dataset import PrecomputedDataset, get_precomputed_dataloader, SpecAugment
from src.data.features import get_frontend
from src.models.resnet import ResNet18Classifier
from src.models.snn import SpikingClassifierV2
from src.evaluation.metrics import compute_eer, compute_spike_sparsity


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    model_name: str,
    model_cfg: dict,
    freq_dim: int,
    time_dim: int,
) -> nn.Module:
    if model_name == "resnet":
        return ResNet18Classifier(
            num_classes=model_cfg["resnet"]["num_classes"],
            pretrained=model_cfg["resnet"].get("pretrained", False),
        )
    elif model_name == "snn":
        cfg = model_cfg.get("snn_v2", model_cfg.get("snn", {}))
        return SpikingClassifierV2(
            freq_dim=freq_dim,
            time_dim=time_dim,
            num_steps=cfg["num_steps"],
            beta_init=cfg["beta_init"],
            threshold_init=cfg["threshold_init"],
            num_classes=cfg["num_classes"],
            firing_rate_target=cfg.get("firing_rate_target", 0.2),
            firing_rate_lambda=cfg.get("firing_rate_lambda", 0.01),
            temporal_mode=cfg.get("temporal_mode", "conv"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    frontend,  # nn.Module or None (precomputed)
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_snn: bool,
    grad_clip: float = 0.0,
    augment: nn.Module = None,
) -> dict:
    if frontend is not None:
        frontend.train()
    model.train()

    total_loss = 0.0
    all_spikes = []
    n_batches = 0

    for batch_data, labels, _meta in tqdm(loader, desc="  train", leave=False):
        batch_data = batch_data.to(device)
        labels = labels.to(device)

        # Frontend: (B, T) → (B, 1, freq, time), or already features if precomputed
        features = frontend(batch_data) if frontend is not None else batch_data

        # SpecAugment (training only)
        if augment is not None:
            features = augment(features)

        # Model forward
        if is_snn:
            logits, spk_rec = model(features)
            all_spikes.append(spk_rec.detach().cpu().numpy())
        else:
            logits = model(features)

        loss = criterion(logits, labels)

        # Firing rate regularization (SNN only)
        if is_snn and hasattr(model, "firing_rate_loss"):
            loss = loss + model.firing_rate_loss(spk_rec)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (mandatory for SNN stability)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    result = {"loss": total_loss / max(n_batches, 1)}

    if is_snn and all_spikes:
        spikes = np.concatenate(all_spikes, axis=0)
        result["spike_sparsity"] = compute_spike_sparsity(spikes)

    return result


@torch.no_grad()
def validate(
    frontend,  # nn.Module or None (precomputed)
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_snn: bool,
) -> dict:
    if frontend is not None:
        frontend.eval()
    model.eval()

    total_loss = 0.0
    all_scores = []
    all_labels = []
    all_spikes = []
    n_batches = 0

    for batch_data, labels, _meta in tqdm(loader, desc="  val", leave=False):
        batch_data = batch_data.to(device)
        labels = labels.to(device)

        features = frontend(batch_data) if frontend is not None else batch_data

        if is_snn:
            logits, spk_rec = model(features)
            all_spikes.append(spk_rec.cpu().numpy())
        else:
            logits = model(features)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        # Use log_softmax for numerically stable scoring with large logits
        log_probs = torch.log_softmax(logits.float(), dim=1)
        all_scores.append(log_probs[:, 1].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    eer, eer_thresh = compute_eer(labels, scores)

    result = {
        "loss": total_loss / max(n_batches, 1),
        "eer": eer,
        "eer_pct": eer * 100,
        "threshold": eer_thresh,
    }

    if is_snn and all_spikes:
        spikes = np.concatenate(all_spikes, axis=0)
        result["spike_sparsity"] = compute_spike_sparsity(spikes)

    return result


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument(
        "--frontend", choices=["mel", "wavelet", "wavelet_v2", "wavelet_v3", "wavelet_v3_order1"], required=True,
        help="Feature frontend",
    )
    parser.add_argument(
        "--model", choices=["resnet", "snn"], required=True,
        help="Classifier backend",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Use small data subset for pipeline testing",
    )
    parser.add_argument(
        "--precomputed", action="store_true",
        help="Use precomputed features from data/processed/ (much faster)",
    )
    parser.add_argument(
        "--cross-eval", action="store_true",
        help="Run cross-domain mini-eval on In-the-Wild probe set when val EER improves",
    )
    parser.add_argument(
        "--augmented", action="store_true",
        help="Use augmented training data from data/processed/asvspoof_train_aug/",
    )
    parser.add_argument(
        "--augment-mode", choices=["simple", "kaldi"], default="simple",
        help="Which augmented dataset to use: 'simple' (Gaussian) or 'kaldi' (MUSAN/RIR/codec)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Build experiment name with augmentation suffix to prevent checkpoint overwrites
    aug_suffix = ""
    if args.augmented and hasattr(args, 'augment_mode') and args.augment_mode:
        aug_suffix = f"_{args.augment_mode}"
    elif args.augmented:
        aug_suffix = "_aug"
    else:
        aug_suffix = "_noaug"
    experiment_name = f"{args.frontend}_{args.model}{aug_suffix}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Frontend: {args.frontend} | Model: {args.model}")
    print(f"{'='*60}\n")

    # Setup — apply per-model training overrides if defined
    train_cfg = dict(config["training"])  # copy so we can mutate
    overrides = train_cfg.pop("overrides", {})
    if args.model in overrides:
        print(f"Applying training overrides for {args.model}: {overrides[args.model]}")
        train_cfg.update(overrides[args.model])
    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data_cfg = config["data"]
    use_precomputed = args.precomputed and not args.debug

    if use_precomputed:
        processed_root = Path(data_cfg["processed_root"])
        if args.augmented:
            train_dir = "asvspoof_train_kaldi" if args.augment_mode == "kaldi" else "asvspoof_train_aug"
        else:
            train_dir = "asvspoof_train"
        train_dataset = PrecomputedDataset(
            root=str(processed_root / train_dir), frontend=args.frontend,
        )
        dev_dataset = PrecomputedDataset(
            root=str(processed_root / "asvspoof_dev"), frontend=args.frontend,
        )
        train_loader = get_precomputed_dataloader(
            train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        )
        dev_loader = get_precomputed_dataloader(
            dev_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        )
        # Get dims from first sample
        sample_feat, _, _ = train_dataset[0]
        freq_dim, time_dim = sample_feat.shape[1], sample_feat.shape[2]
        frontend = None  # not needed
        print(f"[PRECOMPUTED] Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")
        print(f"Feature shape: (1, {freq_dim}, {time_dim})")
    else:
        train_dataset = ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="train",
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        dev_dataset = ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="dev",
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )

        if args.debug:
            from torch.utils.data import Subset
            n_train = min(500, len(train_dataset))
            n_dev = min(200, len(dev_dataset))
            train_indices = list(range(0, len(train_dataset), len(train_dataset) // n_train))[:n_train]
            dev_indices = list(range(0, len(dev_dataset), len(dev_dataset) // n_dev))[:n_dev]
            train_dataset = Subset(train_dataset, train_indices)
            dev_dataset = Subset(dev_dataset, dev_indices)

        train_loader = get_dataloader(
            train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        )
        dev_loader = get_dataloader(
            dev_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        )

        # Frontend
        frontend = get_frontend(args.frontend, config["features"]).to(device)
        with torch.no_grad():
            dummy = torch.randn(1, data_cfg["max_samples"]).to(device)
            out = frontend(dummy)
            freq_dim, time_dim = out.shape[2], out.shape[3]
        print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")
        print(f"Frontend output: (1, {freq_dim}, {time_dim})")

    # Model
    is_snn = args.model == "snn"
    model = build_model(
        args.model, config["model"], freq_dim, time_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Channel normalization stats (for precomputed wavelet frontends)
    channel_stats = None
    if use_precomputed and args.frontend in ("wavelet_v2", "wavelet_v3", "wavelet_v3_order1"):
        stats_dir = train_dir  # use same dir as training data
        stats_path = Path(data_cfg["processed_root"]) / stats_dir / args.frontend / "channel_stats.pt"
        if stats_path.exists():
            channel_stats = torch.load(stats_path, weights_only=True)
            print(f"[CHANNEL NORM] Loaded stats from {stats_path}")
            print(f"  mean range: [{channel_stats['mean'].min():.3f}, {channel_stats['mean'].max():.3f}]")
            print(f"  std range:  [{channel_stats['std'].min():.3f}, {channel_stats['std'].max():.3f}]")
        else:
            print(f"[CHANNEL NORM] No stats at {stats_path} — run precompute --compute-stats first")

    # Pass channel stats to model if it supports it
    if channel_stats is not None and hasattr(model, "set_channel_stats"):
        model.set_channel_stats(channel_stats["mean"].to(device), channel_stats["std"].to(device))

    # Optimizer (include frontend params in case any are learnable)
    all_params = list(model.parameters())
    if frontend is not None:
        all_params = list(frontend.parameters()) + all_params
    optimizer = Adam(
        all_params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler: optional linear warmup + cosine decay
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    total_epochs = train_cfg["epochs"]
    if warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(f"Scheduler: {warmup_epochs}-epoch linear warmup + cosine decay")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Label smoothing: prevents overconfident predictions, reduces domain memorization
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing}")

    # SpecAugment: mask random time/freq bands during training
    augment = None
    if train_cfg.get("spec_augment", False):
        augment = SpecAugment(
            freq_masks=2, freq_width=4, time_masks=2, time_width=20,
        ).to(device)
        print("SpecAugment enabled (2 freq masks w=4, 2 time masks w=20)")

    # Checkpoint directory
    ckpt_dir = Path("results") / "checkpoints" / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Cross-domain probe set (loaded once, evaluated when val EER improves)
    cross_eval_loader = None
    if args.cross_eval:
        probe_root = Path(data_cfg["processed_root"]) / "inthewild_probe"
        probe_feat_dir = probe_root / args.frontend
        if probe_feat_dir.exists():
            cross_eval_ds = PrecomputedDataset(str(probe_root), args.frontend)
            cross_eval_loader = get_precomputed_dataloader(
                cross_eval_ds, batch_size=train_cfg["batch_size"], shuffle=False,
            )
            print(f"[CROSS-EVAL] Loaded In-the-Wild probe set: {len(cross_eval_ds)} samples")
        else:
            print(f"[CROSS-EVAL] No probe set at {probe_feat_dir}")
            print(f"  Run: python -m src.data.precompute --probe-inthewild 2000 --frontends {args.frontend}")

    # Training loop
    best_eer = float("inf")
    patience_counter = 0
    history = []
    grad_clip = 1.0 if is_snn else 0.0

    print(f"\nTraining: lr={train_cfg['lr']}, epochs={train_cfg['epochs']}, "
          f"patience={train_cfg['early_stopping_patience']}, "
          f"warmup={warmup_epochs}, wd={train_cfg['weight_decay']}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_result = train_epoch(
            frontend, model, train_loader, criterion, optimizer, device, is_snn,
            grad_clip=grad_clip, augment=augment,
        )
        val_result = validate(
            frontend, model, dev_loader, criterion, device, is_snn
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        log_line = (
            f"Epoch {epoch:02d}/{train_cfg['epochs']} | "
            f"train_loss={train_result['loss']:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_EER={val_result['eer_pct']:.2f}% | "
            f"{elapsed:.1f}s"
        )
        if is_snn and "spike_sparsity" in val_result:
            sp = val_result["spike_sparsity"]
            log_line += f" | sparsity={sp['sparsity']:.3f}"

        print(log_line)

        # History
        entry = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_eer": val_result["eer"],
            "val_eer_pct": val_result["eer_pct"],
        }
        if is_snn and "spike_sparsity" in val_result:
            entry["spike_sparsity"] = val_result["spike_sparsity"]
        history.append(entry)

        # Write history incrementally so we can monitor progress
        history_path = ckpt_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping on val EER
        if val_result["eer"] < best_eer:
            best_eer = val_result["eer"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "frontend_name": args.frontend,
                "model_name": args.model,
                "model_state": model.state_dict(),
                "frontend_state": frontend.state_dict() if frontend is not None else {},
                "config": config,
                "freq_dim": freq_dim,
                "time_dim": time_dim,
                "best_eer": best_eer,
            }
            torch.save(checkpoint, ckpt_dir / "best.pt")
            print(f"  >> Saved best checkpoint (EER={best_eer*100:.2f}%)")

            # Cross-domain mini-eval on improvement
            if cross_eval_loader is not None:
                cross_result = validate(
                    None, model, cross_eval_loader, criterion, device, is_snn
                )
                print(f"  >> [CROSS-EVAL] In-the-Wild probe EER: {cross_result['eer_pct']:.2f}%")
                entry["cross_eval_eer"] = cross_result["eer"]
                entry["cross_eval_eer_pct"] = cross_result["eer_pct"]
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_counter} epochs)")
                break

    # Final history save (also written incrementally per-epoch above)
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Checkpoint consistency check
    if args.cross_eval and best_eer < float("inf"):
        print(f"\n{'='*50}")
        print(f"  Checkpoint consistency check")
        print(f"{'='*50}")
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model_reload = build_model(
            args.model, config["model"], freq_dim, time_dim
        ).to(device)
        model_reload.load_state_dict(ckpt["model_state"])
        if channel_stats is not None and hasattr(model_reload, "set_channel_stats"):
            model_reload.set_channel_stats(
                channel_stats["mean"].to(device), channel_stats["std"].to(device)
            )
        reload_result = validate(
            None if use_precomputed else frontend,
            model_reload, dev_loader, criterion, device, is_snn,
        )
        print(f"  Training best EER: {best_eer*100:.2f}%")
        print(f"  Reloaded EER:      {reload_result['eer_pct']:.2f}%")
        delta = abs(best_eer - reload_result["eer"]) * 100
        if delta > 0.1:
            print(f"  WARNING: mismatch of {delta:.2f}% — checkpoint may be unreliable!")
        else:
            print(f"  OK — consistent (delta={delta:.3f}%)")
        del model_reload

    print(f"\nDone. Best val EER: {best_eer*100:.2f}%")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
