#!/usr/bin/env python3
"""
Fast evaluation using precomputed features.

Skips live feature extraction — loads precomputed .pt files directly.
Use when evaluate.py is too slow (especially for wavelet scattering).

Usage:
  # First precompute eval features:
  python -m src.data.precompute --config configs/default.yaml --frontends wavelet_v3 --splits eval

  # Then evaluate:
  python -m src.evaluation.evaluate_precomputed \
    --checkpoint results/checkpoints/wavelet_v3_order1_snn_kaldi/best.pt \
    --frontend wavelet_v3 \
    --eval-dir data/processed/asvspoof_eval \
    --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from src.data.precomputed_dataset import PrecomputedDataset, get_precomputed_dataloader
from src.data.datasets import ASVspoof2019LA, WaveFake, InTheWild, get_dataloader
from src.data.features import get_frontend
from src.evaluation.evaluate import load_experiment, evaluate_dataset
from src.evaluation.metrics import compute_all_metrics, compute_spike_sparsity


@torch.no_grad()
def evaluate_precomputed(
    model: nn.Module,
    loader,
    device: torch.device,
    is_snn: bool,
    dataset_name: str,
) -> dict:
    """Evaluate model on precomputed features."""
    model.eval()

    all_scores = []
    all_labels = []
    all_spikes = []

    for batch in tqdm(loader, desc=f"  {dataset_name}", leave=False):
        features, labels = batch[0], batch[1]
        features = features.to(device)

        if is_snn:
            logits, spk_rec = model(features)
            all_spikes.append(spk_rec.cpu().numpy())
        else:
            logits = model(features)

        log_probs = torch.log_softmax(logits.float(), dim=1)
        all_scores.append(log_probs[:, 1].cpu().numpy())
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        return {"dataset": dataset_name, "error": "no valid labels"}

    if valid_mask.sum() < len(labels):
        scores = scores[valid_mask]
        labels = labels[valid_mask]

    metrics = compute_all_metrics(labels, scores)
    metrics["dataset"] = dataset_name

    if is_snn and all_spikes:
        spikes = np.concatenate(all_spikes, axis=0)
        if valid_mask.sum() < len(valid_mask):
            spikes = spikes[valid_mask]
        metrics["spike_sparsity"] = compute_spike_sparsity(spikes)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate with precomputed features")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=256)
    # Optionally eval specific datasets via precomputed dirs
    parser.add_argument("--eval-dirs", nargs="*",
                        help="Precomputed dirs to eval (auto-detects if not given)")
    # Or eval raw datasets (live feature extraction, slow for wavelet)
    parser.add_argument("--raw-datasets", nargs="*",
                        help="Raw datasets to eval with live feature extraction")
    parser.add_argument("--output-name", default=None,
                        help="Custom output directory name under results/metrics/ (default: auto)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model from checkpoint
    print(f"\nLoading: {args.checkpoint}")
    frontend, model, ckpt_config, is_snn = load_experiment(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Experiment: {ckpt['frontend_name']}_{ckpt['model_name']}")
    print(f"Best training EER: {ckpt['best_eer']*100:.2f}%")

    data_cfg = config["data"]
    processed_root = Path(data_cfg["processed_root"])
    results = {"experiment": f"{args.frontend}_snn", "datasets": {}}

    # Auto-detect available precomputed eval sets
    eval_sets = {
        "asvspoof_eval": processed_root / "asvspoof_eval",
        "inthewild_full": processed_root / "inthewild_full",
    }

    for ds_name, ds_dir in eval_sets.items():
        feat_dir = ds_dir / args.frontend
        labels_path = ds_dir / "labels.pt"
        if not feat_dir.exists() or not labels_path.exists():
            print(f"  [SKIP] {ds_name}: no precomputed features at {feat_dir}")
            continue

        print(f"\n  Evaluating {ds_name} (precomputed)...")
        dataset = PrecomputedDataset(str(ds_dir), args.frontend)
        loader = get_precomputed_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        metrics = evaluate_precomputed(model, loader, device, is_snn, ds_name)
        results["datasets"][ds_name] = metrics
        print(f"  {ds_name} — EER: {metrics['eer_pct']:.2f}%  AUC: {metrics['auc']:.4f}")

    # Also try raw datasets (live feature extraction) for mel (fast) or small datasets
    raw_datasets = {
        "asvspoof_eval": lambda: ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="eval",
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
        "in_the_wild": lambda: InTheWild(
            root=data_cfg["inthewild_root"],
            max_samples=data_cfg["max_samples"], sample_rate=data_cfg["sample_rate"],
        ),
    }

    for ds_name, ds_factory in raw_datasets.items():
        if ds_name in results["datasets"]:
            continue  # already evaluated via precomputed
        if args.raw_datasets and ds_name not in args.raw_datasets:
            continue

        try:
            print(f"\n  Evaluating {ds_name} (live feature extraction)...")
            dataset = ds_factory()
            if len(dataset) == 0:
                continue
            loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
            metrics = evaluate_dataset(frontend, model, loader, device, is_snn, ds_name)
            results["datasets"][ds_name] = metrics
            if "error" not in metrics:
                print(f"  {ds_name} — EER: {metrics['eer_pct']:.2f}%  AUC: {metrics['auc']:.4f}")
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")

    # Save results
    out_name = args.output_name or f"{args.frontend}_snn_kaldi"
    out_dir = Path("results") / "metrics" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY: {args.frontend} + SNN")
    print(f"{'='*60}")
    for ds_name, m in results["datasets"].items():
        if "error" in m:
            continue
        print(f"  {ds_name:<20} EER: {m['eer_pct']:>6.2f}%  AUC: {m['auc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
