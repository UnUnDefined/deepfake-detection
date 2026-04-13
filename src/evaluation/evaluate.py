#!/usr/bin/env python3
"""
Multi-dataset evaluation for the 2x2 factorial experiment.

Loads a trained checkpoint and evaluates on:
  1. ASVspoof 2019 LA (eval split) — in-domain
  2. WaveFake — cross-domain
  3. In-the-Wild — cross-domain

The generalization story lives in the gap between (1) and (2)/(3).

Usage:
  python -m src.evaluation.evaluate --checkpoint results/checkpoints/wavelet_snn/best.pt
  python -m src.evaluation.evaluate --checkpoint results/checkpoints/mel_resnet/best.pt

  # Evaluate all four experiments:
  python -m src.evaluation.evaluate --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.datasets import (
    ASVspoof2019LA, WaveFake, InTheWild, get_dataloader,
)
from src.data.features import get_frontend
from src.models.resnet import ResNet18Classifier
from src.models.snn import SpikingClassifierV2
from src.evaluation.metrics import compute_all_metrics, compute_spike_sparsity


def load_experiment(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, nn.Module, dict, bool]:
    """Load frontend + model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    frontend_name = ckpt["frontend_name"]
    model_name = ckpt["model_name"]
    freq_dim = ckpt["freq_dim"]
    time_dim = ckpt["time_dim"]

    # Rebuild frontend
    frontend = get_frontend(frontend_name, config["features"]).to(device)
    frontend_state = ckpt.get("frontend_state", {})
    if frontend_state:
        try:
            frontend.load_state_dict(frontend_state)
        except RuntimeError:
            # Wavelet frontend uses lazy init, no saved state to load
            pass

    # Rebuild model
    is_snn = "snn" in model_name
    if model_name == "resnet":
        model = ResNet18Classifier(
            num_classes=config["model"]["resnet"]["num_classes"],
        )
    elif "snn" in model_name:
        cfg = config["model"].get("snn_v2", config["model"].get("snn", {}))
        model = SpikingClassifierV2(
            freq_dim=freq_dim,
            time_dim=time_dim,
            num_steps=cfg["num_steps"],
            beta_init=cfg.get("beta_init", cfg.get("beta", 0.8)),
            threshold_init=cfg.get("threshold_init", cfg.get("threshold", 0.3)),
            num_classes=cfg["num_classes"],
            firing_rate_target=cfg.get("firing_rate_target", 0.2),
            firing_rate_lambda=cfg.get("firing_rate_lambda", 0.05),
            temporal_mode=cfg.get("temporal_mode", "conv"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    return frontend, model, config, is_snn


@torch.no_grad()
def evaluate_dataset(
    frontend: nn.Module,
    model: nn.Module,
    loader,
    device: torch.device,
    is_snn: bool,
    dataset_name: str,
) -> dict:
    """Run inference on a dataset and compute metrics."""
    frontend.eval()
    model.eval()

    all_scores = []
    all_labels = []
    all_spikes = []

    for waveforms, labels, _meta in tqdm(loader, desc=f"  {dataset_name}", leave=False):
        waveforms = waveforms.to(device)

        features = frontend(waveforms)

        if is_snn:
            logits, spk_rec = model(features)
            all_spikes.append(spk_rec.cpu().numpy())
        else:
            logits = model(features)

        # Use log_softmax for numerically stable scoring with large logits
        log_probs = torch.log_softmax(logits.float(), dim=1)
        all_scores.append(log_probs[:, 1].cpu().numpy())
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Skip if labels are unknown (-1)
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        print(f"    [WARN] No valid labels for {dataset_name}, skipping metrics")
        return {"dataset": dataset_name, "error": "no valid labels"}

    if valid_mask.sum() < len(labels):
        print(f"    [WARN] {(~valid_mask).sum()} samples with unknown labels, skipping those")
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


def evaluate_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Evaluate a single checkpoint across all datasets."""
    print(f"\nLoading: {checkpoint_path}")
    frontend, model, config, is_snn = load_experiment(checkpoint_path, device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment_name = f"{ckpt['frontend_name']}_{ckpt['model_name']}"
    print(f"Experiment: {experiment_name}")
    print(f"Best training EER: {ckpt['best_eer']*100:.2f}%")

    data_cfg = config["data"]
    batch_size = config["training"]["batch_size"]
    results = {"experiment": experiment_name, "datasets": {}}

    # --- ASVspoof 2019 LA eval (in-domain) ---
    try:
        asvspoof_eval = ASVspoof2019LA(
            root=data_cfg["asvspoof_root"], split="eval",
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        loader = get_dataloader(asvspoof_eval, batch_size=batch_size, shuffle=False)
        metrics = evaluate_dataset(
            frontend, model, loader, device, is_snn, "asvspoof_eval"
        )
        results["datasets"]["asvspoof_eval"] = metrics
        print(f"  ASVspoof eval — EER: {metrics['eer_pct']:.2f}%  AUC: {metrics['auc']:.4f}")
    except FileNotFoundError as e:
        print(f"  [SKIP] ASVspoof eval: {e}")

    # --- WaveFake (cross-domain) ---
    try:
        wavefake = WaveFake(
            root=data_cfg["wavefake_root"],
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        if len(wavefake) > 0:
            loader = get_dataloader(wavefake, batch_size=batch_size, shuffle=False)
            metrics = evaluate_dataset(
                frontend, model, loader, device, is_snn, "wavefake"
            )
            results["datasets"]["wavefake"] = metrics
            print(f"  WaveFake     — EER: {metrics['eer_pct']:.2f}%  AUC: {metrics['auc']:.4f}")
        else:
            print("  [SKIP] WaveFake: no samples found")
    except (FileNotFoundError, OSError) as e:
        print(f"  [SKIP] WaveFake: {e}")

    # --- In-the-Wild (cross-domain) ---
    try:
        inthewild = InTheWild(
            root=data_cfg["inthewild_root"],
            max_samples=data_cfg["max_samples"],
            sample_rate=data_cfg["sample_rate"],
        )
        if len(inthewild) > 0:
            loader = get_dataloader(inthewild, batch_size=batch_size, shuffle=False)
            metrics = evaluate_dataset(
                frontend, model, loader, device, is_snn, "in_the_wild"
            )
            results["datasets"]["in_the_wild"] = metrics
            if "error" not in metrics:
                print(f"  In-the-Wild  — EER: {metrics['eer_pct']:.2f}%  AUC: {metrics['auc']:.4f}")
        else:
            print("  [SKIP] In-the-Wild: no samples found")
    except (FileNotFoundError, OSError) as e:
        print(f"  [SKIP] In-the-Wild: {e}")

    # Save results
    out_dir = Path("results") / "metrics" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")

    return results


def print_summary_table(all_results: list[dict]):
    """Print a 2x2 factorial results table for the paper."""
    print(f"\n{'='*72}")
    print("  2x2 FACTORIAL RESULTS — EER (%)")
    print(f"{'='*72}")

    # Build lookup: experiment_name -> dataset -> eer_pct
    lookup = {}
    for r in all_results:
        name = r["experiment"]
        lookup[name] = {}
        for ds_name, ds_metrics in r["datasets"].items():
            if "eer_pct" in ds_metrics:
                lookup[name][ds_name] = ds_metrics["eer_pct"]

    datasets = ["asvspoof_eval", "wavefake", "in_the_wild"]
    experiments = ["mel_resnet", "wavelet_resnet", "mel_snn", "wavelet_snn"]
    labels = {
        "mel_resnet": "ResNet + Mel",
        "wavelet_resnet": "ResNet + Wavelet",
        "mel_snn": "SNN + Mel",
        "wavelet_snn": "SNN + Wavelet",
    }
    ds_labels = {
        "asvspoof_eval": "ASVspoof",
        "wavefake": "WaveFake",
        "in_the_wild": "In-Wild",
    }

    # Header
    header = f"{'Experiment':<22}"
    for ds in datasets:
        header += f" | {ds_labels.get(ds, ds):>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for exp in experiments:
        if exp not in lookup:
            continue
        row = f"{labels.get(exp, exp):<22}"
        for ds in datasets:
            val = lookup[exp].get(ds)
            if val is not None:
                row += f" | {val:>9.2f}%"
            else:
                row += f" | {'—':>10}"
        print(row)

    print(f"{'='*72}")

    # Generalization gap
    print("\n  Generalization Gap (WaveFake EER - ASVspoof EER):")
    for exp in experiments:
        if exp not in lookup:
            continue
        asv = lookup[exp].get("asvspoof_eval")
        wf = lookup[exp].get("wavefake")
        if asv is not None and wf is not None:
            gap = wf - asv
            print(f"    {labels.get(exp, exp):<22}: {gap:+.2f} pp")

    # Spike sparsity for SNN models
    snn_exps = [e for e in experiments if "snn" in e and e in lookup]
    if snn_exps:
        print("\n  SNN Spike Sparsity:")
        for r in all_results:
            if r["experiment"] not in snn_exps:
                continue
            for ds_name, ds_metrics in r["datasets"].items():
                sp = ds_metrics.get("spike_sparsity", {})
                if sp:
                    print(f"    {labels.get(r['experiment'])} / {ds_labels.get(ds_name, ds_name)}: "
                          f"sparsity={sp['sparsity']:.3f}, "
                          f"active={sp['active_neuron_fraction']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to a single checkpoint",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all four experiments from results/checkpoints/",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.checkpoint:
        evaluate_checkpoint(args.checkpoint, device)

    elif args.all:
        ckpt_base = Path("results") / "checkpoints"
        experiments = ["mel_resnet", "wavelet_resnet", "mel_snn", "wavelet_snn"]
        all_results = []

        for exp in experiments:
            ckpt_path = ckpt_base / exp / "best.pt"
            if ckpt_path.exists():
                result = evaluate_checkpoint(str(ckpt_path), device)
                all_results.append(result)
            else:
                print(f"\n[SKIP] No checkpoint for {exp} at {ckpt_path}")

        if all_results:
            print_summary_table(all_results)

            # Save combined results
            combined_path = Path("results") / "metrics" / "factorial_results.json"
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nCombined results: {combined_path}")
    else:
        parser.error("Provide --checkpoint or --all")


if __name__ == "__main__":
    main()
