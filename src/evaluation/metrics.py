#!/usr/bin/env python3
"""
Evaluation metrics for deepfake detection.

Primary metric: Equal Error Rate (EER)
Secondary: accuracy, F1, AUC-ROC, spike sparsity (for SNN models)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score,
    f1_score, confusion_matrix
)


def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate.
    
    Args:
        y_true: ground truth labels (0 = real, 1 = spoof)
        y_scores: predicted scores (higher = more likely spoof)
    
    Returns:
        eer: Equal Error Rate
        threshold: threshold at EER
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Guard: need both classes for a meaningful EER
    if len(np.unique(y_true)) < 2:
        return 1.0, 0.5

    # Guard: NaN scores (can happen with unstable SSM/SNN)
    if np.any(np.isnan(y_scores)):
        return 1.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER: point where FPR == FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x), 0.0, 1.0)
        thresh = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        # Fallback: find closest point
        diff = np.abs(fpr - fnr)
        if np.all(np.isnan(diff)):
            return 1.0, 0.5
        idx = np.nanargmin(diff)
        eer = (fpr[idx] + fnr[idx]) / 2
        thresh = thresholds[idx]

    return float(eer), float(thresh)


def compute_all_metrics(y_true, y_scores, threshold=None):
    """
    Compute full metric suite.
    
    Args:
        y_true: ground truth labels
        y_scores: predicted scores (higher = more likely spoof)
        threshold: decision threshold (if None, uses EER threshold)
    
    Returns:
        dict with eer, auc, accuracy, f1, threshold, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    eer, eer_thresh = compute_eer(y_true, y_scores)
    
    if threshold is None:
        threshold = eer_thresh
    
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        "eer": eer,
        "eer_pct": eer * 100,
        "auc": roc_auc_score(y_true, y_scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_spoof": int((y_true == 1).sum()),
    }


def compute_spike_sparsity(spike_tensor):
    """
    Compute spike sparsity metrics for SNN models.
    
    Args:
        spike_tensor: binary tensor of spikes (batch, timesteps, neurons)
    
    Returns:
        dict with mean_rate, sparsity, active_fraction
    """
    spike_tensor = np.asarray(spike_tensor)
    
    mean_rate = spike_tensor.mean()
    sparsity = 1.0 - mean_rate
    
    # Fraction of neurons that fire at least once per sample
    per_sample = spike_tensor.any(axis=1) if spike_tensor.ndim == 3 else spike_tensor
    active_fraction = per_sample.mean()
    
    return {
        "mean_spike_rate": float(mean_rate),
        "sparsity": float(sparsity),
        "active_neuron_fraction": float(active_fraction),
    }
