from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, roc_auc_score)
from src.evaluation.calibration import expected_calibration_error, maximum_calibration_error, brier_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2], zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    for i, cls in enumerate(["up", "stat", "down"]):
        metrics[f"precision_{cls}"] = float(precision[i])
        metrics[f"recall_{cls}"] = float(recall[i])
        metrics[f"f1_{cls}"] = float(f1_per_class[i])
    
    if y_prob is not None:
        try:
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["roc_auc_weighted"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except ValueError:
            pass
        ece, _, _, _ = expected_calibration_error(y_true, y_prob)
        metrics["ece"] = float(ece)
        metrics["mce"] = float(maximum_calibration_error(y_true, y_prob))
        metrics["brier_score"] = float(brier_score(y_true, y_prob))
    
    return metrics


def print_metrics(metrics: Dict[str, float], split: str = "test") -> None:
    print(f"\n  {split.upper()} Metrics:")
    print(f"  {'Metric':<22} {'Value':>10}")
    print(f"  {'-'*34}")
    for key in ["accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "balanced_accuracy"]:
        if key in metrics:
            print(f"  {key.replace('_', ' ').title():<22} {metrics[key]:>10.4f}")
    if "roc_auc_macro" in metrics:
        print(f"  {'ROC-AUC (macro)':<22} {metrics['roc_auc_macro']:>10.4f}")
    if "ece" in metrics:
        print(f"  {'-'*34}")
        for key in ["ece", "mce", "brier_score"]:
            print(f"  {key.upper():<22} {metrics[key]:>10.4f}")
    print(f"  {'-'*34}")
    for cls in ["up", "stat", "down"]:
        if f"f1_{cls}" in metrics:
            print(f"  {f'F1-{cls.title()}':<22} {metrics[f'f1_{cls}']:>10.4f}")
