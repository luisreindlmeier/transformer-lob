import json
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from src import config as cfg


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies, bin_confidences = np.zeros(n_bins), np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
    
    ece = np.sum(bin_counts / len(y_true) * np.abs(bin_accuracies - bin_confidences))
    return float(ece), bin_accuracies, bin_confidences, bin_counts


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    _, bin_acc, bin_conf, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)
    mask = bin_counts > 0
    return float(np.max(np.abs(bin_acc[mask] - bin_conf[mask]))) if np.any(mask) else 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_samples, n_classes = y_prob.shape
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)
    return {"bin_edges": np.linspace(0, 1, n_bins + 1), "bin_accuracies": bin_acc, "bin_confidences": bin_conf,
            "bin_counts": bin_counts, "ece": ece, "mce": maximum_calibration_error(y_true, y_prob, n_bins)}


class TemperatureScaler(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1, device=device))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def set_temperature(self, val_loader: DataLoader, max_iter: int = 50, lr: float = 0.01, use_tqdm: bool = False) -> float:
        self.model.eval()
        all_logits, all_labels = [], []
        loader = val_loader
        if use_tqdm:
            from tqdm import tqdm
            loader = tqdm(val_loader, desc="  Temperature scaling", unit="batch", leave=True, mininterval=0.5)
        with torch.no_grad():
            for X_batch, y_batch in loader:
                all_logits.append(self.model(X_batch.to(self.device)).cpu())
                all_labels.append(y_batch)
        all_logits = torch.cat(all_logits, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(all_logits), all_labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        return float(self.temperature.item())


def confidence_binned_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    confidences = np.max(y_prob, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_f1s, bin_counts = np.zeros(n_bins), np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] >= 3:
            bin_f1s[i] = f1_score(y_true[in_bin], y_pred[in_bin], average="macro", labels=[0, 1, 2], zero_division=0)
    
    corr, _ = spearmanr(confidences, (y_pred == y_true).astype(np.float64))
    valid_bins = bin_counts >= 3
    monotonic = np.sum(np.diff(bin_f1s[valid_bins]) >= -0.01) >= len(bin_f1s[valid_bins]) * 0.7 - 1 if np.sum(valid_bins) >= 2 else False
    
    return {"bin_edges": bin_boundaries, "bin_f1s": bin_f1s, "bin_counts": bin_counts,
            "bin_coverages": bin_counts / len(y_true), "correlation": float(corr) if not np.isnan(corr) else 0.0, "monotonic": bool(monotonic)}


def optimal_confidence_threshold(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                                  min_coverage: float = 0.1) -> Tuple[float, float, float]:
    confidences = np.max(y_prob, axis=1)
    best_threshold, best_f1, best_coverage = 0.33, 0.0, 1.0
    for thresh in np.linspace(0.33, 0.95, 50):
        mask = confidences >= thresh
        coverage = np.mean(mask)
        if coverage < min_coverage: break
        if np.sum(mask) >= 3:
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", labels=[0, 1, 2], zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold, best_coverage = f1, thresh, coverage
    return float(best_threshold), float(best_f1), float(best_coverage)


def compute_sharpe_ratio(returns: np.ndarray, n_trades: Optional[int] = None) -> float:
    if len(returns) == 0:
        return 0.0
    std = np.std(returns)
    raw = float(np.mean(returns) / std) if std > 0 and not np.isnan(std) else 0.0
    if n_trades is not None and n_trades > 0:
        return raw * float(np.sqrt(n_trades))
    return raw


def strategy_returns(y_pred: np.ndarray, y_ret: np.ndarray) -> np.ndarray:
    strat_ret = np.zeros_like(y_ret, dtype=np.float64)
    strat_ret[y_pred == 0] = y_ret[y_pred == 0]   # Long when predict UP
    strat_ret[y_pred == 2] = -y_ret[y_pred == 2]   # Short when predict DOWN
    return strat_ret


def filtered_strategy_analysis(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, y_ret: np.ndarray,
                               confidence_thresholds: Optional[np.ndarray] = None) -> dict:
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(0.33, 0.95, 25)
    
    confidences = np.max(y_prob, axis=1)
    baseline_sharpe = compute_sharpe_ratio(strategy_returns(y_pred, y_ret))
    sharpe_ratios, f1_scores, coverages = [], [], []

    for thresh in confidence_thresholds:
        mask = confidences >= thresh
        n_above = np.sum(mask)
        if n_above >= 10:
            filtered_strat_ret = strategy_returns(y_pred[mask], y_ret[mask])
            sharpe_ratios.append(compute_sharpe_ratio(filtered_strat_ret))
            f1_scores.append(f1_score(y_true[mask], y_pred[mask], average="macro", labels=[0, 1, 2], zero_division=0))
        else:
            sharpe_ratios.append(0.0)
            f1_scores.append(0.0)
        coverages.append(np.mean(mask))

    sharpe_ratios, coverages = np.array(sharpe_ratios), np.array(coverages)
    valid_mask = coverages >= 0.1
    if np.any(valid_mask):
        valid_sharpes = sharpe_ratios.copy()
        valid_sharpes[~valid_mask] = -np.inf
        best_idx = np.argmax(valid_sharpes)
        optimal_threshold, optimal_sharpe = confidence_thresholds[best_idx], sharpe_ratios[best_idx]
    else:
        optimal_threshold, optimal_sharpe = confidence_thresholds[0], baseline_sharpe

    return {"thresholds": confidence_thresholds, "sharpe_ratios": sharpe_ratios, "f1_scores": np.array(f1_scores),
            "coverages": coverages, "optimal_threshold": float(optimal_threshold),
            "optimal_sharpe": float(optimal_sharpe), "baseline_sharpe": float(baseline_sharpe)}


def analyze_confidence(model: torch.nn.Module, val_loader: DataLoader, test_loader: DataLoader,
                       y_ret_test: Optional[np.ndarray] = None, output_dir: Optional[Path] = None,
                       model_name: str = "Model", device: str = cfg.DEVICE) -> dict:
    from src.evaluation.plotting import plot_reliability_diagram
    output_dir = output_dir or Path("./results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\n  H3 Confidence Analysis: {model_name}\n{'='*60}")
    model = model.to(device)
    model.eval()
    
    scaler = TemperatureScaler(model, device)
    optimal_temp = scaler.set_temperature(val_loader)
    print(f"  Optimal temperature: {optimal_temp:.3f}")
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            all_logits.append(model(X_batch.to(device)).cpu())
            all_labels.append(y_batch)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    probs_uncal = torch.softmax(all_logits, dim=1).numpy()
    preds = np.argmax(probs_uncal, axis=1)
    probs_cal = torch.softmax(all_logits / optimal_temp, dim=1).numpy()
    
    ece_uncal, _, _, _ = expected_calibration_error(all_labels, probs_uncal)
    ece_cal, _, _, _ = expected_calibration_error(all_labels, probs_cal)
    print(f"  ECE: {ece_uncal:.4f} -> {ece_cal:.4f} (calibrated)")
    
    binned_metrics = confidence_binned_metrics(all_labels, preds, probs_cal)
    print(f"  Confidence-Correctness Correlation: {binned_metrics['correlation']:.3f}")
    
    plot_path = output_dir / f"reliability_{model_name.replace(' ', '_').replace('-', '_')}.png"
    plot_reliability_diagram(all_labels, probs_cal, model_name, plot_path)
    
    results = {"optimal_temperature": optimal_temp,
               "calibration": {"uncalibrated": {"ece": ece_uncal}, "calibrated": {"ece": ece_cal}},
               "binned_metrics": {"correlation": binned_metrics["correlation"], "monotonic": binned_metrics["monotonic"]},
               "plot_path": str(plot_path)}
    
    if y_ret_test is not None:
        results["backtesting"] = filtered_strategy_analysis(all_labels, preds, probs_cal, y_ret_test)
    
    json_path = output_dir / f"confidence_{model_name.replace(' ', '_').replace('-', '_')}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    return results
