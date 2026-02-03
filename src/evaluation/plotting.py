from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

COLORS = {"green": "#A8D5BA", "red": "#F5B7B1", "blue": "#AED6F1", "orange": "#FAD7A0",
          "purple": "#D7BDE2", "teal": "#A3E4D7", "gray": "#D5D8DC", "yellow": "#F9E79F"}
CLASS_NAMES = ["Up", "Stationary", "Down"]  # 0=UP, 1=stat, 2=DOWN (matches preprocessing labeling)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, split: str, save_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("pastel_blue", ["#FFFFFF", COLORS["blue"], "#2E86AB"])
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(3), yticks=np.arange(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel="True Label", xlabel="Predicted Label")
    ax.set_title(f"{model_name} - {split.upper()}", fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center",
                   color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def plot_training_history(history: Dict, model_name: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], color=COLORS["blue"], linewidth=2, marker='o', markersize=4, label="Train")
    axes[0].plot(epochs, history["val_loss"], color=COLORS["orange"], linewidth=2, marker='s', markersize=4, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_f1"], color=COLORS["green"], linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"{model_name} - Val F1", fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, model_name: str,
                             save_path: Path, n_bins: int = 10) -> None:
    from src.evaluation.calibration import expected_calibration_error, maximum_calibration_error
    ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)
    data = {"bin_edges": np.linspace(0, 1, n_bins + 1), "bin_accuracies": bin_acc, "bin_confidences": bin_conf,
            "bin_counts": bin_counts, "ece": ece, "mce": maximum_calibration_error(y_true, y_prob, n_bins)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    bin_centers = (data["bin_edges"][:-1] + data["bin_edges"][1:]) / 2
    width = 1 / n_bins * 0.8

    ax1.bar(bin_centers, data["bin_accuracies"], width=width, color="#AED6F1", edgecolor='white', label="Accuracy")
    ax1.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=2, label="Perfect")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name} - ECE={data['ece']:.3f}", fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    total = sum(data["bin_counts"])
    rel_counts = [c / total if total > 0 else 0 for c in data["bin_counts"]]
    ax2.bar(bin_centers, rel_counts, width=width, color="#A8D5BA", edgecolor='white')
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Relative Frequency")
    ax2.set_title(f"{model_name} - Confidence Distribution", fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def plot_reliability_diagram_before_after(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    model_name: str,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    from src.evaluation.calibration import expected_calibration_error
    ece_b, bin_acc_b, bin_conf_b, _ = expected_calibration_error(y_true, y_prob_before, n_bins)
    ece_a, bin_acc_a, bin_conf_a, _ = expected_calibration_error(y_true, y_prob_after, n_bins)
    bin_centers = (np.linspace(0, 1, n_bins + 1)[:-1] + np.linspace(0, 1, n_bins + 1)[1:]) / 2
    width = 1 / n_bins * 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bin_centers - width / 2, bin_acc_b, width=width, color=COLORS["blue"], edgecolor="white", label=f"Before scaling (ECE={ece_b:.3f})")
    ax.bar(bin_centers + width / 2, bin_acc_a, width=width, color=COLORS["teal"], edgecolor="white", label=f"After scaling (ECE={ece_a:.3f})")
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Confidence (bin center)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} – Reliability diagram (before vs after temperature scaling)", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
