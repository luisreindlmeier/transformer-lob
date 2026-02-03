#!/usr/bin/env python3
"""Print class distribution (relative %) for train/val/test.npy for all tickers; generate academic plots."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config as cfg

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


HORIZON = 10
HORIZON_TO_COL = {10: 4, 20: 3, 50: 2, 100: 1}
SPLITS = ("train", "val", "test")
CLASS_LABELS = ("Down", "Neutral", "Up")  # 0, 1, 2
# Academic bar colors: light green, light blue, light red (match thesis figure)
BAR_COLORS = ["#90EE90", "#87CEEB", "#F5B7B1"]
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def get_labels(data_dir: Path, split: str, horizon: int = HORIZON) -> np.ndarray:
    path = data_dir / f"{split}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    col = HORIZON_TO_COL[horizon]
    labels = data[cfg.SEQ_SIZE - cfg.LEN_SMOOTH :, -col]
    labels = labels[np.isfinite(labels)]
    return labels.astype(np.int32)


def plot_split_distribution(
    tickers: list,
    split_data: dict,
    split: str,
    out_path: Path,
) -> None:
    """One bar chart per split: tickers on x-axis, relative frequency (3 bars per ticker)."""
    if not HAS_MPL:
        return
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tickers))
    width = 0.25
    for i, cls in enumerate((0, 1, 2)):
        vals = [split_data[t][i] / 100.0 for t in tickers]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=CLASS_LABELS[i], color=BAR_COLORS[i], edgecolor="white", linewidth=0.5)
        for b in bars:
            h = b.get_height()
            if h >= 0.01:
                ax.annotate(f"{h:.2%}", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="normal")
    ax.set_ylabel("Relative class frequency")
    ax.set_xlabel("Ticker")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    base = Path(cfg.DATA_DIR) / "preprocessed"
    if not base.exists():
        print(f"Error: {base} not found.")
        sys.exit(1)

    tickers = sorted([
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "test.npy").exists()
    ])
    if not tickers:
        print(f"No ticker directories with test.npy found under {base}")
        sys.exit(1)

    # Collect rows and per-split data for plots
    rows = []
    by_split = {s: {} for s in SPLITS}  # split -> ticker -> (p0, p1, p2)
    for ticker in tickers:
        data_dir = base / ticker
        for split in SPLITS:
            try:
                labels = get_labels(data_dir, split)
            except Exception as e:
                print(f"  Skip {ticker} {split}: {e}", file=sys.stderr)
                continue
            total = len(labels)
            p0 = 100.0 * (labels == 0).sum() / total if total else 0.0
            p1 = 100.0 * (labels == 1).sum() / total if total else 0.0
            p2 = 100.0 * (labels == 2).sum() / total if total else 0.0
            rows.append((ticker, split, p0, p1, p2))
            by_split[split][ticker] = (p0, p1, p2)

    # Table with separators between tickers
    print("Class distribution (relative %, horizon=10):")
    print(f"  {'Ticker':<8}   {'Split':<6}   {'Down%':>8}   {'Neutral%':>10}   {'Up%':>8}")
    print("  " + "-" * 50)
    prev_ticker = None
    for ticker, split, p0, p1, p2 in rows:
        if prev_ticker is not None and ticker != prev_ticker:
            print("  " + "-" * 50)
        prev_ticker = ticker
        print(f"  {ticker:<8}   {split:<6}   {p0:>7.1f}%   {p1:>9.1f}%   {p2:>7.1f}%")
    print("  " + "-" * 50)

    # Plots: one per split (only tickers that have data for this split)
    if HAS_MPL:
        for split in SPLITS:
            tickers_in_split = [t for t in tickers if t in by_split[split]]
            if tickers_in_split:
                out_path = ASSETS_DIR / f"class_distribution_{split}.png"
                plot_split_distribution(tickers_in_split, by_split[split], split, out_path)
                print(f"  Saved: {out_path}")
    else:
        print("  (matplotlib not available; skipping plots)")


if __name__ == "__main__":
    main()
