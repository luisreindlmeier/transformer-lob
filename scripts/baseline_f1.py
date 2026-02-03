#!/usr/bin/env python3
"""Compute F1 (stationary class) and macro-F1 baseline when only the stationary class is predicted, per ticker."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score
from src import config as cfg


HORIZON = 10
HORIZON_TO_COL = {10: 4, 20: 3, 50: 2, 100: 1}
LABELS = [0, 1, 2]  # down, stat, up
STATIONARY_CLASS = 1


def get_labels(data_dir: Path, split: str, horizon: int = HORIZON) -> np.ndarray:
    path = data_dir / f"{split}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    col = HORIZON_TO_COL[horizon]
    labels = data[cfg.SEQ_SIZE - cfg.LEN_SMOOTH :, -col]
    labels = labels[np.isfinite(labels)]
    return labels.astype(np.int32)


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

    rows = []
    for ticker in tickers:
        data_dir = base / ticker
        try:
            test_y = get_labels(data_dir, "test")
        except Exception as e:
            print(f"  Skip {ticker}: {e}", file=sys.stderr)
            continue
        y_pred = np.full_like(test_y, STATIONARY_CLASS)
        f1_per_class = f1_score(test_y, y_pred, average=None, labels=LABELS, zero_division=0)
        macro_f1 = float(f1_score(test_y, y_pred, average="macro", labels=LABELS, zero_division=0))
        f1_stationary = float(f1_per_class[1])
        rows.append((ticker, f1_stationary, macro_f1))

    print("Baseline (predict only stationary class; evaluated on test):")
    print(f"  {'Ticker':<8}   {'F1 (stationary)':>16}   {'Macro F1':>12}")
    print("  " + "-" * 45)
    for ticker, f1_stat, macro in rows:
        print(f"  {ticker:<8}   {f1_stat:>16.4f}   {macro:>12.4f}")
    print("  " + "-" * 45)


if __name__ == "__main__":
    main()
