#!/usr/bin/env python3
"""Print Table 4.4: Top-of-book liquidity at Level 1 (Mean/Median Size Ask and Bid). Uses raw orderbook CSVs for sizes in shares."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config as cfg

# Orderbook CSV: sell1, vsell1, buy1, vbuy1, ... (same order as src.data.preprocessing)
# Index 1 = ask size L1, 3 = bid size L1
VSELL1_COL = 1
VBUY1_COL = 3


def load_level1_sizes(raw_dir: Path):
    csv_files = sorted(raw_dir.glob("*.csv"))
    ob_files = [f for i, f in enumerate(csv_files) if i % 2 == 1]
    if not ob_files:
        return np.array([]), np.array([])
    ask_sizes, bid_sizes = [], []
    for path in ob_files:
        try:
            data = np.loadtxt(path, delimiter=",", usecols=(VSELL1_COL, VBUY1_COL), dtype=np.float64)
            ask_sizes.append(data[:, 0])
            bid_sizes.append(data[:, 1])
        except Exception:
            continue
    if not ask_sizes:
        return np.array([]), np.array([])
    return np.concatenate(ask_sizes), np.concatenate(bid_sizes)


def main():
    base = Path(cfg.DATA_DIR) / "raw"
    if not base.exists():
        print(f"Error: {base} not found. Table 4.4 requires raw LOBSTER orderbook CSVs.")
        sys.exit(1)

    tickers = sorted([d.name for d in base.iterdir() if d.is_dir()])
    if not tickers:
        print(f"No ticker directories in {base}.")
        sys.exit(1)

    rows = []
    for ticker in tickers:
        raw_dir = base / ticker
        ask_s, bid_s = load_level1_sizes(raw_dir)
        if len(ask_s) == 0:
            continue
        ask_s = ask_s[np.isfinite(ask_s)]
        bid_s = bid_s[np.isfinite(bid_s)]
        if len(ask_s) == 0 or len(bid_s) == 0:
            continue
        rows.append({
            "ticker": ticker,
            "mean_ask": np.mean(ask_s),
            "mean_bid": np.mean(bid_s),
            "median_ask": np.median(ask_s),
            "median_bid": np.median(bid_s),
        })

    if not rows:
        print("No orderbook size data loaded.")
        sys.exit(1)

    print("Table 4.4: Top-of-book liquidity at Level 1. Sizes = displayed queue volumes (shares).\n")
    print(f"  {'Ticker':<8}   {'Mean Size Ask':>14}   {'Mean Size Bid':>14}   {'Median Size Ask':>16}   {'Median Size Bid':>16}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r['ticker']:<8}   {r['mean_ask']:>14.1f}   {r['mean_bid']:>14.1f}   {r['median_ask']:>16.0f}   {r['median_bid']:>16.0f}")
    print("  " + "-" * 80)


if __name__ == "__main__":
    main()
