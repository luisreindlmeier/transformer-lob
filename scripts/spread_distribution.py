#!/usr/bin/env python3
"""Plot spread (best ask - best bid) boxplot for all tickers; save to assets."""
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


BEST_ASK_COL = cfg.LEN_ORDER + 0
BEST_BID_COL = cfg.LEN_ORDER + 2
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SPLITS = ("train", "val", "test")

LIGHT_GREEN = "#E0F4E0"
DARK_GREEN = "#008000"


def load_spreads(data_dir: Path, split: str) -> np.ndarray:
    path = data_dir / f"{split}.npy"
    if not path.exists():
        return np.array([])
    data = np.load(path)
    best_ask = data[:, BEST_ASK_COL].astype(np.float64)
    best_bid = data[:, BEST_BID_COL].astype(np.float64)
    spread = best_ask - best_bid
    valid = np.isfinite(spread) & (spread >= 0)
    return spread[valid]


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

    if not HAS_MPL:
        print("matplotlib not available; cannot generate plot.")
        sys.exit(1)

    ticker_spreads = {}
    for ticker in tickers:
        data_dir = base / ticker
        all_s = [load_spreads(data_dir, s) for s in SPLITS]
        all_s = [x for x in all_s if len(x) > 0]
        if all_s:
            ticker_spreads[ticker] = np.concatenate(all_s)

    if not ticker_spreads:
        print("No spread data loaded.")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    data_list = [ticker_spreads[t] for t in tickers]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        data_list,
        tick_labels=tickers,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=DARK_GREEN, linewidth=2.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(LIGHT_GREEN)
        patch.set_edgecolor("black")
    for el in bp["whiskers"], bp["caps"]:
        for line in el:
            line.set_color("black")

    ax.set_ylabel("Spread (best ask − best bid)")
    ax.set_xlabel("Ticker")
    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "spread_distribution_boxplot.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {ASSETS_DIR / 'spread_distribution_boxplot.png'}")


if __name__ == "__main__":
    main()
