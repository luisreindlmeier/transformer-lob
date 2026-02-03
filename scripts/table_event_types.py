#!/usr/bin/env python3
"""Print Table 4.2: Relative frequencies of LOB event types (percent). Reads raw message CSVs from data/raw/{ticker}/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config as cfg

# LOBSTER: 1=Submission, 2=Partial cancel, 3=Deletion (total), 4=Visible execution, 5=Hidden execution
EVENT_NAMES = {
    1: "Limit order submission",
    2: "Partial cancellation",
    3: "Total cancellation",
    4: "Visible execution",
    5: "Hidden execution",
}


def load_event_types(raw_dir: Path) -> np.ndarray:
    csv_files = sorted(raw_dir.glob("*.csv"))
    message_files = [f for i, f in enumerate(csv_files) if i % 2 == 0]
    if not message_files:
        return np.array([], dtype=np.int32)
    # message columns: time, event_type, order_id, size, price, direction
    event_col = 1
    all_events = []
    for path in message_files:
        try:
            data = np.loadtxt(path, delimiter=",", usecols=(event_col,), dtype=np.int32, max_rows=10_000_000)
            all_events.append(data)
        except Exception:
            continue
    if not all_events:
        return np.array([], dtype=np.int32)
    return np.concatenate(all_events)


def main():
    base = Path(cfg.DATA_DIR) / "raw"
    if not base.exists():
        print(f"Error: {base} not found. Table 4.2 requires raw LOBSTER message CSVs.")
        sys.exit(1)

    tickers = sorted([d.name for d in base.iterdir() if d.is_dir()])
    if not tickers:
        print(f"No ticker directories in {base}. Add raw message/orderbook CSVs per ticker.")
        sys.exit(1)

    # Count event types per ticker (only 1..5; 6,7 excluded to match table)
    ticker_counts = {}
    for ticker in tickers:
        raw_dir = base / ticker
        events = load_event_types(raw_dir)
        if len(events) == 0:
            continue
        events = events[(events >= 1) & (events <= 5)]
        counts = np.bincount(events, minlength=6)[1:6]
        ticker_counts[ticker] = counts

    if not ticker_counts:
        print("No event data loaded.")
        sys.exit(1)

    tickers = sorted(ticker_counts.keys())
    n_total = sum(ticker_counts[t].sum() for t in tickers)
    col_width = 10
    header = f"  {'Event type':<28}  " + "  ".join(f"{t:>{col_width}}" for t in tickers) + f"  {'Average':>{col_width}}"
    sep = "  " + "-" * (28 + (col_width + 2) * (len(tickers) + 1))

    print("Table 4.2: Relative frequencies of LOB event types. All values in percent.\n")
    print(header)
    print(sep)

    for code in (1, 2, 3):
        row_name = EVENT_NAMES[code]
        vals = []
        for t in tickers:
            c = ticker_counts[t][code - 1]
            tot = ticker_counts[t].sum()
            pct = 100.0 * c / tot if tot else 0
            vals.append(pct)
        avg = np.mean(vals)
        vals_str = "  ".join(f"{v:>{col_width}.2f}" for v in vals)
        print(f"  {row_name:<28}  {vals_str}  {avg:>{col_width}.2f}")

    # Subtotal: Liquidity-providing
    liq_vals = []
    for t in tickers:
        tot = ticker_counts[t].sum()
        c = ticker_counts[t][0] + ticker_counts[t][1] + ticker_counts[t][2]
        pct = 100.0 * c / tot if tot else 0
        liq_vals.append(pct)
    liq_avg = np.mean(liq_vals)
    liq_str = "  ".join(f"{v:>{col_width}.2f}" for v in liq_vals)
    print(sep)
    print(f"  {'Liquidity-providing events':<28}  {liq_str}  {liq_avg:>{col_width}.2f}")
    print(sep)

    for code in (4, 5):
        row_name = EVENT_NAMES[code]
        vals = []
        for t in tickers:
            c = ticker_counts[t][code - 1]
            tot = ticker_counts[t].sum()
            pct = 100.0 * c / tot if tot else 0
            vals.append(pct)
        avg = np.mean(vals)
        vals_str = "  ".join(f"{v:>{col_width}.2f}" for v in vals)
        print(f"  {row_name:<28}  {vals_str}  {avg:>{col_width}.2f}")

    exec_vals = []
    for t in tickers:
        tot = ticker_counts[t].sum()
        c = ticker_counts[t][3] + ticker_counts[t][4]
        pct = 100.0 * c / tot if tot else 0
        exec_vals.append(pct)
    exec_avg = np.mean(exec_vals)
    exec_str = "  ".join(f"{v:>{col_width}.2f}" for v in exec_vals)
    print(sep)
    print(f"  {'Execution events':<28}  {exec_str}  {exec_avg:>{col_width}.2f}")
    print(sep)


if __name__ == "__main__":
    main()
