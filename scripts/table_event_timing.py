#!/usr/bin/env python3
"""Print Table 4.3: Event rate and inter-event timing (Mean/Median/Min/Max Delta t in seconds, Events/s). Reads raw message CSVs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config as cfg


def load_times_per_file(raw_dir: Path):
    """Load time column from each message file; return list of arrays and total duration in seconds.
    Message files = even index in sorted CSV list (same as src.data.preprocessing). Uses all rows (no limit)."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    message_files = [f for i, f in enumerate(csv_files) if i % 2 == 0]
    if not message_files:
        return [], 0.0
    all_times = []
    total_sec = 0.0
    for path in message_files:
        try:
            time_col = np.loadtxt(path, delimiter=",", usecols=(0,), dtype=np.float64)
            if len(time_col) >= 2:
                all_times.append(time_col)
                total_sec += float(time_col[-1] - time_col[0])
        except Exception:
            continue
    return all_times, total_sec


def main():
    base = Path(cfg.DATA_DIR) / "raw"
    if not base.exists():
        print(f"Error: {base} not found. Table 4.3 requires raw LOBSTER message CSVs.")
        sys.exit(1)

    tickers = sorted([d.name for d in base.iterdir() if d.is_dir()])
    if not tickers:
        print(f"No ticker directories in {base}.")
        sys.exit(1)

    rows = []
    for ticker in tickers:
        raw_dir = base / ticker
        file_times_list, total_sec = load_times_per_file(raw_dir)
        if not file_times_list or total_sec <= 0:
            continue
        times = np.concatenate(file_times_list)
        dt = np.concatenate([np.diff(t) for t in file_times_list])
        dt = dt[dt >= 0]
        if len(dt) == 0:
            continue
        total_events = len(times)
        events_per_sec = total_events / total_sec if total_sec > 0 else 0
        rows.append({
            "ticker": ticker,
            "mean_dt": np.mean(dt),
            "median_dt": np.median(dt),
            "min_dt": np.min(dt),
            "max_dt": np.max(dt),
            "events_per_sec": events_per_sec,
        })

    if not rows:
        print("No timing data loaded.")
        sys.exit(1)

    print("Table 4.3: Event rate and inter-event timing. Inter-event times Delta t in seconds.\n")
    print(f"  {'Ticker':<8}   {'Mean Delta t':>14}   {'Median Delta t':>16}   {'Min Delta t':>12}   {'Max Delta t':>12}   {'Events/s':>10}")
    print("  " + "-" * 90)
    for r in rows:
        print(f"  {r['ticker']:<8}   {r['mean_dt']:>14.4f}   {r['median_dt']:>16.5f}   {r['min_dt']:>12.4f}   {r['max_dt']:>12.4f}   {r['events_per_sec']:>10.2f}")
    print("  " + "-" * 90)


if __name__ == "__main__":
    main()
