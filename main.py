#!/usr/bin/env python3
"""
Single entry point: python main.py <command> [options]
  run        – Train + evaluate + analyse (default pipeline)
  preprocess – Preprocess raw LOBSTER data
  train      – Train TLOB with Lightning only
  eval       – Evaluate saved model (best.pt), full metrics + confidence analysis
  quick-eval – Test-set only: accuracy + macro F1 → results-quick_eval.json
  backtest   – Backtest trained model (PnL, Sharpe, calibration plots)
"""
from src.cli import main

if __name__ == "__main__":
    main()
