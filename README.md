# Transformer-Based LOB Prediction

Bachelor thesis: *Transformer-Based Prediction of Limit Order Book Dynamics for Market Making*

## Project Structure

The project has a **single package**: `src`. Entry point: `python main.py <command> ...`

```
├── src/
│   ├── config.py              # Configuration constants
│   ├── cli.py                 # Unified CLI entry point
│   ├── cli_helpers.py         # Data loading, run resolution, evaluation loop
│   ├── data/
│   │   ├── dataset.py         # LOBDataset, LOBDataModule
│   │   ├── loading.py         # Data loading utilities
│   │   └── preprocessing.py   # LOBSTER preprocessing
│   ├── models/
│   │   ├── components.py      # BiN, MLP, positional embeddings
│   │   ├── attention.py       # Standard & Decay attention
│   │   ├── tlob.py            # TLOB & TLOB-Decay
│   │   ├── deeplob.py         # DeepLOB (CNN-GRU)
│   │   └── lit.py             # LiT Transformer
│   ├── training/
│   │   ├── trainer.py         # Lightning training module
│   │   └── loops.py           # Vanilla PyTorch training
│   ├── evaluation/
│   │   ├── metrics.py         # Accuracy, F1, MCC, etc.
│   │   ├── calibration.py     # Calibration & temperature scaling
│   │   ├── backtest.py        # PnL, Sharpe, MDD, confidence plots
│   │   ├── decay_analysis.py  # Decay rate analysis (TLOB-Decay)
│   │   └── plotting.py        # Visualization
│   └── utils/
│       ├── seed.py            # Reproducibility
│       ├── helpers.py         # Utility functions
│       └── logging.py         # Logging setup
├── data/
│   ├── raw/{TICKER}/          # Raw LOBSTER CSV files
│   └── preprocessed/{TICKER}/
├── scripts/                   # Scripts for thesis tables/figures
└── results/                   # Outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Configuration:** Code constants (paths, sequence length, device) are in `src/config.py`. Experiment defaults (epochs, lr, seed, model choice) are in `params.toml` and can be overridden with CLI flags.

## Data

**Source**: LOBSTER Level 2 data (academic license from [lobsterdata.com](https://lobsterdata.com))

Place raw data in `data/raw/{TICKER}/`:

```
data/raw/CSCO/
├── CSCO_2023-07-03_34200000_57600000_message_10.csv
├── CSCO_2023-07-03_34200000_57600000_orderbook_10.csv
└── ...
```

Format: LOBSTER message + orderbook CSVs.

## Usage

**Single entry point:** `python main.py <command> [options]`

```bash
# Preprocessing
python main.py preprocess --ticker CSCO

# Full pipeline (train + evaluate + analyze)
python main.py run --ticker CSCO --model TLOB --epochs 5

# With decay attention
python main.py run --ticker CSCO --model TLOB --decay --epochs 5
```

Results can be verified faster using the pre-computed `.pt` files:`

```bash
# Quick eval (test accuracy + macro F1 only → results-quick_eval.json)
python main.py quick-eval --run-dir results/CSCO_TLOB

# Backtest
python main.py backtest --run-dir results/CSCO_TLOB
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Preprocess raw LOBSTER data |
| `train` | Train TLOB with Lightning (EMA, LR scheduling) |
| `run` | Full pipeline with training, evaluation, and analysis |
| `eval` | Evaluate saved model (best.pt), confidence analysis, full metrics |
| `quick-eval` | Test-set only: accuracy + macro F1 → results-quick_eval.json |
| `backtest` | Backtest trained model: PnL, Sharpe, MDD, hit rate, confidence plots |

### Main arguments (run)

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker (CSCO, INTC, ...) |
| `--model` | TLOB | Model: TLOB, DeepLOB, LiT, or Majority (baseline) |
| `--decay` | False | Use decay variant (TLOB only) |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--horizon` | 10 | Prediction horizon (10, 20, 50, 100) |
| `--lr` | 0.0001 | Learning rate |
| `--data-fraction` | 1.0 | Fraction of train/val data (0–1) |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |

All arguments (including preprocess, train, eval, quick-eval, backtest): [CLI.md](CLI.md). Override defaults via `params.toml` or run `python main.py <command> --help`.

## Model details

| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **TLOB** | Transformer | Alternating spatial and temporal attention |
| **TLOB-Decay** | Transformer | TLOB + learnable decay attention |
| **LiT** | Transformer | Lightweight with CLS token (simplified inspired version) |
| **DeepLOB** | CNN-GRU | Convolutional + recurrent baseline (simplified inspired version) |
| **Majority** | Baseline | Always predicts majority class from training labels |

Implementations of LiT and DeepLOB are simplified, inspired versions of the originals; TLOB follows the paper more closely.

## References

- **TLOB:** Berti, L., & Kasneci, G. (2025). TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data. *arXiv*. [https://arxiv.org/abs/2502.15757](https://arxiv.org/abs/2502.15757)
- **LiT:** Xiao, Y., Ventre, C., Wang, Y., Li, H., Huan, Y., & Liu, B. (2025). LiT: limit order book transformer. *Frontiers in Artificial Intelligence*. [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full)
- **DeepLOB:** Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Trans. Signal Process.* [https://arxiv.org/abs/1808.03668](https://arxiv.org/abs/1808.03668)

## License

MIT - See `LICENSE`
