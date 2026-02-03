# CLI Reference

Full list of commands and arguments. Defaults can be overridden via `params.toml`. See also: `python main.py <command> --help`.

## preprocess

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker |
| `--seed` | 42 | Random seed |

## train

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker |
| `--epochs` | from config | Training epochs |
| `--batch-size` | from config | Batch size |
| `--horizon` | 10 | Prediction horizon (10, 20, 50, 100) |
| `--data-fraction` | 1.0 | Fraction of train/val data (0–1) |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |
| `--num-workers` | 4 | DataLoader workers |
| `--seed` | 1 | Random seed |

## run

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker |
| `--model` | TLOB | TLOB, DeepLOB, LiT, or Majority |
| `--decay` | False | Use decay variant (TLOB only) |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--patience` | 5 | Early stopping patience |
| `--horizon` | 10 | Prediction horizon (10, 20, 50, 100) |
| `--data-fraction` | 1.0 | Fraction of train/val data (0–1) |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |
| `--seed` | 42 | Random seed |
| `--lit-use-bin` | True | LiT: BiN input normalization |
| `--no-lit-use-bin` | — | LiT: disable BiN |
| `--lit-use-event-embed` | False | LiT: embed event_type column |
| `--lit-use-mean-pool` | False | LiT: CLS + mean-pool aggregation |
| `--lit-use-lr-schedule` | False | LiT: halve LR on no val improvement |
| `--init-decay` | 0.1 | TLOB-Decay: initial decay rate |
| `--decay-lr-multiplier` | 10.0 | TLOB-Decay: LR multiplier for decay params |
| `--weight-decay` | 0.0 | Adam weight decay |
| `--label-smoothing` | 0.0 | CrossEntropy label smoothing |
| `--use-f1-for-best` | False | Pick best checkpoint by val F1 (not val loss) |

## eval

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-dir` | Required | Path to run directory (results.json; optional *_best.pt) |
| `--model-path` | None | Path to .pt file if not in run-dir |
| `--data-fraction` | 1.0 | Fraction of train/val data (0–1) |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |
| `--batch-size` | from config | Batch size |
| `--seed` | 42 | Random seed |

## quick-eval

Test-set only evaluation: writes `test_accuracy` and `test_macro_f1` to `results-quick_eval.json` in the run directory (no overwrite of existing eval/backtest outputs).

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-dir` | Required | Path to run directory (results.json + *_best.pt) |
| `--model-path` | None | Path to .pt file if not in run-dir |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |
| `--batch-size` | from config | Batch size |
| `--seed` | 42 | Random seed |

## backtest

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-dir` | Required | Path to run directory (results.json + *_best.pt) |
| `--model-path` | None | Path to .pt file if not in run-dir |
| `--data-fraction` | 1.0 | Fraction of val data for temperature scaling (0–1) |
| `--test-fraction` | 1.0 | Fraction of test data (0–1) |
| `--batch-size` | from config | Batch size |
| `--seed` | 42 | Random seed |
