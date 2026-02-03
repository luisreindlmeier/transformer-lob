#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from tqdm import tqdm

from src import config as cfg
from src.utils import set_seed, banner, get_model_name, suppress_warnings
from src.data import LOBDataset, LOBDataModule, LOBSTERPreprocessor
from src.models import TLOB, TLOBDecay, DeepLOB, LiTTransformer, MajorityBaseline
from src.training import TLOBTrainer, train_model
from src.evaluation import (
    analyze_model_decay, analyze_confidence,
    plot_training_history
)
from src.evaluation.calibration import TemperatureScaler, expected_calibration_error
from src.evaluation.backtest import run_backtest_outputs
from src.evaluation.plotting import plot_reliability_diagram_before_after
from src.cli_helpers import (
    die, load_run_data, resolve_run_and_model, run_evaluation_loop, run_quick_eval, make_train_loader_for_run,
)


def _load_params_toml() -> dict:
    base = Path(cfg.BASE_DIR)
    path = base / "params.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def create_model(model_name: str, decay: bool, n_features: int, seq_size: int,
                 lit_use_bin: bool = True, lit_use_event_embed: bool = False,
                 lit_use_mean_pool: bool = False, init_decay: float = 0.1) -> nn.Module:
    model_upper = model_name.upper().replace("-", "").replace("_", "")
    if model_upper == "TLOB":
        if decay:
            return TLOBDecay(hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NUM_LAYERS, seq_size=seq_size,
                            num_features=n_features, num_heads=cfg.NUM_HEADS, init_decay=init_decay)
        return TLOB(hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NUM_LAYERS, seq_size=seq_size,
                   num_features=n_features, num_heads=cfg.NUM_HEADS)
    elif model_upper == "DEEPLOB":
        if decay:
            raise ValueError("DeepLOB does not support decay variant")
        return DeepLOB(n_features=n_features)
    elif model_upper == "LIT":
        if decay:
            raise ValueError("LiT does not support decay variant")
        kw = dict(use_bin=lit_use_bin, use_event_embed=lit_use_event_embed, use_mean_pool=lit_use_mean_pool, dropout=0.25)
        return LiTTransformer(n_features=n_features, window=seq_size, d_model=256, n_heads=8, n_layers=6, **kw)
    elif model_upper == "MAJORITY":
        return MajorityBaseline(n_classes=3)
    raise ValueError(f"Unknown model: {model_name}")


def cmd_preprocess(args):
    suppress_warnings()
    set_seed(args.seed)
    raw_dir = Path(cfg.DATA_DIR) / "raw" / args.ticker
    output_dir = Path(cfg.DATA_DIR) / "preprocessed" / args.ticker
    if not raw_dir.exists() or not raw_dir.is_dir():
        die(f"Raw data not found: {raw_dir}")
    banner(f"PREPROCESSING {args.ticker}")
    print(f"Raw: {raw_dir} | Output: {output_dir} | Split: {cfg.SPLIT_RATES}\n")
    LOBSTERPreprocessor(raw_data_dir=str(raw_dir), output_dir=str(output_dir)).preprocess()


def cmd_train(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    train_x, train_y, val_x, val_y, test_x, test_y = load_run_data(
        args.ticker, args.horizon, args.data_fraction, args.test_fraction, cfg.SEQ_SIZE, return_midprice=False
    )
    data_dir = Path(cfg.DATA_DIR) / "preprocessed" / args.ticker
    banner(f"TRAINING TLOB - {args.ticker}")
    print(f"Data: {data_dir} | Epochs: {args.epochs} | Horizon: {args.horizon} | Device: {cfg.DEVICE}\n")
    print(f"Train: {train_x.shape} | Val: {val_x.shape} | Test: {test_x.shape}")
    for name, labels in [("Train", train_y), ("Val", val_y), ("Test", test_y)]:
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"{name}: " + " ".join([f"{u.item()}:{(c/len(labels)).item():.1%}" for u, c in zip(unique, counts)]))
    
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    data_module = LOBDataModule(train_set, val_set, test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = TLOBTrainer(num_features=train_x.shape[1], max_epochs=args.epochs, horizon=args.horizon, ticker=args.ticker)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = L.Trainer(
        accelerator="cpu" if cfg.DEVICE == "cpu" else "gpu", precision=cfg.PRECISION, max_epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=False, min_delta=0.002),
                   TQDMProgressBar(refresh_rate=100)],
        num_sanity_val_steps=0, check_val_every_n_epoch=1, enable_progress_bar=True, enable_model_summary=False, logger=False)
    
    print("\nTraining...")
    trainer.fit(model, datamodule=data_module)
    
    print("\nTesting...")
    best_path = model.last_ckpt_path
    if best_path and os.path.exists(best_path):
        best_model = TLOBTrainer.load_from_checkpoint(best_path, map_location=cfg.DEVICE)
    else:
        best_model = model
    best_model.experiment_type = "EVALUATION"
    trainer.test(best_model, data_module.test_dataloader())
    banner("COMPLETE")


def cmd_run(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    
    model_name = get_model_name(args.model, args.decay)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("./results") / f"{args.ticker}_{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    banner("LOB PREDICTION PIPELINE")
    print(f"  Model: {model_name} | Ticker: {args.ticker} | Epochs: {args.epochs} | Device: {cfg.DEVICE}")
    start_time = time.time()
    
    banner("LOADING DATA")
    train_x, train_y, val_x, val_y, test_x, test_y, test_midprice, test_returns = load_run_data(
        args.ticker, args.horizon, args.data_fraction, args.test_fraction, cfg.SEQ_SIZE, return_midprice=True
    )
    print(f"  Train: {train_x.shape[0]:,} | Val: {val_x.shape[0]:,} | Test: {test_x.shape[0]:,}")
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    train_loader, class_weights_tensor = make_train_loader_for_run(args.model, train_set, args.batch_size)
    if class_weights_tensor is not None:
        w = class_weights_tensor
        label = "LiT" if args.model.upper() == "LIT" else "DeepLOB"
        print(f"  {label} class weights (Up/Stat/Down): [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    is_lit = args.model.upper() == "LIT"
    is_deeplob = args.model.upper() == "DEEPLOB"
    banner("CREATING MODEL")
    lit_bin = getattr(args, "lit_use_bin", True)
    lit_embed = getattr(args, "lit_use_event_embed", False)
    lit_pool = getattr(args, "lit_use_mean_pool", False)
    model = create_model(args.model, args.decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool,
                         init_decay=args.init_decay)
    print(f"  {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    if args.model.upper() == "LIT" and (lit_bin or lit_embed or lit_pool):
        print(f"  LiT options: bin={lit_bin}, event_embed={lit_embed}, mean_pool={lit_pool}")

    banner("TRAINING")
    is_majority = args.model.upper() == "MAJORITY"
    if is_majority:
        train_y_np = train_y.numpy().ravel().astype(np.int32)
        majority_class = int(np.argmax(np.bincount(train_y_np, minlength=3)))
        model.set_majority_class(majority_class)
        print(f"  Majority class (train): {majority_class} (0=Up, 1=Stat, 2=Down)")
        history = {"train_loss": [0.0], "val_loss": [0.0], "val_f1": [0.0]}
    else:
        use_lr_schedule = getattr(args, "lit_use_lr_schedule", False)
        if is_lit:
            model, history = train_model(model, train_loader, val_loader, args.epochs, args.lr, patience=args.patience,
                                         use_lr_schedule=use_lr_schedule,
                                         class_weights=class_weights_tensor, use_f1_for_best=True, label_smoothing=0.1,
                                         weight_decay=1e-5)
        elif is_deeplob:
            model, history = train_model(model, train_loader, val_loader, args.epochs, lr=0.001, patience=args.patience,
                                         use_ema=False, use_lr_schedule=False,
                                         class_weights=class_weights_tensor, use_f1_for_best=True, label_smoothing=0.1,
                                         weight_decay=1e-5)
        else:
            decay_lr = getattr(args, "decay_lr_multiplier", 10.0)
            model, history = train_model(model, train_loader, val_loader, args.epochs, args.lr, patience=args.patience,
                                         use_lr_schedule=use_lr_schedule, decay_lr_multiplier=decay_lr,
                                         weight_decay=getattr(args, "weight_decay", 0.0),
                                         label_smoothing=getattr(args, "label_smoothing", 0.0),
                                         use_f1_for_best=getattr(args, "use_f1_for_best", False))
    torch.save(model.state_dict(), run_dir / f"{model_name}_{args.ticker}_best.pt")
    if not is_majority:
        plot_training_history(history, model_name, run_dir / "training_history.png")
    
    banner("EVALUATION")
    results = run_evaluation_loop(
        model, train_loader, val_loader, test_loader, run_dir, model_name, cfg.DEVICE, ticker=args.ticker
    )
    if args.decay:
        banner("DECAY ANALYSIS")
        decay_results = analyze_model_decay(model, analysis_dir, model_name)
        results["decay_analysis"] = decay_results
        if decay_results.get("decay_rates"):
            for layer, heads in decay_results["decay_rates"].items():
                for head, lam in heads.items():
                    print(f"  {layer} {head}: λ = {lam:.4f}")
    
    banner("CONFIDENCE ANALYSIS")
    try:
        n_test = len(test_set)
        y_ret = test_returns[:n_test] if len(test_returns) >= n_test else None
        confidence_results = analyze_confidence(model, val_loader, test_loader, y_ret, analysis_dir, model_name)
        results["confidence_analysis"] = confidence_results
    except Exception as e:
        print(f"  [WARN] Confidence analysis failed: {e}")
    
    banner("SAVING RESULTS")
    run_results = {
        "model": model_name,
        "ticker": args.ticker,
        "test_accuracy": results.get("test_accuracy"),
        "test_macro_f1": results.get("test_macro_f1"),
        "history": history,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(run_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    banner("COMPLETE")
    print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f} | Test F1: {results.get('test_macro_f1', 0):.4f}")
    print(f"  Time: {(time.time() - start_time)/60:.1f} min | Output: {run_dir}")
    return results


def cmd_eval(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    info = resolve_run_and_model(args.run_dir, getattr(args, "model_path", None))
    run_dir, run_results, model_path = info.run_dir, info.run_results, info.model_path
    model_name, ticker, model_type, decay, horizon = (
        info.model_name, info.ticker, info.model_type, info.decay, info.horizon
    )

    banner("EVAL (no training)")
    print(f"  Run dir: {run_dir} | Model: {model_name} | Ticker: {ticker} | Checkpoint: {model_path.name}")
    start_time = time.time()
    banner("LOADING DATA")
    data_fraction = getattr(args, "data_fraction", 1.0)
    test_fraction = getattr(args, "test_fraction", 1.0)
    train_x, train_y, val_x, val_y, test_x, test_y, test_midprice, test_returns = load_run_data(
        ticker, horizon, data_fraction, test_fraction, cfg.SEQ_SIZE, return_midprice=True
    )
    print(f"  Train: {train_x.shape[0]:,} | Val: {val_x.shape[0]:,} | Test: {test_x.shape[0]:,}")
    batch_size = getattr(args, "batch_size", cfg.BATCH_SIZE)
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    banner("LOADING MODEL")
    lit_bin = run_results.get("lit_use_bin", True)
    lit_embed = run_results.get("lit_use_event_embed", False)
    lit_pool = run_results.get("lit_use_mean_pool", False)
    model = create_model(model_type, decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    print(f"  Loaded: {model_path}")
    banner("EVALUATION")
    results = run_evaluation_loop(
        model, train_loader, val_loader, test_loader, run_dir, model_name, cfg.DEVICE, ticker=ticker
    )
    results["eval_from"] = str(run_dir)
    analysis_dir = run_dir / "analysis"
    if decay:
        banner("DECAY ANALYSIS")
        decay_results = analyze_model_decay(model, analysis_dir, model_name)
        results["decay_analysis"] = decay_results
        if decay_results.get("decay_rates"):
            for layer, heads in decay_results["decay_rates"].items():
                for head, lam in heads.items():
                    print(f"  {layer} {head}: λ = {lam:.4f}")

    banner("CONFIDENCE ANALYSIS")
    try:
        n_test = len(test_set)
        y_ret = test_returns[:n_test] if len(test_returns) >= n_test else None
        confidence_results = analyze_confidence(model, val_loader, test_loader, y_ret, analysis_dir, model_name)
        results["confidence_analysis"] = confidence_results
    except Exception as e:
        print(f"  [WARN] Confidence analysis failed: {e}")

    results["elapsed_minutes"] = (time.time() - start_time) / 60
    eval_json = run_dir / "eval_results.json"
    with open(eval_json, "w") as f:
        json.dump({k: v for k, v in results.items() if k not in ("test_labels", "test_preds", "test_probs")},
                  f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

    banner("COMPLETE")
    print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f} | Test F1: {results.get('test_macro_f1', 0):.4f}")
    print(f"  Time: {results['elapsed_minutes']:.1f} min | Output: {run_dir}")
    return results


def cmd_quick_eval(args):
    """Evaluate only on test set; write model, ticker, test_accuracy, test_macro_f1 to results-quick_eval.json."""
    suppress_warnings()
    set_seed(getattr(args, "seed", 42))
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    info = resolve_run_and_model(args.run_dir, getattr(args, "model_path", None))
    run_dir, run_results, model_path = info.run_dir, info.run_results, info.model_path
    model_name, ticker, model_type, decay, horizon = (
        info.model_name, info.ticker, info.model_type, info.decay, info.horizon
    )
    from_pt_file = getattr(info, "from_pt_file", False)

    data_fraction = getattr(args, "data_fraction", 1.0)
    test_fraction = getattr(args, "test_fraction", 1.0)
    train_x, train_y, val_x, val_y, test_x, test_y = load_run_data(
        ticker, horizon, data_fraction, test_fraction, cfg.SEQ_SIZE, return_midprice=False
    )
    batch_size = getattr(args, "batch_size", cfg.BATCH_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    lit_bin = run_results.get("lit_use_bin", True)
    lit_embed = run_results.get("lit_use_event_embed", False)
    lit_pool = run_results.get("lit_use_mean_pool", False)
    init_decay = run_results.get("init_decay", 0.1)
    model = create_model(model_type, decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool,
                         init_decay=init_decay)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    results = run_quick_eval(model, test_loader, cfg.DEVICE)
    out = {
        "model": model_name,
        "ticker": ticker,
        "test_accuracy": results["test_accuracy"],
        "test_macro_f1": results["test_macro_f1"],
    }
    out_path = run_dir / (f"results-quick_eval_{ticker}.json" if from_pt_file else "results-quick_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Test accuracy: {out['test_accuracy']:.4f} | Test macro F1: {out['test_macro_f1']:.4f}")
    print(f"  Saved: {out_path}")
    return out


def cmd_backtest(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    info = resolve_run_and_model(args.run_dir, getattr(args, "model_path", None))
    run_dir, run_results, model_path = info.run_dir, info.run_results, info.model_path
    model_name, ticker, model_type, decay, horizon = (
        info.model_name, info.ticker, info.model_type, info.decay, info.horizon
    )
    out_dir = run_dir / "backtesting"
    out_dir.mkdir(parents=True, exist_ok=True)
    banner("BACKTEST")
    print(f"  Run dir: {run_dir} | Model: {model_name} | Ticker: {ticker}")
    print(f"  Output: {out_dir}")
    start_time = time.time()
    data_fraction = getattr(args, "data_fraction", 1.0)
    test_fraction = getattr(args, "test_fraction", 1.0)
    train_x, train_y, val_x, val_y, test_x, test_y, test_midprice, test_returns = load_run_data(
        ticker, horizon, data_fraction, test_fraction, cfg.SEQ_SIZE, return_midprice=True
    )
    batch_size = getattr(args, "batch_size", cfg.BATCH_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    lit_bin = run_results.get("lit_use_bin", True)
    lit_embed = run_results.get("lit_use_event_embed", False)
    lit_pool = run_results.get("lit_use_mean_pool", False)
    init_decay = run_results.get("init_decay", 0.1)
    model = create_model(model_type, decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool,
                         init_decay=init_decay)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(cfg.DEVICE)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Backtest inference", unit="batch", leave=True, mininterval=0.5):
            all_logits.append(model(x.to(cfg.DEVICE)).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_labels, dim=0).numpy()
    n_test = len(test_set)
    y_ret = test_returns[:n_test] if len(test_returns) >= n_test else test_returns

    # Before temperature scaling
    y_prob_before = torch.softmax(all_logits, dim=1).numpy()
    y_pred_before = np.argmax(y_prob_before, axis=1)
    results_before = run_backtest_outputs(
        y_true, y_pred_before, y_prob_before, y_ret,
        output_dir=out_dir,
        model_name=model_name,
        write_plots=False,
    )

    # Save pre-scaling bin data for separate pre-scaling plots (ECE + confidence distribution)
    n_bins = 10
    _, bin_acc, _, bin_counts = expected_calibration_error(y_true, y_prob_before, n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pre_scaling_df = pd.DataFrame({
        "bin_center": bin_centers,
        "bin_accuracy": bin_acc,
        "bin_count": bin_counts,
    })
    pre_scaling_df.to_csv(out_dir / "pre_scaling_bins.csv", index=False)

    # Temperature scaling
    scaler = TemperatureScaler(model, cfg.DEVICE)
    optimal_temp = scaler.set_temperature(val_loader, use_tqdm=True)
    y_prob_after = torch.softmax(all_logits / optimal_temp, dim=1).numpy()
    y_pred_after = np.argmax(y_prob_after, axis=1)
    results_after = run_backtest_outputs(
        y_true, y_pred_after, y_prob_after, y_ret,
        output_dir=out_dir,
        model_name=model_name,
        write_plots=True,
    )

    # Save post-scaling bin data for ECE + confidence distribution plots
    _, bin_acc_after, _, bin_counts_after = expected_calibration_error(y_true, y_prob_after, n_bins)
    post_scaling_df = pd.DataFrame({
        "bin_center": bin_centers,
        "bin_accuracy": bin_acc_after,
        "bin_count": bin_counts_after,
    })
    post_scaling_df.to_csv(out_dir / "post_scaling_bins.csv", index=False)

    # Reliability diagram: before vs after
    plot_reliability_diagram_before_after(
        y_true, y_prob_before, y_prob_after, model_name,
        out_dir / "reliability_diagram_before_after.png",
        n_bins=10,
    )

    # One comprehensive CSV: scalar metrics + strategy comparison (before/after)
    scalar_rows = [
        {"metric": "ECE", "before_scaling": results_before["ece"], "after_scaling": results_after["ece"]},
        {"metric": "brier_score", "before_scaling": results_before["brier_score"], "after_scaling": results_after["brier_score"]},
        {"metric": "sharpe_ratio", "before_scaling": results_before["sharpe_ratio"], "after_scaling": results_after["sharpe_ratio"]},
        {"metric": "max_drawdown", "before_scaling": results_before["max_drawdown"], "after_scaling": results_after["max_drawdown"]},
        {"metric": "hit_rate", "before_scaling": results_before["hit_rate"], "after_scaling": results_after["hit_rate"]},
        {"metric": "profit_factor", "before_scaling": results_before["profit_factor"], "after_scaling": results_after["profit_factor"]},
        {"metric": "n_samples", "before_scaling": results_before["n_samples"], "after_scaling": results_after["n_samples"]},
    ]
    comp_before = {r["strategy"]: r for r in results_before["confidence_strategy_comparison"]}
    comp_after = {r["strategy"]: r for r in results_after["confidence_strategy_comparison"]}
    strategy_rows = []
    for name in comp_after:
        b = comp_before.get(name, {})
        a = comp_after[name]
        strategy_rows.append({
            "strategy": name,
            "sharpe_ratio_before": b.get("sharpe_ratio"), "sharpe_ratio_after": a.get("sharpe_ratio"),
            "max_drawdown_before": b.get("max_drawdown"), "max_drawdown_after": a.get("max_drawdown"),
            "hit_rate_before": b.get("hit_rate"), "hit_rate_after": a.get("hit_rate"),
            "profit_factor_before": b.get("profit_factor"), "profit_factor_after": a.get("profit_factor"),
        })
    df_scalar = pd.DataFrame(scalar_rows)
    df_strategy = pd.DataFrame(strategy_rows)
    df_scalar.to_csv(out_dir / "backtest_results.csv", index=False)
    with open(out_dir / "backtest_results.csv", "a") as f:
        f.write("\n")
    df_strategy.to_csv(out_dir / "backtest_results.csv", mode="a", index=False)

    elapsed_min = (time.time() - start_time) / 60
    banner("COMPLETE")
    print(f"  Sharpe (after): {results_after['sharpe_ratio']:.4f} | MDD: {results_after['max_drawdown']:.4f} | Hit rate: {results_after['hit_rate']:.2%} | PF: {results_after['profit_factor']:.4f}")
    print(f"  ECE before/after: {results_before['ece']:.4f} / {results_after['ece']:.4f}")
    print(f"  Time: {elapsed_min:.1f} min | Output: {out_dir}")
    return results_after


def main():
    params = _load_params_toml()
    p = params.get("preprocess", {})
    t = params.get("train", {})
    r = params.get("run", {})
    e = params.get("eval", {})
    b = params.get("backtest", {})

    parser = argparse.ArgumentParser(description="LOB Prediction Pipeline", prog="src")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess LOBSTER data")
    preprocess_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    preprocess_parser.add_argument("--seed", type=int, default=p.get("seed", 42))

    train_parser = subparsers.add_parser("train", help="Train TLOB with Lightning")
    train_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    train_parser.add_argument("--epochs", type=int, default=t.get("epochs", cfg.MAX_EPOCHS))
    train_parser.add_argument("--batch-size", type=int, default=t.get("batch_size", cfg.BATCH_SIZE))
    train_parser.add_argument("--horizon", type=int, default=t.get("horizon", 10), choices=[10, 20, 50, 100])
    train_parser.add_argument("--data-fraction", type=float, default=t.get("data_fraction", 1.0), help="Fraction of train/val data to use (0–1)")
    train_parser.add_argument("--test-fraction", type=float, default=t.get("test_fraction", 1.0), help="Fraction of test data to use (0–1)")
    train_parser.add_argument("--num-workers", type=int, default=t.get("num_workers", 4))
    train_parser.add_argument("--seed", type=int, default=t.get("seed", 1))

    run_parser = subparsers.add_parser("run", help="Full pipeline with analysis")
    run_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    run_parser.add_argument("--model", type=str, default=r.get("model", "TLOB"), choices=["TLOB", "DeepLOB", "LiT", "Majority"])
    run_parser.add_argument("--decay", action="store_true", default=r.get("decay", False), help="Use decay variant")
    run_parser.add_argument("--epochs", type=int, default=r.get("epochs", 10))
    run_parser.add_argument("--batch-size", type=int, default=r.get("batch_size", 128))
    run_parser.add_argument("--lr", type=float, default=r.get("lr", 0.0001))
    run_parser.add_argument("--patience", type=int, default=r.get("patience", 5))
    run_parser.add_argument("--horizon", type=int, default=r.get("horizon", 10), choices=[10, 20, 50, 100])
    run_parser.add_argument("--data-fraction", type=float, default=r.get("data_fraction", 1.0), help="Fraction of train/val data to use (0–1)")
    run_parser.add_argument("--test-fraction", type=float, default=r.get("test_fraction", 1.0), help="Fraction of test data to use (0–1)")
    run_parser.add_argument("--seed", type=int, default=r.get("seed", 42))
    run_parser.add_argument("--lit-use-bin", action="store_true", default=r.get("lit_use_bin", True), dest="lit_use_bin", help="LiT: use BiN input normalization (default: True)")
    run_parser.add_argument("--no-lit-use-bin", action="store_false", dest="lit_use_bin", help="LiT: disable BiN")
    run_parser.add_argument("--lit-use-event-embed", action="store_true", default=r.get("lit_use_event_embed", False), help="LiT: embed event_type column (step 2)")
    run_parser.add_argument("--lit-use-mean-pool", action="store_true", default=r.get("lit_use_mean_pool", False), help="LiT: CLS + mean-pool aggregation (step 4)")
    run_parser.add_argument("--lit-use-lr-schedule", action="store_true", default=r.get("lit_use_lr_schedule", False), help="LiT: halve LR on no val improvement (step 3)")
    run_parser.add_argument("--init-decay", type=float, default=r.get("init_decay", 0.1), help="Initial decay rate for decay models (default: 0.1)")
    run_parser.add_argument("--decay-lr-multiplier", type=float, default=r.get("decay_lr_multiplier", 10.0), help="LR multiplier for decay params (TLOB-Decay); base_lr * this (default: 10)")
    run_parser.add_argument("--weight-decay", type=float, default=r.get("weight_decay", 0.0), help="Adam weight decay (TLOB/TLOB-Decay; default: 0)")
    run_parser.add_argument("--label-smoothing", type=float, default=r.get("label_smoothing", 0.0), help="CrossEntropy label smoothing (TLOB/TLOB-Decay; default: 0)")
    run_parser.add_argument("--use-f1-for-best", action="store_true", default=r.get("use_f1_for_best", False), help="TLOB/TLOB-Decay: pick best checkpoint by val F1 instead of val loss")

    eval_parser = subparsers.add_parser("eval", help="Eval + confidence + backtesting from saved best.pt (no training)")
    eval_parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (contains results.json; optional *_best.pt)")
    eval_parser.add_argument("--model-path", type=str, default=None, help="Path to .pt file if not in run-dir (e.g. from another run)")
    eval_parser.add_argument("--data-fraction", type=float, default=e.get("data_fraction", 1.0), help="Fraction of train/val data (0–1); use e.g. 0.01 for quick check")
    eval_parser.add_argument("--test-fraction", type=float, default=e.get("test_fraction", 1.0), help="Fraction of test data (0–1)")
    eval_parser.add_argument("--batch-size", type=int, default=e.get("batch_size", cfg.BATCH_SIZE))
    eval_parser.add_argument("--seed", type=int, default=e.get("seed", 42))

    quick_eval_parser = subparsers.add_parser("quick-eval", help="Test-set only: accuracy + macro F1 → results-quick_eval.json (no overwrite of existing files)")
    quick_eval_parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (results.json + *_best.pt)")
    quick_eval_parser.add_argument("--model-path", type=str, default=None, help="Path to .pt if not in run-dir")
    quick_eval_parser.add_argument("--test-fraction", type=float, default=e.get("test_fraction", 1.0), help="Fraction of test data (0–1)")
    quick_eval_parser.add_argument("--batch-size", type=int, default=e.get("batch_size", cfg.BATCH_SIZE))
    quick_eval_parser.add_argument("--seed", type=int, default=e.get("seed", 42))

    backtest_parser = subparsers.add_parser("backtest", help="Backtest trained model only: PnL, Sharpe, MDD, hit rate, PF, confidence plots, trade table")
    backtest_parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (results.json + *_best.pt)")
    backtest_parser.add_argument("--model-path", type=str, default=None, help="Path to .pt if not in run-dir")
    backtest_parser.add_argument("--data-fraction", type=float, default=b.get("data_fraction", 1.0), help="Fraction of val data for temperature scaling (0–1)")
    backtest_parser.add_argument("--test-fraction", type=float, default=b.get("test_fraction", 1.0), help="Fraction of test data (0–1)")
    backtest_parser.add_argument("--batch-size", type=int, default=b.get("batch_size", cfg.BATCH_SIZE))
    backtest_parser.add_argument("--seed", type=int, default=b.get("seed", 42))

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        if args.model == "DeepLOB" and args.decay:
            parser.error("DeepLOB does not support --decay")
        if args.model == "Majority" and args.decay:
            parser.error("Majority does not support --decay")
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "quick-eval":
        cmd_quick_eval(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
