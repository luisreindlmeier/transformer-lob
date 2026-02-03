import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix

from src import config as cfg
from src.data import lobster_load, LOBDataset, compute_returns
from src.evaluation import compute_metrics, print_metrics, plot_confusion_matrix


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def load_run_data(
    ticker: str,
    horizon: int,
    data_fraction: float,
    test_fraction: float,
    seq_size: int,
    return_midprice: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
]:
    data_dir = Path(cfg.DATA_DIR) / "preprocessed" / ticker
    if not data_dir.exists() or not data_dir.is_dir():
        die(f"Data not found: {data_dir}")
    train_x, train_y = lobster_load(str(data_dir / "train.npy"), horizon=horizon)
    val_x, val_y = lobster_load(str(data_dir / "val.npy"), horizon=horizon)
    if return_midprice:
        test_x, test_y, test_midprice = lobster_load(
            str(data_dir / "test.npy"), horizon=horizon, return_midprice=True
        )
        test_returns = compute_returns(np.asarray(test_midprice), horizon=horizon)
    else:
        test_x, test_y = lobster_load(str(data_dir / "test.npy"), horizon=horizon)
        test_midprice = None
        test_returns = None

    if data_fraction < 1.0:
        n_train = max(seq_size, int(len(train_y) * data_fraction))
        n_val = max(1, int(len(val_y) * data_fraction))
        train_max_start = max(0, len(train_y) - n_train)
        val_max_start = max(0, len(val_y) - n_val)
        train_offset = int(np.random.randint(0, train_max_start + 1)) if train_max_start > 0 else 0
        val_offset = int(np.random.randint(0, val_max_start + 1)) if val_max_start > 0 else 0
        train_x = train_x[train_offset : train_offset + n_train]
        train_y = train_y[train_offset : train_offset + n_train]
        val_x = val_x[val_offset : val_offset + n_val]
        val_y = val_y[val_offset : val_offset + n_val]

    if test_fraction < 1.0:
        n_test = int(len(test_y) * test_fraction)
        test_x, test_y = test_x[:n_test], test_y[:n_test]
        if return_midprice and test_midprice is not None:
            test_midprice = test_midprice[:n_test]
            test_returns = test_returns[:n_test]

    if return_midprice:
        return train_x, train_y, val_x, val_y, test_x, test_y, np.asarray(test_midprice), test_returns
    return train_x, train_y, val_x, val_y, test_x, test_y


def _parse_pt_filename(path: Path) -> tuple:
    """From path like results/DeepLOB/DeepLOB_AAPL.pt return (model_name, ticker)."""
    stem = path.stem  # e.g. "DeepLOB_AAPL" or "TLOB-Decay_GOOG"
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        die(f"Cannot parse model and ticker from filename: {path.name} (expected <Model>_<Ticker>.pt)")
    return parts[0], parts[1]


def resolve_run_and_model(run_dir: Union[str, Path], model_path: Optional[Union[str, Path]] = None):
    path = Path(run_dir).resolve()
    from_pt_file = path.is_file() and path.suffix == ".pt"

    if from_pt_file:
        if not path.exists():
            die(f"Model file not found: {path}")
        run_dir = path.parent
        model_path = path
        model_name, ticker = _parse_pt_filename(path)
        model_type = (
            "Majority" if model_name == "Majority"
            else "LiT" if model_name == "LiT"
            else "DeepLOB" if model_name == "DeepLOB"
            else "TLOB"
        )
        decay = "Decay" in model_name or model_name.endswith("-Decay")
        horizon = 10
        run_results = {
            "model": model_name,
            "ticker": ticker,
            "horizon": horizon,
            "args": {"model": model_type, "decay": decay, "horizon": horizon},
            "lit_use_bin": True,
            "lit_use_event_embed": False,
            "lit_use_mean_pool": False,
            "init_decay": 0.1,
        }
        return type(
            "RunModelInfo",
            (),
            {
                "run_dir": run_dir,
                "run_results": run_results,
                "model_path": Path(model_path).resolve(),
                "model_name": model_name,
                "ticker": ticker,
                "model_type": model_type,
                "decay": decay,
                "horizon": horizon,
                "from_pt_file": True,
            },
        )()

    run_dir = path
    if not run_dir.exists() or not run_dir.is_dir():
        die(f"Run directory not found: {run_dir}")
    results_path = run_dir / "results.json"
    if not results_path.exists():
        die(f"No results.json in {run_dir}")
    with open(results_path) as f:
        run_results = json.load(f)
    model_name = run_results.get("model") or run_results.get("model_name", "TLOB")
    ticker = run_results.get("ticker", "CSCO")
    run_args = run_results.get("args", {})
    model_type = run_args.get("model") or (
        "Majority" if "Majority" in model_name else "LiT" if "LiT" in model_name else "DeepLOB" if "DeepLOB" in model_name else "TLOB"
    )
    decay = run_args.get("decay", "Decay" in model_name or model_name.endswith("-Decay"))
    horizon = run_args.get("horizon") or run_results.get("horizon", 10)

    if model_path is not None:
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            die(f"Model file not found: {model_path}")
    else:
        pt_candidates = list(run_dir.glob("*_best.pt"))
        if not pt_candidates:
            die(f"No *_best.pt in {run_dir}. Run 'run' once to save the model.")
        model_path = pt_candidates[0]

    return type(
        "RunModelInfo",
        (),
        {
            "run_dir": run_dir,
            "run_results": run_results,
            "model_path": model_path,
            "model_name": model_name,
            "ticker": ticker,
            "model_type": model_type,
            "decay": decay,
            "horizon": horizon,
            "from_pt_file": False,
        },
    )()


def run_evaluation_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    run_dir: Path,
    model_name: str,
    device: str,
    ticker: Optional[str] = None,
) -> dict:
    model.eval()
    model = model.to(device)
    results = {"model": model_name}
    if ticker is not None:
        results["ticker"] = ticker
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"  Eval {split}", unit="batch", leave=True, mininterval=0.5):
                all_logits.append(model(x.to(device)).cpu())
                all_labels.append(y)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.softmax(all_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        metrics = compute_metrics(all_labels, preds, y_prob=probs)
        for k, v in metrics.items():
            results[f"{split}_{k}"] = v
        cm = confusion_matrix(all_labels, preds, labels=[0, 1, 2])
        plot_confusion_matrix(cm, model_name, split, run_dir / f"confusion_matrix_{split}.png")
        if split == "test":
            results["test_labels"] = all_labels
            results["test_preds"] = preds
            results["test_probs"] = probs

    print_metrics(
        {k.replace("test_", ""): v for k, v in results.items() if k.startswith("test_")}, "test"
    )
    return results


def run_quick_eval(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
) -> dict:
    """Run inference only on test set; return test_accuracy and test_macro_f1."""
    model.eval()
    model = model.to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Eval test", unit="batch", leave=True, mininterval=0.5):
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    metrics = compute_metrics(all_labels, preds, y_prob=probs)
    return {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}


def make_train_loader_for_run(
    model_name: str, train_set: LOBDataset, batch_size: int
) -> Tuple[DataLoader, Optional[torch.Tensor]]:
    model_upper = model_name.upper().replace("-", "").replace("_", "")
    if model_upper == "MAJORITY":
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        return loader, None
    if model_upper in ("LIT", "DEEPLOB"):
        n_train = len(train_set)
        train_y_np = train_set.y.numpy().ravel()
        counts = np.bincount(train_y_np.astype(np.int32), minlength=3)
        counts = np.maximum(counts, 1)
        class_weights_arr = (n_train / (3 * counts)).astype(np.float32)
        class_weights_arr[1] = max(class_weights_arr[1], 1.0)  # stationary (index 1) >= 1.0 for class balance
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
        sample_weights = torch.tensor(
            [1.0 / counts[train_y_np[i]] for i in range(n_train)], dtype=torch.float32
        )
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=n_train)
        bs = min(64, batch_size) if model_upper == "DEEPLOB" else batch_size
        loader = DataLoader(
            train_set, batch_size=bs, sampler=train_sampler, shuffle=False, num_workers=0
        )
        return loader, class_weights_tensor
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, None
