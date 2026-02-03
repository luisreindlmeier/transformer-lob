from pathlib import Path
from typing import Optional
import numpy as np
import torch


def extract_decay_rates(state_dict: dict) -> dict:
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    decay_rates = {}
    for key, value in state_dict.items():
        if "lambda_raw" in key:
            parts = key.split(".")
            layer_idx = next((int(parts[i + 1]) for i, part in enumerate(parts) 
                            if part in ("layers", "encoder") and i + 1 < len(parts) and parts[i + 1].isdigit()), 0)
            layer_name = f"Layer {layer_idx + 1}"
            lambda_vals = np.log(1 + np.exp(value.cpu().numpy()))
            if layer_name not in decay_rates:
                decay_rates[layer_name] = {}
            for head_idx, lam in enumerate(lambda_vals):
                decay_rates[layer_name][f"Head {head_idx + 1}"] = float(lam)
    return decay_rates


def analyze_model_decay(model: torch.nn.Module, output_dir: Optional[Path] = None, model_name: str = "Model") -> dict:
    output_dir = output_dir or Path("./results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "get_decay_rates"):
        rates = model.get_decay_rates()
        decay_rates = {key.replace("_", " ").title(): {f"Head {i+1}": float(v) for i, v in enumerate(vals.tolist())}
                       for key, vals in rates.items()}
    else:
        decay_rates = extract_decay_rates(model.state_dict())
    
    if not decay_rates:
        return {"decay_rates": {}, "summary": {}}
    
    all_lambdas = [lam for heads in decay_rates.values() for lam in heads.values()]
    summary = {"min_lambda": float(np.min(all_lambdas)), "max_lambda": float(np.max(all_lambdas)),
               "mean_lambda": float(np.mean(all_lambdas)), "std_lambda": float(np.std(all_lambdas)), "n_heads": len(all_lambdas)}
    return {"decay_rates": decay_rates, "summary": summary}
