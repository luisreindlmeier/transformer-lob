from typing import Tuple, Union
import numpy as np
import torch
from src import config as cfg


def lobster_load(
    path: str,
    horizon: int = 10,
    seq_size: int = cfg.SEQ_SIZE,
    all_features: bool = True,
    return_midprice: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    data = np.load(path)
    
    horizon_to_col = {10: 4, 20: 3, 50: 2, 100: 1}
    if horizon not in horizon_to_col:
        raise ValueError(f"Unsupported horizon: {horizon}")
    
    col = horizon_to_col[horizon]
    labels = data[seq_size - cfg.LEN_SMOOTH:, -col]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    
    if all_features:
        lob_features = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        order_features = data[:, :cfg.LEN_ORDER]
        lob_tensor = torch.from_numpy(lob_features).float()
        order_tensor = torch.from_numpy(order_features).float()
        inputs = torch.cat((lob_tensor, order_tensor), dim=1)
    else:
        inputs = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        inputs = torch.from_numpy(inputs).float()
    
    if return_midprice:
        lob_features = data[:, cfg.LEN_ORDER:cfg.LEN_ORDER + 40]
        best_ask = lob_features[:, 0]
        best_bid = lob_features[:, 2]
        midprice = (best_ask + best_bid) / 2
        midprice = midprice[seq_size - cfg.LEN_SMOOTH:len(labels) + seq_size - cfg.LEN_SMOOTH]
        return inputs, labels, midprice
    
    return inputs, labels


def compute_returns(midprice: np.ndarray, horizon: int = 10) -> np.ndarray:
    returns = np.zeros(len(midprice))
    for i in range(len(midprice) - horizon):
        if midprice[i] > 0 and midprice[i + horizon] > 0:
            ratio = midprice[i + horizon] / midprice[i]
            if ratio > 0:
                returns[i] = np.log(ratio)
    return returns
