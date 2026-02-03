"""
Code and data-structure constants. Experiment defaults (epochs, lr, seed, etc.)
live in params.toml and are overridden by CLI flags.
"""
from typing import Final, List
import torch
import os

DEVICE: Final[str] = 'cuda' if torch.cuda.is_available() else 'cpu'

LEN_ORDER: Final[int] = 6
LEN_LEVEL: Final[int] = 4
N_LOB_LEVELS: Final[int] = 10
LEN_SMOOTH: Final[int] = 10
LOBSTER_HORIZONS: Final[List[int]] = [10, 20, 50, 100]

SEQ_SIZE: Final[int] = 128
HIDDEN_DIM: Final[int] = 46
NUM_LAYERS: Final[int] = 4
NUM_HEADS: Final[int] = 1
IS_SIN_EMB: Final[bool] = True
LR: Final[float] = 0.0001
BATCH_SIZE: Final[int] = 128

MAX_EPOCHS: Final[int] = 10
PRECISION: Final[int] = 32

BASE_DIR: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: Final[str] = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
CHECKPOINT_DIR: Final[str] = os.path.join(BASE_DIR, "checkpoints")

SPLIT_RATES: Final[List[float]] = [0.85, 0.05, 0.10]
