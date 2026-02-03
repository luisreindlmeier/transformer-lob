from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from lightning import LightningDataModule
from src import config as cfg


class LOBDataset(TorchDataset):
    def __init__(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                 seq_size: int = cfg.SEQ_SIZE) -> None:
        self.seq_size = seq_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).long()
        max_sequences = len(self.x) - seq_size + 1
        self.length = min(len(self.y), max_sequences)
        self.data = self.x

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i:i + self.seq_size, :].clone(), self.y[i].clone()


class LOBDataModule(LightningDataModule):
    def __init__(self, train_set: LOBDataset, val_set: LOBDataset, test_set: Optional[LOBDataset] = None,
                 batch_size: int = cfg.BATCH_SIZE, num_workers: int = 4) -> None:
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = train_set.data.device.type != cfg.DEVICE

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_set is None:
            return None
        return DataLoader(self.test_set, batch_size=self.batch_size * 4, shuffle=False,
                         pin_memory=self.pin_memory, num_workers=self.num_workers,
                         persistent_workers=self.num_workers > 0)
