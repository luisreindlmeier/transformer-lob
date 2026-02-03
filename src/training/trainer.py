import os
from typing import Tuple
import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from torch_ema import ExponentialMovingAverage
from sklearn.metrics import classification_report
from src import config as cfg
from src.models import TLOB


class TLOBTrainer(LightningModule):
    def __init__(self, seq_size: int = cfg.SEQ_SIZE, hidden_dim: int = cfg.HIDDEN_DIM,
                 num_layers: int = cfg.NUM_LAYERS, num_heads: int = cfg.NUM_HEADS,
                 num_features: int = 46, lr: float = cfg.LR, max_epochs: int = cfg.MAX_EPOCHS,
                 horizon: int = 10, experiment_type: str = "TRAINING", ticker: str = "CSCO") -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TLOB(hidden_dim=hidden_dim, num_layers=num_layers, seq_size=seq_size,
                         num_features=num_features, num_heads=num_heads)
        
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cfg.DEVICE)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = lr
        self.experiment_type = experiment_type
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.val_targets = []
        self.val_predictions = []
        self.test_targets = []
        self.test_predictions = []
        self.min_loss = np.inf
        self.last_ckpt_path = None
        self.horizon = horizon
        self.ticker = ticker
        self.dir_ckpt = f"{ticker}_seq{seq_size}_h{horizon}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.train_losses.append(loss.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f"train loss: {sum(self.train_losses) / len(self.train_losses):.4f}")
        return loss

    def on_train_epoch_start(self) -> None:
        print(f"learning rate: {self.optimizers().param_groups[0]['lr']}")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        with self.ema.average_parameters():
            y_hat = self(x)
            loss = self.loss_function(y_hat, y)
            self.val_losses.append(loss.item())
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
        return loss

    def on_validation_epoch_start(self) -> None:
        if self.train_losses:
            print(f"Train loss epoch {self.current_epoch}: {sum(self.train_losses) / len(self.train_losses):.4f}")
            self.train_losses = []

    def on_validation_epoch_end(self) -> None:
        val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        
        optimizer = self.optimizers()
        if val_loss < self.min_loss:
            improvement = self.min_loss - val_loss
            if improvement < 0.002:
                optimizer.param_groups[0]["lr"] /= 2
                print(f"  Small improvement ({improvement:.4f}), halving LR")
            self.min_loss = val_loss
            self._save_checkpoint(val_loss)
        else:
            optimizer.param_groups[0]["lr"] /= 2
            print(f"  No improvement, halving LR")
        
        self.log("val_loss", val_loss)
        print(f"Validation loss epoch {self.current_epoch}: {val_loss:.4f}")
        
        targets = np.concatenate(self.val_targets)
        predictions = np.concatenate(self.val_predictions)
        report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        
        self.log("val_f1", report["macro avg"]["f1-score"])
        self.log("val_acc", report["accuracy"])
        self.val_targets = []
        self.val_predictions = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        if self.experiment_type == "TRAINING":
            with self.ema.average_parameters():
                y_hat = self(x)
                loss = self.loss_function(y_hat, y)
                self.test_losses.append(loss.item())
                self.test_targets.append(y.cpu().numpy())
                self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
        else:
            y_hat = self(x)
            loss = self.loss_function(y_hat, y)
            self.test_losses.append(loss.item())
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
        return loss

    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)
        predictions = np.concatenate(self.test_predictions)
        
        pred_dir = os.path.join(cfg.CHECKPOINT_DIR, self.dir_ckpt)
        os.makedirs(pred_dir, exist_ok=True)
        np.save(os.path.join(pred_dir, "predictions.npy"), predictions)
        
        test_loss = sum(self.test_losses) / len(self.test_losses)
        print(f"\nTest loss: {test_loss:.4f}")
        report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        
        self.log("test_loss", test_loss)
        self.log("f1_score", report["macro avg"]["f1-score"])
        self.log("accuracy", report["accuracy"])
        self.log("precision", report["macro avg"]["precision"])
        self.log("recall", report["macro avg"]["recall"])
        
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-8)

    def _save_checkpoint(self, val_loss: float) -> None:
        ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, self.dir_ckpt)
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.last_ckpt_path and os.path.exists(self.last_ckpt_path):
            os.remove(self.last_ckpt_path)
        filename = f"val_loss={val_loss:.3f}_epoch={self.current_epoch}.pt"
        path = os.path.join(ckpt_dir, filename)
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path)
        self.last_ckpt_path = path
        print(f"  Saved checkpoint: {filename}")
