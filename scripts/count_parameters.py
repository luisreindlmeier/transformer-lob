#!/usr/bin/env python3
"""Print trainable parameter counts for all models (TLOB, TLOB-Decay, DeepLOB, LiT, Majority)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEQ_SIZE, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS
from src.models import TLOB, TLOBDecay, DeepLOB, LiTTransformer, MajorityBaseline


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    n_features = 46
    seq_size = SEQ_SIZE

    tlob = TLOB(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        seq_size=seq_size,
        num_features=n_features,
        num_heads=NUM_HEADS,
    )
    tlob_decay = TLOBDecay(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        seq_size=seq_size,
        num_features=n_features,
        num_heads=NUM_HEADS,
        init_decay=0.1,
    )
    deeplob = DeepLOB(n_features=n_features)
    lit = LiTTransformer(
        n_features=n_features,
        window=seq_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        use_bin=True,
        use_event_embed=False,
        use_mean_pool=False,
        dropout=0.25,
    )
    majority = MajorityBaseline(n_classes=3)

    print("Trainable parameters (n_features=46, seq_size=128):")
    print(f"  TLOB:        {count_params(tlob):>10,}")
    print(f"  TLOB-Decay:  {count_params(tlob_decay):>10,}")
    print(f"  DeepLOB:    {count_params(deeplob):>10,}")
    print(f"  LiT:        {count_params(lit):>10,}")
    print(f"  Majority:   {count_params(majority):>10,}")


if __name__ == "__main__":
    main()
