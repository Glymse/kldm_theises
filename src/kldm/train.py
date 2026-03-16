from __future__ import annotations

from pathlib import Path

import typer

from kldm.data import CSPDataModule, DeNovoDataModule
from kldm.model import Model

#uv run train.py --mode csp

def _get_batch(mode: str):
    """Load one batch from either de-novo or CSP data pipelines."""
    normalized_mode = mode.lower().strip()

    if normalized_mode in {"denovo", "dng"}:
        datamodule = DeNovoDataModule(
            transform=None,
            train_path=Path("data/mp_20/train.pt"),
            val_path=Path("data/mp_20/val.pt"),
            test_path=Path("data/mp_20/test.pt"),
            train_batch_size=32,
            val_batch_size=64,
            test_batch_size=64,
            num_workers=0,
            pin_memory=False,
        )
        return next(iter(datamodule.train_dataloader()))

    if normalized_mode == "csp":
        datamodule = CSPDataModule(
            formulas=["SiO2", "LiFePO4", "NaCl"],
            batch_size=8,
            n_samples_per_formula=4,
            transform=None,
            num_workers=0,
            pin_memory=False,
        )
        return next(iter(datamodule.predict_dataloader()))

    raise ValueError("mode must be one of: denovo, dng, csp")


def train(mode: str = "denovo") -> None:
    """Simple training scaffold with KLDM-like target creation."""
    batch = _get_batch(mode=mode)
    model = Model()
    targets = model.training_targets(
        initial_sample=batch,
        task=mode,
        timestep=None,
    )
    loss = model.loss_from_targets(targets)

    print(f"Loaded mode: {mode}")
    print(f"num_graphs: {batch.num_graphs}")
    print(f"pos shape: {tuple(batch.pos.shape)}")
    print(f"h shape: {tuple(batch.h.shape)}")
    print(f"model: {model.__class__.__name__}")
    print(f"x_t shape: {tuple(targets['x_t'].shape)}")
    print(f"noise shape: {tuple(targets['noise'].shape)}")
    print(f"pred_noise shape: {tuple(targets['pred_noise'].shape)}")
    print(f"loss: {float(loss.item()):.6f}")


if __name__ == "__main__":
    typer.run(train)
