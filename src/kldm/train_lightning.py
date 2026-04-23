from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kldm.diffusionModels.TDMdev import TrivialisedDiffusionDev
from kldm.kldm import ModelKLDM
from kldm.lightning_datamodule import CSPDataModule
from kldm.lightning_module import LitKLDM
from kldm.sample_evaluation.sample_evaluation import CSPMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightning runner matching facitKLDM more closely.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--project", type=str, default="kldm-csp-hpc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)

    datamodule = CSPDataModule(
        root=args.root,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        num_val_subset=1024,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdm = TrivialisedDiffusionDev(
        eps=1e-3,
        n_lambdas=512 if device.type == "cuda" else 128,
        lambda_num_batches=32 if device.type == "cuda" else 8,
        n_sigmas=2000 if device.type == "cuda" else 512,
    )
    model = ModelKLDM(device=device, diffusion_v=tdm).to(device)
    datamodule.setup()
    model.tdm.precompute_lambda_v_table_from_loader(
        datamodule.train_dataloader(),
        device=device,
    )
    lit_module = LitKLDM(
        model=model,
        lr=1e-3,
        with_ema=True,
        ema_decay=0.999,
        ema_start=500,
        metrics=CSPMetrics(),
        sampling_kwargs={
            "val": {"force_ema": False, "n_steps": args.sampling_steps},
            "test": {"force_ema": True, "n_steps": args.sampling_steps},
        },
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="artifacts/HPC/checkpoints",
        filename="epoch_{epoch:03d}",
        monitor="val/match_rate",
        save_last=True,
        save_top_k=3,
        mode="max",
        every_n_epochs=args.validate_every,
        save_on_train_epoch_end=True,
    )
    wandb_logger = WandbLogger(project=args.project, log_model=False)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.validate_every,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=[wandb_logger],
    )
    trainer.fit(model=lit_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
