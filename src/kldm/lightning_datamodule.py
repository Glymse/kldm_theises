from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from kldm.data import CSPTask, resolve_data_root


class CSPDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path | None = None,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        num_val_subset: Optional[int] = 1024,
        test_batch_size: Optional[int] = 256,
        num_test_subset: Optional[int] = 10000,
        num_workers: int = 1,
        pin_memory: bool = True,
        subset_seed: int = 42,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.root = resolve_data_root(root)
        self.task = CSPTask()
        self.save_hyperparameters(logger=False)

        self.train_dataset = self.task.fit_dataset(
            root=self.root,
            split="train",
            download=self.hparams.download,
        )

        val_dataset = self.task.fit_dataset(
            root=self.root,
            split="val",
            download=self.hparams.download,
        )
        if (
            isinstance(self.hparams.num_val_subset, int)
            and self.hparams.num_val_subset > -1
            and self.hparams.num_val_subset < len(val_dataset)
        ):
            val_dataset = self.get_random_subset(
                val_dataset,
                subset_size=self.hparams.num_val_subset,
                seed=self.hparams.subset_seed,
            )
        self.val_dataset = val_dataset

        test_dataset = self.task.fit_dataset(
            root=self.root,
            split="test",
            download=self.hparams.download,
        )
        if (
            isinstance(self.hparams.num_test_subset, int)
            and self.hparams.num_test_subset > -1
            and self.hparams.num_test_subset < len(test_dataset)
        ):
            test_dataset = self.get_random_subset(
                test_dataset,
                subset_size=self.hparams.num_test_subset,
                seed=self.hparams.subset_seed,
            )
        self.test_dataset = test_dataset

    def setup(self, stage: str | None = None) -> None:
        return None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def get_random_subset(dataset, subset_size: int, seed: int):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(subset_size))
        return Subset(dataset, indices=indices)
