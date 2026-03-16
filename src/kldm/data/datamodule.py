from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch_geometric.transforms as T
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from kldm.data.dataset import Dataset, SampleDatasetCSP


class DataModule:
    def __init__(
        self,
        transform: Optional[T.BaseTransform],
        train_path: str | Path,
        val_path: str | Path,
        train_batch_size: int,
        val_batch_size: int,
        num_val_subset: Optional[int] = -1,
        test_path: Optional[str | Path] = None,
        test_batch_size: Optional[int] = None,
        num_test_subset: Optional[int] = 10000,
        num_workers: int = 0,
        pin_memory: bool = False,
        subset_seed: int = 42,
    ) -> None:
        del pin_memory
        self.train_dataset = Dataset(path=str(train_path), transform=transform)

        val_dataset = Dataset(path=str(val_path), transform=transform)
        if (
            isinstance(num_val_subset, int)
            and num_val_subset > -1
            and num_val_subset < len(val_dataset)
        ):
            val_dataset = self.get_random_subset(val_dataset, num_val_subset, seed=subset_seed)
        self.val_dataset = val_dataset

        if test_path:
            test_dataset = Dataset(path=str(test_path), transform=transform)
            if (
                isinstance(num_test_subset, int)
                and num_test_subset > -1
                and num_test_subset < len(test_dataset)
            ):
                test_dataset = self.get_random_subset(test_dataset, num_test_subset, seed=subset_seed)
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        batch_size = self.test_batch_size or self.val_batch_size
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def get_random_subset(dataset, subset_size, seed):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(subset_size))
        return Subset(dataset, indices=indices)


DeNovoDataModule = DataModule


class CSPDataModule:
    def __init__(
        self,
        formulas: Sequence[str],
        batch_size: int,
        n_samples_per_formula: int = 5,
        transform: Optional[T.BaseTransform] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        del pin_memory
        self.dataset = SampleDatasetCSP(
            formulas=formulas,
            n_samples_per_formula=n_samples_per_formula,
            transform=transform,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self.predict_dataloader()
