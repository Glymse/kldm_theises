from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

try:
    from lightning import LightningDataModule  # type: ignore
except ImportError:
    try:
        from pytorch_lightning import LightningDataModule  # type: ignore
    except ImportError:
        class LightningDataModule:  # type: ignore[no-redef]
            """Minimal fallback when Lightning is not installed."""

            def save_hyperparameters(self, logger: bool = False) -> None:
                # Keep a small, practical fallback for environments without Lightning.
                del logger
                attrs = {
                    k: v
                    for k, v in self.__dict__.items()
                    if not k.endswith("_dataset") and not k.startswith("_")
                }
                self.hparams = SimpleNamespace(**attrs)

from pymatgen.core import Composition, Element

from kldm.data import MyDataset


class SampleDatasetDNG(torch.utils.data.Dataset):
    """Synthetic de-novo dataset that samples atom counts from a length distribution."""

    def __init__(
        self,
        empirical_distribution: np.ndarray,
        n_samples: int = 10_000,
        transform: Optional[Callable[[Data], Data]] = None,
        seed: int = 42,
    ) -> None:
        distribution = np.asarray(empirical_distribution, dtype=np.float64)
        if distribution.ndim != 1 or distribution.size == 0:
            raise ValueError("empirical_distribution must be a non-empty 1D array")
        if np.any(distribution < 0):
            raise ValueError("empirical_distribution cannot contain negative values")

        total = distribution.sum()
        if total <= 0:
            raise ValueError("empirical_distribution must sum to a positive value")

        self.empirical_distribution = distribution / total
        rng = np.random.RandomState(seed)
        self.num_atoms = rng.choice(
            len(self.empirical_distribution), n_samples, p=self.empirical_distribution
        )
        self.transform = transform

    def __getitem__(self, idx: int):
        n = int(self.num_atoms[idx])
        data = Data(
            pos=torch.randn(n, 3),
            h=torch.full((n,), 6, dtype=torch.long),  # default to carbon
        )
        return data if self.transform is None else self.transform(data)

    def __len__(self) -> int:
        return len(self.num_atoms)

    @classmethod
    def from_len_range(
        cls,
        len_range: str,
        n_samples: int = 10_000,
        transform: Optional[Callable[[Data], Data]] = None,
        seed: int = 42,
    ) -> "SampleDatasetDNG":
        empirical_distribution = cls.uniform_from_range(len_range)
        return cls(
            empirical_distribution=empirical_distribution,
            n_samples=n_samples,
            transform=transform,
            seed=seed,
        )

    @staticmethod
    def uniform_from_range(len_range: str) -> np.ndarray:
        lb_s, ub_s = len_range.split("-", maxsplit=1)
        lb, ub = int(lb_s), int(ub_s)
        if lb < 0 or ub <= lb:
            raise ValueError("len_range must look like 'a-b' with 0 <= a < b")

        empirical_distribution = np.zeros(ub + 1, dtype=np.float64)
        empirical_distribution[lb:ub] = 1.0
        empirical_distribution /= np.sum(empirical_distribution)
        return empirical_distribution


class SampleDatasetCSP(torch.utils.data.Dataset):
    """Synthetic CSP dataset sampling structures from chemical formulas."""

    def __init__(
        self,
        formulas: Sequence[str],
        n_samples_per_formula: int = 5,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        self.transform = transform
        self.parsed_formulas: list[dict[str, float]] = []

        for formula in formulas:
            parsed = Composition(formula).as_dict()
            for _ in range(n_samples_per_formula):
                self.parsed_formulas.append(parsed)

    def __getitem__(self, idx: int):
        formula = self.parsed_formulas[idx]
        h = torch.LongTensor(
            [int(Element(el).Z) for el in formula for _ in range(int(formula[el]))]
        )

        data = Data(pos=torch.randn(len(h), 3), h=h)
        return data if self.transform is None else self.transform(data)

    def __len__(self) -> int:
        return len(self.parsed_formulas)


class DeNovoDataModule(LightningDataModule):
    """DataModule for de-novo training/evaluation on preprocessed MP-20 splits."""

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
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_val_subset = num_val_subset
        self.num_test_subset = num_test_subset
        self.subset_seed = subset_seed

        self.train_dataset = MyDataset(train_path, transform=transform)

        val_dataset = MyDataset(val_path, transform=transform)
        if (
            isinstance(num_val_subset, int)
            and num_val_subset > -1
            and num_val_subset < len(val_dataset)
        ):
            val_dataset = self.get_random_subset(val_dataset, num_val_subset, seed=subset_seed)
        self.val_dataset = val_dataset

        if test_path:
            test_dataset = MyDataset(test_path, transform=transform)
            if (
                isinstance(num_test_subset, int)
                and num_test_subset > -1
                and num_test_subset < len(test_dataset)
            ):
                test_dataset = self.get_random_subset(test_dataset, num_test_subset, seed=subset_seed)
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None

        self.save_hyperparameters(logger=False)

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
        if self.test_dataset is None:
            return None

        batch_size = self.hparams.test_batch_size or self.hparams.val_batch_size
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def get_random_subset(dataset, subset_size: int, seed: int):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(len(dataset)))[:subset_size]
        return Subset(dataset, indices=indices)


class CSPDataModule(LightningDataModule):
    """DataModule for CSP sampling from formulas."""

    def __init__(
        self,
        formulas: Sequence[str],
        batch_size: int,
        n_samples_per_formula: int = 5,
        transform: Optional[T.BaseTransform] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.n_samples_per_formula = n_samples_per_formula
        self.dataset = SampleDatasetCSP(
            formulas=formulas,
            n_samples_per_formula=n_samples_per_formula,
            transform=transform,
        )
        self.save_hyperparameters(logger=False)

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return self.predict_dataloader()


# Backward-compatible alias for existing code that imported DataModule.
DataModule = DeNovoDataModule
