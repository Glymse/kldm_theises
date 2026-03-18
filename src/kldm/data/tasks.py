from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from kldm.data.dataset import CrystalDataset, FormulaDataset, SizePriorDataset, StoredCrystalDataset
from kldm.data.transformations import DEFAULT_ATOMIC_VOCAB, sampling_transform, training_transform


class TaskDataModule(LightningDataModule):
    def __init__(
        self,
        train: StoredCrystalDataset,
        val: Optional[StoredCrystalDataset] = None,
        test: Optional[StoredCrystalDataset] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.datasets = {"train": train, "val": val, "test": test}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _make_loader(self, split: str, shuffle: bool) -> Optional[DataLoader]:
        dataset = self.datasets[split]
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self):
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self):
        return self._make_loader("val", shuffle=False)

    def test_dataloader(self):
        return self._make_loader("test", shuffle=False)


@dataclass(slots=True)
class Task(ABC):
    species_mode: str = "atomic_numbers"
    species_vocab: tuple[int, ...] = DEFAULT_ATOMIC_VOCAB
    length_stats_path: Optional[str | Path] = None

    def _stats_path(self, split_path: str | Path) -> Optional[Path]:
        if self.length_stats_path is not None:
            return Path(self.length_stats_path)
        split_path = Path(split_path)
        candidate = split_path.with_name(f"{split_path.stem}_length_stats.json")
        return candidate if candidate.exists() else None

    def fit_dataset(self, path: str | Path) -> StoredCrystalDataset:
        return StoredCrystalDataset(
            path=path,
            transform=training_transform(
                species_mode=self.species_mode,
                species_vocab=self.species_vocab,
                length_stats_path=self._stats_path(path),
            ),
        )

    def datamodule(
        self,
        train_path: str | Path,
        val_path: Optional[str | Path] = None,
        test_path: Optional[str | Path] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> TaskDataModule:
        return TaskDataModule(
            train=self.fit_dataset(train_path),
            val=None if val_path is None else self.fit_dataset(val_path),
            test=None if test_path is None else self.fit_dataset(test_path),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def sample_loader(
        self,
        *args,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> DataLoader:
        dataset = self.sampling_dataset(*args, **kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
        )

    def _sampling_transform(self):
        return sampling_transform(
            species_mode=self.species_mode,
            species_vocab=self.species_vocab,
        )

    @abstractmethod
    def sampling_dataset(self, *args, **kwargs) -> CrystalDataset:
        raise NotImplementedError


@dataclass(slots=True)
class DNGTask(Task):
    default_atomic_number: int = 6

    def size_prior_from_split(
        self,
        path: str | Path,
        *,
        max_nodes: Optional[int] = None,
    ) -> np.ndarray:
        dataset = StoredCrystalDataset(path)
        counts = np.fromiter((sample.pos.shape[0] for sample in dataset.samples), dtype=int)
        upper = int(counts.max()) if max_nodes is None else int(max_nodes)
        hist = np.bincount(counts, minlength=upper + 1).astype(float)
        return hist / hist.sum()

    def sampling_dataset(
        self,
        size_distribution: np.ndarray,
        *,
        n_samples: int,
        seed: int = 42,
    ) -> SizePriorDataset:
        return SizePriorDataset(
            size_distribution=size_distribution,
            n_samples=n_samples,
            default_atomic_number=self.default_atomic_number,
            seed=seed,
            transform=self._sampling_transform(),
        )


@dataclass(slots=True)
class CSPTask(Task):
    def sampling_dataset(
        self,
        formulas: Sequence[str],
        *,
        repeats: int,
    ) -> FormulaDataset:
        return FormulaDataset(
            formulas=formulas,
            repeats=repeats,
            transform=self._sampling_transform(),
        )
