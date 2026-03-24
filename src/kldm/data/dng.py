from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import MP20, resolve_data_root
from .transform import ContinuousIntervalLattice, DEFAULT_ATOMIC_VOCAB, CopyProperty, FullyConnectedGraph, OneHot, TaskMetadata

TASK_DNG = 1


class DNGTask:
    """Return DNG-ready MatterGen batches with one-hot atom features."""

    def __init__(self, species_vocab: list[int] | None = None) -> None:
        self.species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB
        self.transforms = [
            FullyConnectedGraph(),
            ContinuousIntervalLattice(),
            CopyProperty("atomic_numbers", "h"),
            OneHot(values=self.species_vocab, key="h"),
            TaskMetadata(task_id=TASK_DNG, diffuse_h=True),
        ]

    def fit_dataset(self, root: str | Path | None = None, split: str = "train", download: bool = False) -> MP20:
        return MP20(root=resolve_data_root(root), split=split, transforms=self.transforms, download=download)

    def dataloader(
        self,
        root: str | Path | None = None,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        download: bool = False,
    ) -> DataLoader:
        dataset = self.fit_dataset(root=root, split=split, download=download)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )
