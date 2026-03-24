from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import MP20
from .transform import ContinuousIntervalLattice, CopyProperty, FullyConnectedGraph, TaskMetadata

TASK_CSP = 0


class CSPTask:
    """Return CSP-ready MatterGen batches."""

    def __init__(self) -> None:
        self.transforms = [
            FullyConnectedGraph(),
            ContinuousIntervalLattice(),
            CopyProperty("atomic_numbers", "h"),
            TaskMetadata(task_id=TASK_CSP, diffuse_h=False),
        ]

    def fit_dataset(self, root: str | Path, split: str = "train", download: bool = False) -> MP20:
        return MP20(root=root, split=split, transforms=self.transforms, download=download)

    def dataloader(
        self,
        root: str | Path,
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
