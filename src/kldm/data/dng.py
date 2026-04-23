from __future__ import annotations

from pathlib import Path

from torch_geometric.loader import DataLoader

from .dataset import MP20, resolve_data_root
from .transform import (
    DEFAULT_ATOMIC_VOCAB,
    DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH,
    ContinuousIntervalLattice,
    FACIT_ANGLES_LOC_SCALE,
    FullyConnectedGraph,
    MatterGenToFacitFields,
    OneHot,
    TaskMetadata,
    ensure_lengths_loc_scale_cache,
)

TASK_DNG = 1


class DNGTask:
    """Return DNG-ready MatterGen batches with one-hot atom features."""

    def __init__(self, species_vocab: list[int] | None = None) -> None:
        self.species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB

    def _make_transforms(self, root: str | Path | None = None) -> list:
        data_root = resolve_data_root(root)
        cache_file = data_root / "mp_20" / DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH.name
        ensure_lengths_loc_scale_cache(
            cache_file=cache_file,
            processed_dir=data_root / "mp_20" / "processed" / "train",
        )
        return [
            MatterGenToFacitFields(),
            FullyConnectedGraph(),
            ContinuousIntervalLattice(
                cache_file=cache_file,
                standardize=True,
                angles_loc_scale=FACIT_ANGLES_LOC_SCALE,
            ),
            OneHot(values=self.species_vocab, key="h"),
            TaskMetadata(task_id=TASK_DNG, diffuse_h=True),
        ]

    def fit_dataset(self, root: str | Path | None = None, split: str = "train", download: bool = False) -> MP20:
        return MP20(
            root=resolve_data_root(root),
            split=split,
            transforms=self._make_transforms(root=root),
            download=download,
        )

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
        )
