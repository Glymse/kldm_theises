from __future__ import annotations

from pathlib import Path

from torch_geometric.loader import DataLoader

from .dataset import MP20, resolve_data_root
from .transform import (
    DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH,
    DEFAULT_ATOMIC_VOCAB,
    FACIT_ANGLES_LOC_SCALE,
    ConcatFeatures,
    ContinuousIntervalAngles,
    ContinuousIntervalLengths,
    FullyConnectedGraph,
    MatterGenToFacitFields,
    ToPyGData,
    ensure_lengths_loc_scale_cache,
)


class CSPTask:
    """Return CSP-ready MatterGen batches."""

    def __init__(self, species_vocab: list[int] | None = None) -> None:
        self.species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB

    def _cache_file(self, root: str | Path | None = None) -> Path:
        data_root = resolve_data_root(root)
        cache_file = data_root / "mp_20" / DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH.name
        ensure_lengths_loc_scale_cache(
            cache_file=cache_file,
            processed_dir=data_root / "mp_20" / "processed" / "train",
        )
        return cache_file

    def make_lengths_transform(self, root: str | Path | None = None) -> ContinuousIntervalLengths:
        return ContinuousIntervalLengths(
            out_key="lengths",
            lengths_loc_scale=self._cache_file(root=root),
        )

    def make_angles_transform(self) -> ContinuousIntervalAngles:
        return ContinuousIntervalAngles(
            out_key="angles",
            angles_loc_scale=FACIT_ANGLES_LOC_SCALE,
        )

    def _make_transforms(self, root: str | Path | None = None) -> list:
        return [
            MatterGenToFacitFields(),
            FullyConnectedGraph(),
            self.make_lengths_transform(root=root),
            self.make_angles_transform(),
            ConcatFeatures(in_keys=["lengths", "angles"], out_key="l"),
            ToPyGData(),
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
        num_workers: int = 2,
        pin_memory: bool = False,
        download: bool = False,
    ) -> DataLoader:
        dataset = self.fit_dataset(root=root, split=split, download=download)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
