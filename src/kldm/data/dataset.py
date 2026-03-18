from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch, Data


Transform = Callable[[Data], Data] | None


def _to_crystal_data(sample: Data | dict) -> Data:
    if isinstance(sample, Data):
        return sample.clone()
    return Data(
        pos=torch.as_tensor(sample["pos"], dtype=torch.float32),
        h=torch.as_tensor(sample["h"], dtype=torch.long),
        lengths=torch.as_tensor(sample["lengths"], dtype=torch.float32).view(1, -1),
        angles=torch.as_tensor(sample["angles"], dtype=torch.float32).view(1, -1),
    )


def _random_fractional_positions(num_atoms: int) -> torch.Tensor:
    return torch.rand(num_atoms, 3)


def _formula_to_atomic_numbers(formula: dict[str, float]) -> torch.Tensor:
    from ase.data import atomic_numbers

    species = [
        atomic_numbers[element]
        for element, count in formula.items()
        for _ in range(int(count))
    ]
    return torch.tensor(species, dtype=torch.long)


class CrystalDataset(TorchDataset, ABC):
    """Small PyG-friendly dataset base for clean crystals and sampling priors."""

    def __init__(self, transform: Transform = None) -> None:
        self.transform = transform

    @staticmethod
    def collate_fn(samples: list[Data]) -> Batch:
        return Batch.from_data_list(samples)

    @abstractmethod
    def make_data(self, idx: int) -> Data:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Data:
        data = self.make_data(idx)
        return data if self.transform is None else self.transform(data)


class StoredCrystalDataset(CrystalDataset):
    """Real crystals loaded from processed MP-20 tensor files."""

    def __init__(self, path: str | Path, transform: Transform = None) -> None:
        super().__init__(transform=transform)
        self.path = Path(path)
        self.samples = torch.load(self.path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self.samples)

    def make_data(self, idx: int) -> Data:
        return _to_crystal_data(self.samples[idx])


class FormulaDataset(CrystalDataset):
    """Formula-conditioned prior used for CSP sampling."""

    def __init__(
        self,
        formulas: Sequence[str],
        repeats: int = 1,
        transform: Transform = None,
    ) -> None:
        super().__init__(transform=transform)
        import chemparse

        self.compositions = [
            chemparse.parse_formula(formula)
            for formula in formulas
            for _ in range(int(repeats))
        ]

    def __len__(self) -> int:
        return len(self.compositions)

    def make_data(self, idx: int) -> Data:
        h = _formula_to_atomic_numbers(self.compositions[idx])
        return Data(pos=_random_fractional_positions(len(h)), h=h)


class SizePriorDataset(CrystalDataset):
    """Size-conditioned prior used for DNG sampling."""

    def __init__(
        self,
        size_distribution: np.ndarray,
        n_samples: int = 1000,
        default_atomic_number: int = 6,
        seed: int = 42,
        transform: Transform = None,
    ) -> None:
        super().__init__(transform=transform)
        self.default_atomic_number = int(default_atomic_number)
        rng = np.random.default_rng(seed)
        self.sizes = rng.choice(len(size_distribution), size=n_samples, p=size_distribution)

    @staticmethod
    def uniform_size_prior(size_range: str) -> np.ndarray:
        lower, upper = (int(part) for part in size_range.split("-"))
        prior = np.zeros(upper + 1, dtype=float)
        prior[lower:upper] = 1.0
        return prior / prior.sum()

    def __len__(self) -> int:
        return len(self.sizes)

    def make_data(self, idx: int) -> Data:
        num_atoms = int(self.sizes[idx])
        return Data(
            pos=_random_fractional_positions(num_atoms),
            h=torch.full((num_atoms,), self.default_atomic_number, dtype=torch.long),
        )


Dataset = StoredCrystalDataset
