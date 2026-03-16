from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.utils.data as torchdata
from pymatgen.core import Element
from torch_geometric.data import Data
from torch_geometric.io import fs


EMPIRICAL_LEN_DISTRIBUTIONS: dict[str, np.ndarray] = {}


class Dataset(torchdata.Dataset):
    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        self.transform = transform
        self.data = self.load(path)

    @staticmethod
    def load(path: str):
        try:
            return fs.torch_load(path)
        except AttributeError:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")

    def __getitem__(self, idx: int) -> Data:
        data = self.data[idx]
        if not isinstance(data, Data):
            data = Data(
                pos=torch.tensor(data["pos"], dtype=torch.float32),
                h=torch.tensor(data["h"], dtype=torch.long),
                lengths=torch.tensor(data["lengths"], dtype=torch.float32).view(1, -1),
                angles=torch.tensor(data["angles"], dtype=torch.float32).view(1, -1),
            )
        return data if self.transform is None else self.transform(data)

    def __len__(self) -> int:
        return len(self.data)


class SampleDatasetDNG(torchdata.Dataset):
    def __init__(
        self,
        empirical_distribution: np.ndarray,
        n_samples: int = 10_000,
        transform: Optional[Callable[[Data], Data]] = None,
        seed: int = 42,
    ) -> None:
        self.empirical_distribution = np.asarray(empirical_distribution, dtype=np.float64)
        rng = np.random.RandomState(seed)
        self.num_atoms = rng.choice(
            len(self.empirical_distribution), n_samples, p=self.empirical_distribution
        )
        self.transform = transform

    def __getitem__(self, idx: int) -> Data:
        n = int(self.num_atoms[idx])
        data = Data(
            pos=torch.randn(n, 3),
            h=torch.tensor([6] * n, dtype=torch.long),
        )
        return data if self.transform is None else self.transform(data)

    def __len__(self) -> int:
        return len(self.num_atoms)

    @classmethod
    def from_cmd_args(
        cls,
        data_name: str,
        len_range: Optional[str] = None,
        n_samples: int = 10_000,
        transform: Optional[Callable[[Data], Data]] = None,
        seed: int = 42,
    ) -> "SampleDatasetDNG":
        if len_range is None:
            empirical_distribution = EMPIRICAL_LEN_DISTRIBUTIONS[data_name]
        else:
            empirical_distribution = cls.uniform_from_range(len_range)

        return cls(
            empirical_distribution=empirical_distribution,
            n_samples=n_samples,
            transform=transform,
            seed=seed,
        )

    @staticmethod
    def uniform_from_range(len_range: str) -> np.ndarray:
        lower, upper = (int(value) for value in len_range.split("-", 1))
        empirical_distribution = np.zeros(upper + 1)
        empirical_distribution[lower:upper] = 1.0
        empirical_distribution /= np.sum(empirical_distribution)
        return empirical_distribution


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    return [
        (symbol, int(count) if count else 1)
        for symbol, count in re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    ]


class SampleDatasetCSP(torchdata.Dataset):
    def __init__(
        self,
        formulas: Sequence[str],
        n_samples_per_formula: int = 5,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        self.transform = transform
        self.parsed_formulas = [
            _parse_formula(formula)
            for formula in formulas
            for _ in range(n_samples_per_formula)
        ]

    def __getitem__(self, idx: int) -> Data:
        formula = self.parsed_formulas[idx]
        h = torch.tensor(
            [int(Element(el).Z) for el, count in formula for _ in range(count)],
            dtype=torch.long,
        )
        data = Data(pos=torch.randn(len(h), 3), h=h)
        return data if self.transform is None else self.transform(data)

    def __len__(self) -> int:
        return len(self.parsed_formulas)
