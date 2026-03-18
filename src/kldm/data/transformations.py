from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


DEFAULT_ATOMIC_VOCAB: tuple[int, ...] = tuple(range(1, 119))


def load_length_stats(path: str | Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    with Path(path).open(encoding="utf-8") as fp:
        raw = json.load(fp)
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, value in raw.items():
        loc, scale = value
        stats[int(key)] = (
            torch.tensor(loc, dtype=torch.float32),
            torch.tensor(scale, dtype=torch.float32),
        )
    return stats


@dataclass(slots=True)
class TransformSpec:
    connect_graph: bool = True
    lattice_mode: Literal["kldm", "raw", "prior"] = "kldm"
    species_mode: Literal["atomic_numbers", "one_hot"] = "atomic_numbers"
    species_vocab: tuple[int, ...] = DEFAULT_ATOMIC_VOCAB
    default_lattice: Optional[torch.Tensor] = None
    length_stats_path: Optional[str | Path] = None


class SpeciesEncoder:
    """Project atomic numbers into the representation consumed by the model."""

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return h.to(dtype=torch.long)


class FixedVocabularyOneHotEncoder(SpeciesEncoder):
    def __init__(self, species_vocab: Sequence[int]) -> None:
        if len(species_vocab) == 0:
            raise ValueError("species_vocab must contain at least one atomic number.")
        ordered_vocab = tuple(int(z) for z in species_vocab)
        self.species_vocab = ordered_vocab
        self.mapping = {atomic_number: idx for idx, atomic_number in enumerate(ordered_vocab)}

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(dtype=torch.long)
        if h.numel() == 0:
            return torch.empty((0, len(self.species_vocab)), dtype=torch.float32)

        try:
            class_index = torch.tensor(
                [self.mapping[int(atomic_number)] for atomic_number in h.tolist()],
                dtype=torch.long,
                device=h.device,
            )
        except KeyError as exc:
            raise ValueError(
                f"Encountered atomic number {exc.args[0]} outside the configured species vocabulary."
            ) from exc

        return F.one_hot(class_index, num_classes=len(self.species_vocab)).to(dtype=torch.float32)


class CrystalTransform(BaseTransform):
    """Minimal task-facing transform for KLDM data.

    It can:
    - attach a complete graph
    - project lattice parameters into KLDM-friendly space
    - keep species as atomic numbers or expand them to one-hot vectors
    """

    def __init__(self, spec: TransformSpec) -> None:
        self.spec = spec
        self.length_stats = None if spec.length_stats_path is None else load_length_stats(spec.length_stats_path)
        self.species_encoder = self._build_species_encoder(spec)

    def forward(self, data: Data) -> Data:
        if self.spec.connect_graph and not hasattr(data, "edge_node_index"):
            data.edge_node_index = self._complete_graph(data.pos.shape[0])

        data.l = self._lattice_tensor(data)
        data.h = self.species_encoder.encode(data.h)

        return data

    @staticmethod
    def _complete_graph(num_nodes: int) -> torch.Tensor:
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        return mask.nonzero(as_tuple=False).t().contiguous()

    def _lattice_tensor(self, data: Data) -> torch.Tensor:
        if self.spec.lattice_mode == "prior":
            default = torch.zeros(1, 6, dtype=torch.float32) if self.spec.default_lattice is None else self.spec.default_lattice
            return default.view(1, 6).to(dtype=torch.float32)

        if not hasattr(data, "lengths") or not hasattr(data, "angles"):
            raise ValueError("Expected 'lengths' and 'angles' to build a lattice tensor.")

        lengths = data.lengths.to(dtype=torch.float32)
        angles = data.angles.to(dtype=torch.float32)

        if self.spec.lattice_mode == "raw":
            return torch.cat([lengths, angles], dim=-1)

        lengths = self._normalize_lengths(torch.log(lengths), int(data.pos.shape[0]))
        angles = torch.tan(torch.deg2rad(angles) - math.pi / 2.0)

        data.lengths = lengths
        data.angles = angles
        return torch.cat([lengths, angles], dim=-1)

    @staticmethod
    def _build_species_encoder(spec: TransformSpec) -> SpeciesEncoder:
        if spec.species_mode == "atomic_numbers":
            return SpeciesEncoder()
        if spec.species_mode == "one_hot":
            return FixedVocabularyOneHotEncoder(spec.species_vocab)
        raise ValueError(f"Unsupported species_mode={spec.species_mode!r}")

    def _normalize_lengths(self, lengths: torch.Tensor, num_atoms: int) -> torch.Tensor:
        if self.length_stats is None or num_atoms not in self.length_stats:
            return lengths
        loc, scale = self.length_stats[num_atoms]
        return (lengths - loc.view(1, -1)) / torch.clamp_min(scale.view(1, -1), 1e-6)


def training_transform(
    *,
    species_mode: Literal["atomic_numbers", "one_hot"] = "atomic_numbers",
    species_vocab: Sequence[int] = DEFAULT_ATOMIC_VOCAB,
    length_stats_path: Optional[str | Path] = None,
) -> CrystalTransform:
    return CrystalTransform(
        TransformSpec(
            connect_graph=True,
            lattice_mode="kldm",
            species_mode=species_mode,
            species_vocab=tuple(int(z) for z in species_vocab),
            length_stats_path=length_stats_path,
        )
    )


def sampling_transform(
    *,
    species_mode: Literal["atomic_numbers", "one_hot"] = "atomic_numbers",
    species_vocab: Sequence[int] = DEFAULT_ATOMIC_VOCAB,
    default_lattice: Optional[torch.Tensor] = None,
) -> CrystalTransform:
    return CrystalTransform(
        TransformSpec(
            connect_graph=True,
            lattice_mode="prior",
            species_mode=species_mode,
            species_vocab=tuple(int(z) for z in species_vocab),
            default_lattice=default_lattice,
        )
    )
