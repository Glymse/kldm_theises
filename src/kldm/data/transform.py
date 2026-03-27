from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.transform import Transform
from torch import Tensor
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import dense_to_sparse, one_hot


DEFAULT_ATOMIC_VOCAB: list[int] = list(range(1, 119))


def _cell_lengths_and_angles(cell_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract lattice lengths and angles in radians from a 3x3 cell matrix."""
    lengths = torch.linalg.norm(cell_matrix, dim=1)
    alpha = torch.acos(torch.clamp(torch.dot(cell_matrix[1], cell_matrix[2]) / (lengths[1] * lengths[2]), -1.0, 1.0))
    beta = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[2]) / (lengths[0] * lengths[2]), -1.0, 1.0))
    gamma = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[1]) / (lengths[0] * lengths[1]), -1.0, 1.0))
    return lengths, torch.stack([alpha, beta, gamma])


class PlusOneAtomicNumbers(Transform):
    """Toy transform that adds one to the atomic numbers."""

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        return sample.replace(atomic_numbers=sample.atomic_numbers + 1)


@functional_transform("fully_connected_graph")
class FullyConnectedGraph(Transform):
    """Create a fully connected graph stored in `edge_node_index`."""

    def __init__(self, key: str = "edge_node_index", len_from: str = "pos") -> None:
        self.key = key
        self.len_from = len_from

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        # KLDM uses a dense all-to-all graph for message passing in these examples.
        n = len(getattr(sample, self.len_from))
        fc_graph = torch.ones(n, n, device=sample.pos.device) - torch.eye(n, device=sample.pos.device)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return sample.replace(**{self.key: fc_edges})


@functional_transform("continuous_interval_lengths")
class ContinuousIntervalLengths(Transform):
    """Transform lattice lengths with log, optionally normalized by number of atoms."""

    def __init__(
        self,
        in_key: str = "lengths",
        out_key: str | None = None,
        normalize_by_num_atoms: bool = False,
        cache_file: str | Path | None = None,
        quantile: float = 0.025,
    ) -> None:
        self.in_key = in_key
        self.out_key = out_key
        self.normalize_by_num_atoms = normalize_by_num_atoms
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.quantile = quantile
        self.loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        if self.cache_file and self.cache_file.exists():
            with self.cache_file.open(encoding="utf-8") as f:
                loaded = json.load(f)
            self.loc_scale = {int(k): (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in loaded.items()}

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        if not hasattr(sample, "cell"):
            raise ValueError("ChemGraph must have a 'cell' attribute to use ContinuousIntervalLengths transform.")

        n_atoms = int(sample.num_atoms)
        cell_matrix = sample.cell.squeeze(0)
        lengths, _ = _cell_lengths_and_angles(cell_matrix)
        if self.normalize_by_num_atoms:
            lengths = lengths / (n_atoms ** (1 / 3))
        log_lengths = torch.log(lengths)
        if n_atoms in self.loc_scale:
            loc, scale = self.loc_scale[n_atoms]
            log_lengths = (log_lengths - loc) / scale

        key = self.out_key if self.out_key is not None else self.in_key
        return sample.replace(**{key: log_lengths})

    def compute_loc_scale(self, samples: list[ChemGraph]) -> None:
        lengths_by_n = defaultdict(list)

        for sample in samples:
            n_atoms = int(sample.num_atoms)
            cell_matrix = sample.cell.squeeze(0)
            lengths, _ = _cell_lengths_and_angles(cell_matrix)
            if self.normalize_by_num_atoms:
                lengths = lengths / (n_atoms ** (1 / 3))
            lengths_by_n[n_atoms].append(torch.log(lengths))

        for n_atoms, values in lengths_by_n.items():
            values_stack = torch.stack(values)
            q = int(values_stack.shape[0] * self.quantile)
            values_sorted, _ = torch.sort(values_stack, dim=0)
            values_trimmed = values_sorted[q:-q] if q > 0 else values_sorted
            self.loc_scale[n_atoms] = (values_trimmed.mean(dim=0), values_trimmed.std(dim=0))

        if self.cache_file:
            payload = {n: [loc.tolist(), scale.tolist()] for n, (loc, scale) in self.loc_scale.items()}
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def invert_one(self, log_lengths: Tensor, n_atoms: int) -> Tensor:
        lengths = log_lengths.clone()
        if n_atoms in self.loc_scale:
            loc, scale = self.loc_scale[n_atoms]
            lengths = lengths * scale + loc
        lengths = torch.exp(lengths)
        if self.normalize_by_num_atoms:
            lengths = lengths * (n_atoms ** (1 / 3))
        return lengths


@functional_transform("continuous_interval_lattice")
class ContinuousIntervalLattice(Transform):
    """Combined transform for lattice lengths and angles."""

    def __init__(
        self,
        lengths_in_key: str = "lengths",
        lengths_out_key: str | None = None,
        angles_in_key: str = "angles",
        angles_out_key: str | None = None,
        normalize_lengths_by_num_atoms: bool = False,
        cache_file: str | Path | None = None,
        lengths_quantile: float = 0.025,
        angles_loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
        angles_in_deg: bool = True,
    ) -> None:
        self.lengths_in_key = lengths_in_key
        self.lengths_out_key = lengths_out_key or lengths_in_key
        self.angles_in_key = angles_in_key
        self.angles_out_key = angles_out_key or angles_in_key
        self.normalize_lengths_by_num_atoms = normalize_lengths_by_num_atoms
        self.cache_file = Path(cache_file) if cache_file else None
        self.lengths_quantile = lengths_quantile
        self.angles_loc_scale = angles_loc_scale
        self.angles_in_deg = angles_in_deg

        self.lengths_loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.cache_file and self.cache_file.exists():
            with self.cache_file.open(encoding="utf-8") as f:
                loaded = json.load(f)
            self.lengths_loc_scale = {int(k): (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in loaded.items()}

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        if not hasattr(sample, "cell"):
            raise ValueError("ChemGraph must have a 'cell' attribute to use ContinuousIntervalLattice transform.")

        n_atoms = int(sample.num_atoms)
        cell_matrix = sample.cell.squeeze(0)

        # MatterGen stores the lattice as a full cell matrix; KLDM uses
        # a compact 6D representation [log_lengths, tan(angle - pi/2)].
        lengths, angles_rad = _cell_lengths_and_angles(cell_matrix)
        if self.normalize_lengths_by_num_atoms:
            lengths = lengths / (n_atoms ** (1 / 3))
        log_lengths = torch.log(lengths)
        if n_atoms in self.lengths_loc_scale:
            loc, scale = self.lengths_loc_scale[n_atoms]
            log_lengths = (log_lengths - loc) / scale

        transformed_angles = torch.tan(angles_rad - torch.pi / 2)
        if self.angles_loc_scale is not None:
            loc, scale = self.angles_loc_scale
            transformed_angles = (transformed_angles - loc) / scale

        log_lengths = log_lengths.view(1, 3)
        transformed_angles = transformed_angles.view(1, 3)
        lattice = torch.cat([log_lengths, transformed_angles], dim=-1)

        return sample.replace(
            **{
                self.lengths_out_key: log_lengths,
                self.angles_out_key: transformed_angles,
                "l": lattice,
            }
        )


@functional_transform("one_hot")
class OneHot(Transform):
    """One-hot encode a categorical 1D tensor stored at `key`."""

    def __init__(
        self,
        values: list[int],
        key: str = "h",
        scale: float = 1.0,
        noise_std: float = 0.0,
        dtype: torch.dtype = torch.get_default_dtype(),
        expand_as_vector: bool = True,
    ) -> None:
        self.mapping = {v: i for (i, v) in enumerate(values)}
        self.key = key
        self.dtype = dtype
        self.noise_std = noise_std
        self.scale = scale
        self.expand_as_vector = expand_as_vector

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        data_key = getattr(sample, self.key)
        assert data_key.ndim == 1

        x = torch.as_tensor([self.mapping[xi.item()] for xi in data_key], device=data_key.device)
        if self.expand_as_vector:
            x = self.scale * one_hot(x, num_classes=len(self.mapping)).to(self.dtype)
            if self.noise_std > 0.0:
                x = x + torch.randn_like(x) * self.noise_std
        return sample.replace(**{self.key: x})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.mapping})"


@functional_transform("concat_features")
class ConcatFeatures(Transform):
    """Concatenate multiple features and store them in `out_key`."""

    def __init__(self, in_keys: list[str], out_key: str, dim: int = -1) -> None:
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        features = [getattr(sample, key) for key in self.in_keys]
        concat_features = torch.cat(features, dim=self.dim)
        return sample.replace(**{self.out_key: concat_features})


class CopyProperty(Transform):
    """Copy one attribute to another attribute name."""

    def __init__(self, in_key: str, out_key: str) -> None:
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        return sample.replace(**{self.out_key: getattr(sample, self.in_key)})


class TaskMetadata(Transform):
    """Attach KLDM task metadata to a ChemGraph."""

    def __init__(self, task_id: int, diffuse_h: bool, is_prior: bool = False) -> None:
        self.task_id = int(task_id)
        self.diffuse_h = bool(diffuse_h)
        self.is_prior = bool(is_prior)

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        # These small flags are enough for KLDM to distinguish CSP vs DNG batches.
        return sample.replace(
            task_id=torch.tensor([self.task_id], dtype=torch.long),
            diffuse_h=torch.tensor([self.diffuse_h], dtype=torch.bool),
            is_prior=torch.tensor([self.is_prior], dtype=torch.bool),
            num_atoms=torch.tensor([int(sample.num_atoms)], dtype=torch.long),
        )
