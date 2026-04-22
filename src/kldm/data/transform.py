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
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data"
DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH = DEFAULT_DATA_ROOT / "mp_20" / "train_loc_scale.json"
FACIT_ANGLES_LOC_SCALE: tuple[float, float] = (0.0, 0.35)


def _cell_lengths_and_angles(cell_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract lattice lengths and angles in radians from a 3x3 cell matrix."""
    lengths = torch.linalg.norm(cell_matrix, dim=1)
    alpha = torch.acos(torch.clamp(torch.dot(cell_matrix[1], cell_matrix[2]) / (lengths[1] * lengths[2]), -1.0, 1.0))
    beta = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[2]) / (lengths[0] * lengths[2]), -1.0, 1.0))
    gamma = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[1]) / (lengths[0] * lengths[1]), -1.0, 1.0))
    return lengths, torch.stack([alpha, beta, gamma])


def ensure_lengths_loc_scale_cache(
    *,
    cache_file: str | Path,
    processed_dir: str | Path,
    quantile: float = 0.025,
    eps: float = 1e-8,
) -> Path:
    """Generate facit-style log-length stats from a processed split if missing.

    The KLDM paper/facit preprocessing stores per-atom-count statistics for
    `log(lengths)` and trims 2.5% from each tail before computing mean/std.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        return cache_path

    processed_path = Path(processed_dir)
    cell_path = processed_path / "cell.npy"
    num_atoms_path = processed_path / "num_atoms.npy"
    if not cell_path.exists() or not num_atoms_path.exists():
        return cache_path

    import numpy as np

    cells = np.load(cell_path, allow_pickle=True)
    num_atoms = np.load(num_atoms_path, allow_pickle=True)

    values_by_key: dict[int, list[torch.Tensor]] = defaultdict(list)
    for cell, n_atoms in zip(cells, num_atoms, strict=False):
        cell_tensor = torch.as_tensor(cell, dtype=torch.get_default_dtype())
        if cell_tensor.ndim == 3 and cell_tensor.shape[0] == 1:
            cell_tensor = cell_tensor.squeeze(0)
        lengths, _ = _cell_lengths_and_angles(cell_tensor)
        values_by_key[int(n_atoms)].append(torch.log(lengths.clamp_min(eps)))

    payload: dict[str, list[list[float]]] = {}
    for n_atoms, values in values_by_key.items():
        stacked = torch.stack(values, dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trim = int(len(values) * quantile)
        if trim > 0 and len(values) > 2 * trim:
            trimmed = sorted_vals[trim:-trim]
        else:
            trimmed = sorted_vals
        loc = trimmed.mean(dim=0)
        scale = trimmed.std(dim=0).clamp_min(eps)
        payload[str(n_atoms)] = [loc.tolist(), scale.tolist()]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return cache_path


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
    """
    Lattice transform for KLDM.

    Forward:
        cell -> [log(lengths), tan(angle - pi/2)]

    Inverse:
        unconstrained_l -> physical lengths/angles

    The KLDM paper's epsilon branch uses the unconstrained 6D lattice chart
    directly. Extra dataset standardization is optional and corresponds to the
    later x0/facit-style branch rather than the basic epsilon transform.
    """

    def __init__(
        self,
        out_key: str = "l",
        normalize_lengths_by_num_atoms: bool = False,
        cache_file: str | Path | None = None,
        quantile: float = 0.025,
        standardize: bool = False,
        standardize_by_num_atoms: bool = True,
        angles_loc_scale: tuple[float, float] | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.out_key = out_key
        self.normalize_lengths_by_num_atoms = normalize_lengths_by_num_atoms
        if cache_file is None and standardize:
            cache_file = DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.quantile = float(quantile)
        self.standardize = bool(standardize)
        self.standardize_by_num_atoms = bool(standardize_by_num_atoms)
        if angles_loc_scale is None:
            self.angles_loc_scale = None
        else:
            self.angles_loc_scale = (float(angles_loc_scale[0]), float(angles_loc_scale[1]))
        self.eps = float(eps)

        # key -> (loc[3], scale[3]) for log-lengths only, matching facit.
        self.lengths_loc_scale: dict[int | str, tuple[torch.Tensor, torch.Tensor]] = {}

        if self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw:
                first_value = next(iter(raw.values()))
                if isinstance(first_value, dict):
                    self.lengths_loc_scale = {
                        self._decode_key(k): (
                            torch.tensor(v["loc"][:3], dtype=torch.get_default_dtype()),
                            torch.tensor(v["scale"][:3], dtype=torch.get_default_dtype()),
                        )
                        for k, v in raw.items()
                    }
                else:
                    self.lengths_loc_scale = {
                        self._decode_key(k): (
                            torch.tensor(v[0], dtype=torch.get_default_dtype()),
                            torch.tensor(v[1], dtype=torch.get_default_dtype()),
                        )
                        for k, v in raw.items()
                    }

    def _group_key(self, n_atoms: int) -> int | str:
        return int(n_atoms) if self.standardize_by_num_atoms else "global"

    @staticmethod
    def _decode_key(k: str) -> int | str:
        return "global" if k == "global" else int(k)

    def _encode_key(self, k: int | str) -> str:
        return str(k)

    def _to_unconstrained_lattice(self, cell_matrix: torch.Tensor, n_atoms: int) -> torch.Tensor:
        lengths, angles_rad = _cell_lengths_and_angles(cell_matrix)

        if self.normalize_lengths_by_num_atoms:
            lengths = lengths / (float(n_atoms) ** (1.0 / 3.0))

        log_lengths = torch.log(lengths.clamp_min(self.eps))
        angle_features = torch.tan(angles_rad - torch.pi / 2)

        return torch.cat([log_lengths, angle_features], dim=0)

    def _standardize_lengths(self, log_lengths: torch.Tensor, n_atoms: int) -> torch.Tensor:
        if not self.standardize:
            return log_lengths
        key = self._group_key(n_atoms)
        if key not in self.lengths_loc_scale:
            return log_lengths
        loc, scale = self.lengths_loc_scale[key]
        return (log_lengths - loc.to(device=log_lengths.device, dtype=log_lengths.dtype)) / scale.to(
            device=log_lengths.device,
            dtype=log_lengths.dtype,
        ).clamp_min(self.eps)

    def _unstandardize_lengths(self, log_lengths: torch.Tensor, num_atoms: torch.Tensor | int) -> torch.Tensor:
        if not self.standardize:
            return log_lengths

        if isinstance(num_atoms, int):
            key = self._group_key(num_atoms)
            if key not in self.lengths_loc_scale:
                return log_lengths
            loc, scale = self.lengths_loc_scale[key]
            loc = loc.to(device=log_lengths.device, dtype=log_lengths.dtype)
            scale = scale.to(device=log_lengths.device, dtype=log_lengths.dtype)
            while loc.ndim < log_lengths.ndim:
                loc = loc.unsqueeze(0)
                scale = scale.unsqueeze(0)
            return log_lengths * scale.clamp_min(self.eps) + loc

        num_atoms = num_atoms.to(device=log_lengths.device)
        out = []
        for i in range(log_lengths.shape[0]):
            key = self._group_key(int(num_atoms[i].item()))
            li = log_lengths[i]
            if key in self.lengths_loc_scale:
                loc, scale = self.lengths_loc_scale[key]
                loc = loc.to(device=li.device, dtype=li.dtype)
                scale = scale.to(device=li.device, dtype=li.dtype)
                li = li * scale.clamp_min(self.eps) + loc
            out.append(li)
        return torch.stack(out, dim=0)

    def _standardize_angles(self, angle_features: torch.Tensor) -> torch.Tensor:
        if not self.standardize or self.angles_loc_scale is None:
            return angle_features
        loc, scale = self.angles_loc_scale
        return (angle_features - loc) / max(scale, self.eps)

    def _unstandardize_angles(self, angle_features: torch.Tensor) -> torch.Tensor:
        if not self.standardize or self.angles_loc_scale is None:
            return angle_features
        loc, scale = self.angles_loc_scale
        return angle_features * max(scale, self.eps) + loc

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        if not hasattr(sample, "cell"):
            raise ValueError("ChemGraph must have a 'cell' attribute to use ContinuousIntervalLattice transform.")

        n_atoms = int(sample.num_atoms)
        cell_matrix = sample.cell.squeeze(0)
        l_unscaled = self._to_unconstrained_lattice(cell_matrix=cell_matrix, n_atoms=n_atoms)

        log_lengths = self._standardize_lengths(l_unscaled[:3], n_atoms=n_atoms)
        angle_features = self._standardize_angles(l_unscaled[3:])
        l = torch.cat([log_lengths, angle_features], dim=0)

        return sample.replace(
            **{
                self.out_key: l.view(1, 6),
                "l_unscaled": l_unscaled.view(1, 6),
            }
        )

    def compute_loc_scale(self, samples: list[ChemGraph]) -> None:
        values_by_key = defaultdict(list)

        for sample in samples:
            cell_matrix = sample.cell.squeeze(0)
            n_atoms = int(sample.num_atoms)
            vec = self._to_unconstrained_lattice(cell_matrix=cell_matrix, n_atoms=n_atoms)[:3]
            values_by_key[self._group_key(n_atoms)].append(vec)

        self.lengths_loc_scale = {}
        for key, values in values_by_key.items():
            stacked = torch.stack(values, dim=0)
            q = int(len(stacked) * self.quantile)

            if q > 0 and len(stacked) > 2 * q:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[q:-q]
            else:
                trimmed = stacked

            loc = trimmed.mean(dim=0)
            scale = trimmed.std(dim=0).clamp_min(self.eps)
            self.lengths_loc_scale[key] = (loc, scale)

        if self.cache_file is not None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                self._encode_key(k): [loc.tolist(), scale.tolist()]
                for k, (loc, scale) in self.lengths_loc_scale.items()
            }
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

    def unstandardize(self, l: torch.Tensor, num_atoms: torch.Tensor | int) -> torch.Tensor:
        """
        l: (..., 6) standardized lattice features
        returns: (..., 6) unscaled unconstrained lattice features
        """
        log_lengths = self._unstandardize_lengths(l[..., :3], num_atoms=num_atoms)
        angle_feats = self._unstandardize_angles(l[..., 3:])
        return torch.cat([log_lengths, angle_feats], dim=-1)

    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        l: (..., 6) standardized or unstandardized unconstrained features
        returns:
            lengths (..., 3)
            angles_rad (..., 3)
        """
        l_unscaled = self.unstandardize(l, num_atoms=num_atoms)

        log_lengths = l_unscaled[..., :3]
        angle_feats = l_unscaled[..., 3:]

        lengths = torch.exp(log_lengths)
        if self.normalize_lengths_by_num_atoms:
            if isinstance(num_atoms, int):
                lengths = lengths * (float(num_atoms) ** (1.0 / 3.0))
            else:
                factor = num_atoms.to(device=lengths.device, dtype=lengths.dtype).pow(1.0 / 3.0).unsqueeze(-1)
                lengths = lengths * factor

        angles = torch.atan(angle_feats) + torch.pi / 2
        return lengths, angles


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
