from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.transform import Transform
from torch_geometric.utils import dense_to_sparse


# Atomic vocabulary used when atom types are represented by indices.
# Here the vocabulary is simply all elements with atomic number 1 to 118.
DEFAULT_ATOMIC_VOCAB: list[int] = list(range(1, 119))
DEFAULT_X0_ANGLE_STATS: tuple[float, float] = (0.0, 0.35)


"""Lattice preprocessing for the KLDM CSP pipeline.

Each cell becomes a 6D feature vector:
    [log(a), log(b), log(c), tan(alpha - pi/2), tan(beta - pi/2), tan(gamma - pi/2)]

`eps` mode uses those features directly.
`x0` mode standardizes:
    - log-lengths with stats keyed by number of atoms
    - angle features with one shared loc/scale pair
"""

# Segments marked with
# `#code segment is from original kldm code. data/transforms.py`
# follow the original x0 preprocessing pattern closely.
# Reference:
# /Users/glymov/DTU/6 Semester/Bachelor/Github/Main/kldm/src/facitKLDM/kldm-main-git/src_kldm/data/transforms.py

def cell_lengths_and_angles(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 3x3 lattice matrix to lengths and angles.
            alpha = angle between b and c
            beta  = angle between a and c
            gamma = angle between a and b
    """

    lengths = torch.linalg.norm(cell, dim=1)

    alpha = torch.acos(
        torch.clamp(
            torch.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]),
            -1.0,
            1.0,
        )
    )
    beta = torch.acos(
        torch.clamp(
            torch.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]),
            -1.0,
            1.0,
        )
    )
    gamma = torch.acos(
        torch.clamp(
            torch.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]),
            -1.0,
            1.0,
        )
    )

    return lengths, torch.stack([alpha, beta, gamma])


def lattice_feature_components(
    cell: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a cell into transformed length and angle features."""
    lengths, angles = cell_lengths_and_angles(cell)
    log_lengths = torch.log(lengths.clamp_min(eps))
    angle_features = torch.tan(angles - torch.pi / 2.0)
    return log_lengths, angle_features


def _has_x0_lattice_stats(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("lengths_loc_scale"), dict)
        and isinstance(payload.get("angles_loc_scale"), list)
        and len(payload["angles_loc_scale"]) == 2
    )


def ensure_lattice_standardization_cache(
    *,
    cache_file: str | Path,
    processed_dir: str | Path,
    eps: float = 1e-8,
) -> Path:
    """Create train-set statistics for x0 lattice preprocessing."""
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                existing_payload = json.load(handle)
            if _has_x0_lattice_stats(existing_payload):
                return cache_path
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    cell_path = Path(processed_dir) / "cell.npy"
    num_atoms_path = Path(processed_dir) / "num_atoms.npy"
    cells = np.load(cell_path, allow_pickle=True)
    num_atoms = np.load(num_atoms_path, allow_pickle=True)

    #code segment is from original kldm code. data/transforms.py
    lengths_by_num_atoms: dict[int, list[torch.Tensor]] = {}
    for cell, n_atoms in zip(cells, num_atoms):
        cell = torch.as_tensor(cell, dtype=torch.get_default_dtype())

        # MatterGen may store a cell as shape (1, 3, 3).
        # The transform expects shape (3, 3).
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.squeeze(0)

        log_lengths, _ = lattice_feature_components(cell, eps=eps)
        lengths_by_num_atoms.setdefault(int(n_atoms), []).append(log_lengths)

    stats_by_size: dict[str, list[list[float]]] = {}
    for num_atoms, values in sorted(lengths_by_num_atoms.items()):
        stacked = torch.stack(values, dim=0)
        center = stacked.mean(dim=0)
        spread = stacked.std(dim=0, unbiased=False).clamp_min(eps)
        stats_by_size[str(num_atoms)] = [center.tolist(), spread.tolist()]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "lengths_loc_scale": stats_by_size,
                "angles_loc_scale": list(DEFAULT_X0_ANGLE_STATS),
            },
            handle,
            indent=2,
        )

    return cache_path


class FullyConnectedGraph(Transform):
    """
    Add fully connected directed edges to a crystal graph.

    """

    def __init__(self, key: str = "edge_node_index", len_from: str = "pos") -> None:
        """Store transform configuration.

        Input:
            key:
                Name of the output edge-index field.

            len_from:
                Name of the tensor used to infer the number of atoms.
        """
        self.key = key
        self.len_from = len_from

    def __call__(self, sample: ChemGraph) -> ChemGraph:

        n = len(getattr(sample, self.len_from))

        adjacency = torch.ones(n, n, device=sample.pos.device)
        adjacency = adjacency - torch.eye(n, device=sample.pos.device)

        edge_index, _ = dense_to_sparse(adjacency)

        return sample.replace(**{self.key: edge_index})


class ContinuousIntervalLattice(Transform):
    """
    Encode/decode lattice parameters for Euclidean diffusion.

    Forward transform:
        cell matrix -> 6D lattice vector

    eps mode:
        use the transformed 6D lattice vector directly

    x0 mode:
        standardize log-lengths with graph-size-specific stats
        standardize angle features with one shared loc/scale pair

    Inverse transform:
        l -> physical lengths and angles

    Input ChemGraph:
        Must contain:
            sample.cell, shape usually (1, 3, 3)

    Output ChemGraph:
        Adds:
            sample.l, shape (1, 6)
    """

    def __init__(
        self,
        out_key: str = "l",
        standardize: bool = False,
        cache_file: str | Path | None = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the lattice transform.

        Output:
            Transform object with optional standardization statistics loaded.
        """
        self.out_key = out_key
        self.standardize = standardize
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.eps = eps

        self.lengths_loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.angles_loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None

        if self.standardize:
            if self.cache_file is None or not self.cache_file.exists():
                raise ValueError("x0 lattice mode requires an existing stats cache.")
            with self.cache_file.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)

            if not _has_x0_lattice_stats(stats):
                raise ValueError("x0 lattice stats cache has an unsupported format.")

            #code segment is from original kldm code. data/transforms.py
            self.lengths_loc_scale = {
                int(num_atoms): (
                    torch.tensor(center, dtype=torch.get_default_dtype()),
                    torch.tensor(spread, dtype=torch.get_default_dtype()),
                )
                for num_atoms, (center, spread) in stats["lengths_loc_scale"].items()
            }
            angle_center, angle_spread = stats["angles_loc_scale"]
            self.angles_loc_scale = (
                torch.tensor(angle_center, dtype=torch.get_default_dtype()),
                torch.tensor(angle_spread, dtype=torch.get_default_dtype()),
            )

    def _broadcast_feature_stats(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Broadcast per-feature stats across batched lattice tensors.
        loc = loc.to(device=value.device, dtype=value.dtype)
        scale = scale.to(device=value.device, dtype=value.dtype).clamp_min(self.eps)
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(0)
            scale = scale.unsqueeze(0)
        return loc, scale

    def _length_stats_for_num_atoms(
        self,
        num_atoms: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick the x0 length statistics that match this crystal size.
        if self.lengths_loc_scale is None or num_atoms not in self.lengths_loc_scale:
            raise KeyError(f"Missing x0 length statistics for num_atoms={num_atoms}.")
        loc, scale = self.lengths_loc_scale[num_atoms]
        return self._broadcast_feature_stats(loc, scale, value)

    def _encode_x0_parts(
        self,
        *,
        log_lengths: torch.Tensor,
        angle_features: torch.Tensor,
        num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply the x0 normalization used during training.
        #code segment is from original kldm code. data/transforms.py
        loc_lengths, scale_lengths = self._length_stats_for_num_atoms(num_atoms, log_lengths)
        log_lengths = (log_lengths - loc_lengths) / scale_lengths

        if self.angles_loc_scale is not None:
            angle_loc, angle_scale = self.angles_loc_scale
            angle_loc, angle_scale = self._broadcast_feature_stats(angle_loc, angle_scale, angle_features)
            angle_features = (angle_features - angle_loc) / angle_scale

        return log_lengths, angle_features

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Encode a sample's cell matrix into `sample.l`.

        Input:
            sample:
                ChemGraph containing `cell`.

        Output:
            ChemGraph with added lattice tensor:
                l shape = (1, 6)
        """
        cell = sample.cell.squeeze(0)

        # Always build the same 6D lattice representation before optional x0 scaling.
        log_lengths, angle_features = lattice_feature_components(cell, eps=self.eps)
        if self.standardize:
            log_lengths, angle_features = self._encode_x0_parts(
                log_lengths=log_lengths,
                angle_features=angle_features,
                num_atoms=int(len(sample.pos)),
            )
        features = torch.cat([log_lengths, angle_features], dim=0)

        return sample.replace(**{self.out_key: features.view(1, 6)})

    def invert_to_lengths_angles(
        self,
        l: torch.Tensor,
        num_atoms: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode lattice vector into physical lengths and angles.

        Input:
            l:
                Tensor with last dimension 6.
                May be standardized or unstandardized depending on the transform.
        Output:
            lengths:
                Tensor with last dimension 3.

            angles:
                Tensor with last dimension 3, in radians.

        Inverse map:
            lengths = exp(l[..., :3])
            angles  = atan(l[..., 3:]) + pi/2
        """
        if self.standardize:
            if num_atoms is None:
                raise ValueError("num_atoms is required to invert x0-standardized lattice features.")

            # Undo the x0 normalization with the same stats used on encode.
            #code segment is from original kldm code. data/transforms.py
            flat_features = l.reshape(-1, 6)
            if isinstance(num_atoms, torch.Tensor):
                flat_num_atoms = num_atoms.reshape(-1).detach().cpu().tolist()
            elif isinstance(num_atoms, int):
                flat_num_atoms = [num_atoms] * flat_features.shape[0]
            else:
                flat_num_atoms = list(num_atoms)

            if len(flat_num_atoms) != flat_features.shape[0]:
                raise ValueError("num_atoms must match the batch size of lattice features.")

            log_lengths = flat_features[:, :3].clone()
            angle_features = flat_features[:, 3:].clone()

            restored_lengths = []
            for row_idx, n_atoms in enumerate(flat_num_atoms):
                loc_lengths, scale_lengths = self._length_stats_for_num_atoms(int(n_atoms), log_lengths[row_idx])
                restored_lengths.append(log_lengths[row_idx] * scale_lengths + loc_lengths)
            log_lengths = torch.stack(restored_lengths, dim=0)

            if self.angles_loc_scale is not None:
                angle_loc, angle_scale = self.angles_loc_scale
                angle_loc, angle_scale = self._broadcast_feature_stats(angle_loc, angle_scale, angle_features)
                angle_features = angle_features * angle_scale + angle_loc

            log_lengths = log_lengths.reshape(*l.shape[:-1], 3)
            angle_features = angle_features.reshape(*l.shape[:-1], 3)
        else:
            log_lengths = l[..., :3]
            angle_features = l[..., 3:]

        lengths = torch.exp(log_lengths)
        angles = torch.atan(angle_features) + torch.pi / 2.0

        return lengths, angles
