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


"""
                                        DOCUMENTATION OF OUR TRANSFORM.py

The wanted output of our lattice should be the 1x6 lattice basis [len(a), ... angle(a),...]

The original lattice data is given as a 3x3 matrix, where row 1 is the complete lattice information of basis vector a.

To convert to the 1x6,
 - we first calculate len(a) = ||a||
 - Secondly does alpha = arccos(bc/|b||c|)

For the three lattice

So now the new basis becomes (a,b,c,alpha,beta,gamma)

Secondly, we log-transform the lengths

and transform the using tan(theta-pi/2)

We thus get:
    (log len(a), log len(b), log len(c), tan(alpha - pi/2), tan(beta - pi/2), tan(gamma - pi/2))

Lastly, if we use x0 in TDM, we load all the lattice matrices, convert to 6D vector, then we calculate mean, std of lattice.
These statistics are saved as JSON:

{
  "loc":   [mean_0, mean_1, ..., mean_5],
  "scale": [std_0,  std_1,  ..., std_5]
}


                                                 OVERALL PIPELINE

Inside ContinuousIntervalLattice.__call__(), each sample receives:

    sample.l

where sample.l is the transformed 1x6 lattice vector. If standardization is
enabled, sample.l is also standardized as:

    sample.l = (sample.l - loc) / scale

During sampling or evaluation, this transform can be inverted by first
unstandardizing by recalling the JSON of statistics, then applying:

    lengths = exp(log_lengths)
    angles  = atan(angle_features) + pi/2

"""

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


def lattice_feature_vector(cell: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """

    Output:
        Tensor of shape (6,):

            [
                log(a),
                log(b),
                log(c),
                tan(alpha - pi/2),
                tan(beta  - pi/2),
                tan(gamma - pi/2),
            ]

    """
    lengths, angles = cell_lengths_and_angles(cell)

    log_lengths = torch.log(lengths.clamp_min(eps))
    angle_features = torch.tan(angles - torch.pi / 2.0)

    return torch.cat([log_lengths, angle_features], dim=0)


def ensure_lattice_standardization_cache(
    *,
    cache_file: str | Path,
    processed_dir: str | Path,
    eps: float = 1e-8,
) -> Path:
    """Create train-set mean/std statistics for lattice features.

    Written JSON format:
        {
            "loc":   [mean_0, ..., mean_5],
            "scale": [std_0,  ..., std_5]
        }

    Use case:
        For x0 lattice parameterization, the model predicts standardized
        lattice features. The transform must therefore know the train-set
        mean and standard deviation.
    """
    cache_path = Path(cache_file)
    if cache_path.exists():
        return cache_path

    cell_path = Path(processed_dir) / "cell.npy"
    cells = np.load(cell_path, allow_pickle=True)

    features = []
    for cell in cells:
        cell = torch.as_tensor(cell, dtype=torch.get_default_dtype())

        # MatterGen may store a cell as shape (1, 3, 3).
        # The transform expects shape (3, 3).
        if cell.ndim == 3 and cell.shape[0] == 1:
            cell = cell.squeeze(0)

        features.append(lattice_feature_vector(cell, eps=eps))

    stacked = torch.stack(features, dim=0)

    loc = stacked.mean(dim=0)
    scale = stacked.std(dim=0).clamp_min(eps)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"loc": loc.tolist(), "scale": scale.tolist()},
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

    standardization is x0:
        l_standardized = (l - loc) / scale

    Inverse transform:
        l -> unstandardize -> lengths and angles

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

        self.loc: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

        if self.standardize and self.cache_file is not None and self.cache_file.exists():
            with self.cache_file.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)

            self.loc = torch.tensor(stats["loc"], dtype=torch.get_default_dtype())
            self.scale = torch.tensor(stats["scale"], dtype=torch.get_default_dtype())

    def _move_stats_to(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        loc = self.loc.to(device=value.device, dtype=value.dtype)
        scale = self.scale.to(device=value.device, dtype=value.dtype).clamp_min(self.eps)

        # Make loc and scale broadcastable to value.
        while loc.ndim < value.ndim:
            loc = loc.unsqueeze(0)
            scale = scale.unsqueeze(0)

        return loc, scale

    def standardize_value(self, value: torch.Tensor) -> torch.Tensor:
        """
        Standardize lattice features.

        Output:
            If standardization is enabled:
                (value - loc) / scale

            Otherwise:
                value unchanged.
        """
        if not self.standardize or self.loc is None or self.scale is None:
            return value

        loc, scale = self._move_stats_to(value)
        return (value - loc) / scale

    def unstandardize(self, value: torch.Tensor) -> torch.Tensor:
        """Undo lattice standardization.

        Input:
            value:
                Standardized lattice features.

        Output:
            Physical lattice features in the unconstrained representation:
                [log lengths, angle features]
        """
        if not self.standardize or self.loc is None or self.scale is None:
            return value

        loc, scale = self._move_stats_to(value)
        return value * scale + loc

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

        features = lattice_feature_vector(cell, eps=self.eps)
        features = self.standardize_value(features)

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
        del num_atoms

        features = self.unstandardize(l)

        log_lengths = features[..., :3]
        angle_features = features[..., 3:]

        lengths = torch.exp(log_lengths)
        angles = torch.atan(angle_features) + torch.pi / 2.0

        return lengths, angles
