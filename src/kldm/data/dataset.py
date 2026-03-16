from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch, Data


class Dataset(TorchDataset, ABC):
    """Abstract KLDM-ready graph dataset.

    Concrete datasets should only define how a raw sample is created.
    The base class adds the graph fields needed by KLDM-style diffusion:
    - `edge_node_index`
    - `l`
    """

    def __init__(
        self,
        transform: Optional[Callable[[Data], Data]] = None,
        default_lattice: Optional[torch.Tensor] = None,
    ) -> None:
        self.transform = transform
        self.default_lattice = (
            torch.zeros(1, 6, dtype=torch.float32)
            if default_lattice is None
            else default_lattice.view(1, 6).to(dtype=torch.float32)
        )

    @staticmethod
    def collate_fn(samples: list[Data]) -> Batch:
        return Batch.from_data_list(samples)

    @staticmethod
    def build_complete_graph(num_nodes: int) -> torch.Tensor:
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        return mask.nonzero(as_tuple=False).t().contiguous()

    def apply_transform(self, sample: Data) -> Data:
        return sample if self.transform is None else self.transform(sample)

    def attach_graph(self, sample: Data) -> Data:
        if not hasattr(sample, "edge_node_index"):
            # KLDM uses a node graph in the network forward pass even though the
            # diffusion state itself is stored in `pos`, `h`, and `l`.
            sample.edge_node_index = self.build_complete_graph(sample.pos.shape[0])

        if hasattr(sample, "l"):
            sample.l = sample.l.view(1, 6).to(dtype=torch.float32)
        elif hasattr(sample, "lengths") and hasattr(sample, "angles"):
            # The lattice diffusion sees one graph-level 6D vector per crystal.
            sample.l = torch.cat([sample.lengths, sample.angles], dim=-1).to(dtype=torch.float32)
        else:
            # CSP does not start from an observed lattice, so the task dataset
            # provides only composition/graph structure and the lattice is left
            # to the model prior unless a downstream transform overrides it.
            sample.l = self.default_lattice.clone()

        return sample

    @abstractmethod
    def _get_raw_sample(self, idx: int) -> Data:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Data:
        sample = self._get_raw_sample(idx)
        sample = self.attach_graph(sample)
        return self.apply_transform(sample)
