from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from kldm.data import DatasetCSP, DatasetDNG


def describe(name: str, batch: Batch | Data) -> None:
    num_graphs = batch.num_graphs if isinstance(batch, Batch) else 1
    batch_index = (
        batch.batch
        if getattr(batch, "batch", None) is not None
        else torch.zeros(batch.pos.shape[0], dtype=torch.long)
    )
    graph_t = torch.rand(num_graphs)
    node_t = graph_t[batch_index]

    print(name)
    print(f"  pos={tuple(batch.pos.shape)}")
    print(f"  h={tuple(batch.h.shape)}")
    print(f"  l={tuple(batch.l.shape)}")
    print(f"  edge_node_index={tuple(batch.edge_node_index.shape)}")
    print(f"  batch={tuple(batch_index.shape)}")
    print(f"  graph_t={tuple(graph_t.shape)} node_t={tuple(node_t.shape)}")


def main() -> None:
    torch.manual_seed(0)

    dng = DatasetDNG(path=Path("data/mp_20/train.pt"))
    describe("DNG sample", dng[0])
    describe("DNG batch", Batch.from_data_list([dng[0], dng[1]]))

    print()

    csp = DatasetCSP(formulas=["SiO2", "LiFePO4"], n_samples_per_formula=2)
    describe("CSP sample", csp[0])
    describe("CSP batch", Batch.from_data_list([csp[0], csp[1]]))


if __name__ == "__main__":
    main()
