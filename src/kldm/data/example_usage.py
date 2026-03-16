"""Example usage matching the current KLDM-style data flow."""
from __future__ import annotations

from pathlib import Path

import torch
import torch_geometric.transforms as T

from kldm.data import (
    CSPDataModule,
    ContinuousIntervalAngles,
    ContinuousIntervalLengths,
    DataModule,
    FullyConnectedGraph,
    KLDMState,
)


def build_denovo_transform() -> T.Compose:
    return T.Compose(
        [
            FullyConnectedGraph(),
            ContinuousIntervalLengths(in_key="lengths", out_key="lengths_cont"),
            ContinuousIntervalAngles(in_key="angles", out_key="angles_cont"),
            KLDMState(atom_values=list(range(1, 119))),
        ]
    )


def build_csp_transform() -> T.Compose:
    return T.Compose(
        [
            FullyConnectedGraph(),
            KLDMState(atom_values=list(range(1, 119))),
        ]
    )


def _task_ids(task: str, num_graphs: int) -> torch.Tensor:
    task_to_id = {"dng": 0, "denovo": 0, "csp": 1}
    return torch.full((num_graphs,), task_to_id[task], dtype=torch.long)


def _print_batch(name: str, task: str, batch) -> None:
    t_graph = torch.rand(batch.num_graphs, dtype=torch.float32)
    t_nodes = t_graph[batch.batch]

    print(f"{name}:")
    print(f"  graphs={batch.num_graphs} total_nodes={batch.num_nodes}")
    print(f"  base fields: pos={tuple(batch.pos.shape)} h={tuple(batch.h.shape)} batch={tuple(batch.batch.shape)}")
    if hasattr(batch, "lengths"):
        lattice_line = (
            f"  lattice fields: lengths={tuple(batch.lengths.shape)} angles={tuple(batch.angles.shape)}"
        )
        if hasattr(batch, "lengths_cont") and hasattr(batch, "angles_cont"):
            lattice_line += (
                f" lengths_cont={tuple(batch.lengths_cont.shape)} angles_cont={tuple(batch.angles_cont.shape)}"
            )
        print(lattice_line)
    print(
        "  algorithm-1 fields:"
        f" f0={tuple(batch.f0.shape)} v0={tuple(batch.v0.shape)}"
        f" l0={tuple(batch.l0.shape) if hasattr(batch, 'l0') else None}"
        f" a0={tuple(batch.a0.shape)}"
    )
    print(
        "  forward-ready:"
        f" task_ids={_task_ids(task, batch.num_graphs).tolist()}"
        f" graph_t_shape={tuple(t_graph.shape)} node_t_shape={tuple(t_nodes.shape)}"
        f" edge_node_index={tuple(batch.edge_node_index.shape)}"
    )


def denovo_example() -> None:
    datamodule = DataModule(
        transform=build_denovo_transform(),
        train_path=Path("data/mp_20/train.pt"),
        val_path=Path("data/mp_20/val.pt"),
        test_path=Path("data/mp_20/test.pt"),
        train_batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        num_val_subset=8,
        num_test_subset=8,
        num_workers=0,
        pin_memory=False,
        subset_seed=42,
    )
    batch = next(iter(datamodule.train_dataloader()))
    _print_batch("De-novo batch", "dng", batch)


def csp_example() -> None:
    datamodule = CSPDataModule(
        formulas=["LiFePO4", "SiO2", "NaCl"],
        batch_size=4,
        n_samples_per_formula=2,
        transform=build_csp_transform(),
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(datamodule.predict_dataloader()))
    _print_batch("CSP batch", "csp", batch)


def main() -> None:
    denovo_example()
    print()
    csp_example()


if __name__ == "__main__":
    main()
