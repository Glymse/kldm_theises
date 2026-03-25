from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from kldm.data import CSPTask, DNGTask, MP20, resolve_data_root


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _tensor_preview(name: str, tensor: torch.Tensor, n: int = 3) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}")
    if tensor.ndim == 0:
        print(tensor)
    elif tensor.ndim == 1:
        print(tensor[: min(n, tensor.shape[0])])
    else:
        print(tensor[: min(n, tensor.shape[0])])


def _tensor_stats(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    print(
        f"{name} stats: min={tensor.min().item():.4f} "
        f"max={tensor.max().item():.4f} mean={tensor.mean().item():.4f}"
    )


def main() -> None:
    root = resolve_data_root()
    root.mkdir(parents=True, exist_ok=True)

    _print_header("MatterGen / MP-20 Dataset")
    dataset = MP20(root=root, split="train", download=True)
    print(f"root: {root}")
    print(f"len(dataset): {len(dataset)}")
    print(f"raw folder: {dataset.raw_folder}")
    print(f"processed folder: {dataset.processed_folder}")

    sample = dataset[0]
    print(f"sample type: {type(sample)}")
    print(f"available keys: {sorted(sample.keys())}")
    _tensor_preview("pos", sample.pos)
    _tensor_stats("pos", sample.pos)
    _tensor_preview("atomic_numbers", sample.atomic_numbers)
    print("first atomic numbers:", sample.atomic_numbers[:10].tolist())
    _tensor_preview("cell", sample.cell)
    _tensor_stats("cell", sample.cell)
    print("pos should be fractional coordinates in [0, 1] for MP-20 style inputs.")

    _print_header("CSP Example")
    csp_task = CSPTask()
    csp_loader = csp_task.dataloader(root=root, split="train", batch_size=2, shuffle=False, download=True)
    csp_batch = next(iter(csp_loader))
    print(f"batch type: {type(csp_batch)}")
    _tensor_preview("csp_batch.pos", csp_batch.pos)
    _tensor_stats("csp_batch.pos", csp_batch.pos)
    _tensor_preview("csp_batch.h", csp_batch.h)
    print("csp_batch.h contains integer atomic numbers.")
    print("first CSP atomic numbers:", csp_batch.h[:10].tolist())
    _tensor_preview("csp_batch.lengths", csp_batch.lengths)
    _tensor_preview("csp_batch.angles", csp_batch.angles)
    _tensor_preview("csp_batch.l", csp_batch.l)
    print("first CSP lattice vector:", csp_batch.l[0].tolist())
    _tensor_preview("csp_batch.edge_node_index", csp_batch.edge_node_index)
    _tensor_preview("csp_batch.task_id", csp_batch.task_id)
    _tensor_preview("csp_batch.diffuse_h", csp_batch.diffuse_h)

    _print_header("DNG Example")
    dng_task = DNGTask()
    dng_loader = dng_task.dataloader(root=root, split="train", batch_size=2, shuffle=False, download=True)
    dng_batch = next(iter(dng_loader))
    print(f"batch type: {type(dng_batch)}")
    _tensor_preview("dng_batch.pos", dng_batch.pos)
    _tensor_stats("dng_batch.pos", dng_batch.pos)
    _tensor_preview("dng_batch.h", dng_batch.h)
    print("dng_batch.h contains one-hot encoded atom features.")
    print("first DNG one-hot row argmax:", int(dng_batch.h[0].argmax().item()))
    print("first DNG one-hot row sum   :", float(dng_batch.h[0].sum().item()))
    _tensor_preview("dng_batch.lengths", dng_batch.lengths)
    _tensor_preview("dng_batch.angles", dng_batch.angles)
    _tensor_preview("dng_batch.l", dng_batch.l)
    print("first DNG lattice vector:", dng_batch.l[0].tolist())
    _tensor_preview("dng_batch.edge_node_index", dng_batch.edge_node_index)
    _tensor_preview("dng_batch.task_id", dng_batch.task_id)
    _tensor_preview("dng_batch.diffuse_h", dng_batch.diffuse_h)

    print("\nDone.")


if __name__ == "__main__":
    main()
