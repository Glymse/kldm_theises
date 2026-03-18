from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data


def process_cif(cif_text: str) -> Data:
    """Parse one CIF string into the clean crystal state used by KLDM."""
    structure = Structure.from_str(cif_text, fmt="cif").get_reduced_structure()
    return Data(
        pos=torch.tensor(structure.frac_coords, dtype=torch.float32),
        h=torch.tensor(structure.atomic_numbers, dtype=torch.long),
        lengths=torch.tensor(structure.lattice.abc, dtype=torch.float32).view(1, 3),
        angles=torch.tensor(structure.lattice.angles, dtype=torch.float32).view(1, 3),
    )


def preprocess_csv(
    csv_folder: str | Path,
    output_folder: str | Path | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
    max_atoms: int = -1,
    save_stats: bool = True,
) -> dict[str, int]:
    """Convert MP-20 CSV splits into `.pt` files and optional length statistics."""
    csv_folder = Path(csv_folder)
    output_folder = csv_folder if output_folder is None else Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for split in splits:
        csv_path = csv_folder / f"{split}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if "cif" not in df.columns:
            raise ValueError(f"Expected 'cif' column in {csv_path}")

        data_list = []
        grouped_lengths: dict[int, list[np.ndarray]] = {}
        for cif_text in df["cif"].tolist():
            sample = process_cif(cif_text)
            num_atoms = int(sample.pos.shape[0])
            if 0 < max_atoms < num_atoms:
                continue
            data_list.append(sample)
            grouped_lengths.setdefault(num_atoms, []).append(sample.lengths.view(-1).numpy())

        torch.save(data_list, output_folder / f"{split}.pt")
        if save_stats:
            stats = {
                num_atoms: _length_stats(values)
                for num_atoms, values in grouped_lengths.items()
            }
            with (output_folder / f"{split}_length_stats.json").open("w", encoding="utf-8") as fp:
                json.dump(stats, fp, sort_keys=True)
        summary[split] = len(data_list)

    return summary


def ensure_processed_splits(
    data_dir: str | Path,
    splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    """Create processed `.pt` splits and stats next to the raw MP-20 CSVs if missing."""
    data_dir = Path(data_dir)
    missing = [split for split in splits if not (data_dir / f"{split}.pt").exists()]
    if missing:
        preprocess_csv(csv_folder=data_dir, output_folder=data_dir, splits=missing)

def load_processed_split(data_dir: str | Path, split: str) -> list[Data]:
    """Load one processed MP-20 split, creating it from CSV if needed."""
    data_dir = Path(data_dir)
    ensure_processed_splits(data_dir, splits=(split,))
    path = data_dir / f"{split}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        raise TypeError(f"Expected list of `Data` objects in {path}, got {type(data)!r}")
    return data


def _length_stats(values: list[np.ndarray]) -> tuple[list[float], list[float]]:
    lengths = np.log(np.asarray(values))
    loc = lengths.mean(axis=0).tolist()
    scale = np.clip(lengths.std(axis=0), 1e-6, None).tolist()
    return loc, scale
