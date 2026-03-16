from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import typer
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

from kldm.data.utils import save_json


def process_cif(crystal_str: str) -> dict[str, np.ndarray]:
    """Parse one CIF string into the KLDM base sample representation."""
    crystal = Structure.from_str(crystal_str, fmt="cif").get_reduced_structure()
    canonical = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return {
        "pos": canonical.frac_coords.astype(np.float32),
        "h": np.array(canonical.atomic_numbers, dtype=np.int64),
        "lengths": np.array(canonical.lattice.parameters[:3], dtype=np.float32),
        "angles": np.array(canonical.lattice.parameters[3:], dtype=np.float32),
    }


def _trimmed_loc_scale(lengths_log: np.ndarray) -> tuple[list[float], list[float]]:
    sorted_lengths = np.sort(lengths_log, axis=0)
    trim_idx = int(len(sorted_lengths) * 0.025)
    if trim_idx > 0 and (2 * trim_idx) < len(sorted_lengths):
        sorted_lengths = sorted_lengths[trim_idx:-trim_idx, :]
    return np.mean(sorted_lengths, axis=0).tolist(), np.std(sorted_lengths, axis=0).tolist()


def _to_data_object(item: dict[str, np.ndarray] | Data) -> Data:
    if isinstance(item, Data):
        return item
    return Data(
        pos=torch.tensor(item["pos"], dtype=torch.float32),
        h=torch.tensor(item["h"], dtype=torch.long),
        lengths=torch.tensor(item["lengths"], dtype=torch.float32).view(1, -1),
        angles=torch.tensor(item["angles"], dtype=torch.float32).view(1, -1),
    )


def preprocess_csv(
    csv_folder: str | Path,
    output_folder: str | Path | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
    fmt: str = "pyg",
    max_atoms: int = -1,
) -> None:
    csv_folder = Path(csv_folder)
    output_folder = Path(output_folder) if output_folder is not None else csv_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    for split in splits:
        csv_path = csv_folder / f"{split}.csv"
        if not csv_path.exists():
            print(f"Did not find {csv_path}, skipping...")
            continue

        df = pd.read_csv(csv_path)
        if "cif" not in df.columns:
            raise ValueError(f"'cif' column missing in {csv_path}")

        data_list: list[Data | dict[str, np.ndarray]] = []
        lengths_by_natoms: dict[int, list[np.ndarray]] = {}

        for cif in df["cif"].tolist():
            item = process_cif(cif)
            n_atoms = int(item["pos"].shape[0])
            if 0 < max_atoms < n_atoms:
                continue
            lengths_by_natoms.setdefault(n_atoms, []).append(item["lengths"])
            data_list.append(_to_data_object(item) if fmt == "pyg" else item)

        torch.save(data_list, output_folder / f"{split}.pt")

        loc_scale = {
            n_atoms: _trimmed_loc_scale(np.log(np.asarray(lengths_list, dtype=np.float64)))
            for n_atoms, lengths_list in lengths_by_natoms.items()
        }
        save_json(loc_scale, output_folder / f"{split}_loc_scale.json", sort_keys=True)


def preprocess(
    data_path: Path = Path("data/mp_20"),
    output_folder: Path = Path("data/mp_20"),
) -> None:
    preprocess_csv(
        csv_folder=data_path,
        output_folder=output_folder,
        splits=("train", "val", "test"),
        fmt="pyg",
    )


if __name__ == "__main__":
    typer.run(preprocess)
