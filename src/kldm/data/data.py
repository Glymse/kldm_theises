from __future__ import annotations

from pathlib import Path
import importlib
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
import typer
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data

def _resolve_save_json():
    for module_name in ("src_kldm.data.utils", "kldm.data.utils"):
        try:
            module = importlib.import_module(module_name)
            return module.save_json
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Could not import save_json from src_kldm.data.utils or kldm.data.utils")


save_json = _resolve_save_json()


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Convert lattice lengths/angles to a 3x3 matrix."""

    def cap_abs(value: float, max_abs: float = 1.0) -> float:
        return max(min(value, max_abs), -max_abs)

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    value = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    gamma_star = np.arccos(cap_abs(value))

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c], dtype=np.float64)


def process_cif(crystal_str: str) -> dict[str, np.ndarray]:
    """Parse CIF and return canonicalized crystal features."""
    crystal = Structure.from_str(crystal_str, fmt="cif").get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )

    lengths = np.array(canonical_crystal.lattice.parameters[:3], dtype=np.float32)
    angles = np.array(canonical_crystal.lattice.parameters[3:], dtype=np.float32)

    # Different but equivalent lattice conventions can produce matrix forms
    # that are not numerically identical entry-by-entry.
    _ = np.allclose(
        canonical_crystal.lattice.matrix,
        lattice_params_to_matrix(*lengths, *angles),
    )

    return {
        "pos": canonical_crystal.frac_coords.astype(np.float32),
        "h": np.array(canonical_crystal.atomic_numbers, dtype=np.int64),
        "lengths": lengths,
        "angles": angles,
    }


def _trimmed_loc_scale(lengths_log: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute robust mean/std over central 95% in log-space."""
    sorted_lengths = np.sort(lengths_log, axis=0)
    trim_idx = int(len(sorted_lengths) * 0.025)

    if trim_idx > 0 and (2 * trim_idx) < len(sorted_lengths):
        core = sorted_lengths[trim_idx:-trim_idx, :]
    else:
        core = sorted_lengths

    loc = np.mean(core, axis=0)
    scale = np.std(core, axis=0)
    return loc.tolist(), scale.tolist()


def preprocess_csv(
    csv_folder: str | Path,
    output_folder: str | Path | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
    fmt: Literal["pyg", "numpy"] = "pyg",
    max_atoms: int = -1,
) -> None:
    """Preprocess MP-20 CSVs into torch files and per-split length stats."""
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
            data = process_cif(cif)

            n_atoms = int(data["pos"].shape[0])
            if 0 < max_atoms < n_atoms:
                continue

            lengths_by_natoms.setdefault(n_atoms, []).append(data["lengths"])

            if fmt == "pyg":
                data_obj = Data(
                    pos=torch.tensor(data["pos"], dtype=torch.float32),
                    h=torch.tensor(data["h"], dtype=torch.long),
                    lengths=torch.tensor(data["lengths"], dtype=torch.float32).view(1, -1),
                    angles=torch.tensor(data["angles"], dtype=torch.float32).view(1, -1),
                )
                data_list.append(data_obj)
            else:
                data_list.append(data)

        split_pt = output_folder / f"{split}.pt"
        torch.save(data_list, split_pt)
        print(f"Preprocessed '{split}' split saved in {split_pt}")

        loc_scale: dict[int, tuple[list[float], list[float]]] = {}
        for n_atoms, lengths_list in lengths_by_natoms.items():
            lengths_log = np.log(np.array(lengths_list, dtype=np.float64))
            loc_scale[n_atoms] = _trimmed_loc_scale(lengths_log)

        split_stats = output_folder / f"{split}_loc_scale.json"
        save_json(json_dict=loc_scale, json_path=split_stats, sort_keys=True)
        print(f"Precomputed stats for '{split}' split saved in {split_stats}")


class MyDataset(Dataset):
    """Load one KLDM-preprocessed split, e.g. data/mp_20/train.pt."""

    def __init__(self, data_path: str | Path, transform: Optional[Callable[[Data], Data]] = None) -> None:
        self.data_path = Path(data_path)
        self.transform = transform
        self.data = self.load(self.data_path)

    @staticmethod
    def load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]

        if not isinstance(data, Data):
            data = Data(
                pos=torch.tensor(data["pos"], dtype=torch.float32),
                h=torch.tensor(data["h"], dtype=torch.long),
                lengths=torch.tensor(data["lengths"], dtype=torch.float32).view(1, -1),
                angles=torch.tensor(data["angles"], dtype=torch.float32).view(1, -1),
            )

        if self.transform is not None:
            data = self.transform(data)

        return data

def preprocess(
    data_path: Path = Path("data/mp_20"),
    output_folder: Path = Path("data/mp_20"),
) -> None:
    print("Preprocessing data...")
    preprocess_csv(
        csv_folder=Path(data_path),
        output_folder=Path(output_folder),
        splits=("train", "val", "test"),
        fmt="pyg",
    )


if __name__ == "__main__":
    typer.run(preprocess)
