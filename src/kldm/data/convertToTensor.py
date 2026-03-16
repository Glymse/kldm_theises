from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import typer
from pymatgen.core import Structure
from torch_geometric.data import Data

from kldm.data.utils import write_json


def sample_to_data(sample: Data | dict[str, np.ndarray]) -> Data:
	"""Normalize saved MP-20 samples into a PyG Data object."""
	if isinstance(sample, Data):
		return sample

	return Data(
		pos=torch.as_tensor(sample["pos"], dtype=torch.float32),
		h=torch.as_tensor(sample["h"], dtype=torch.long),
		lengths=torch.as_tensor(sample["lengths"], dtype=torch.float32).view(1, 3),
		angles=torch.as_tensor(sample["angles"], dtype=torch.float32).view(1, 3),
	)


def process_cif(cif_text: str) -> Data:
	"""Convert one CIF string into the clean crystal state used by KLDM.

	We keep fractional coordinates because the momentum diffusion in KLDM/TDM
	operates on wrapped crystal coordinates rather than Cartesian positions.
	"""
	structure = Structure.from_str(cif_text, fmt="cif").get_reduced_structure()

	pos = torch.tensor(structure.frac_coords, dtype=torch.float32)
	h = torch.tensor(structure.atomic_numbers, dtype=torch.long)
	lengths = torch.tensor(structure.lattice.abc, dtype=torch.float32).view(1, 3)
	angles = torch.tensor(structure.lattice.angles, dtype=torch.float32).view(1, 3)

	return Data(pos=pos, h=h, lengths=lengths, angles=angles)


def preprocess_csv(
	csv_folder: str | Path,
	output_folder: str | Path | None = None,
	splits: Iterable[str] = ("train", "val", "test"),
    fmt: str = "pyg",
    max_atoms: int = -1,
) -> dict[str, int]:
	"""Load MP-20 CSVs and save each split as tensor dataset (`.pt`).

	`fmt='pyg'` stores PyG Data objects, `fmt='numpy'` stores dicts of arrays.
	"""
	if fmt not in {"pyg", "numpy"}:
		raise ValueError("fmt must be one of: 'pyg', 'numpy'")

	csv_folder = Path(csv_folder)
	output_folder = Path(output_folder) if output_folder is not None else csv_folder
	output_folder.mkdir(parents=True, exist_ok=True)

	summary: dict[str, int] = {}

	for split in splits:
		csv_path = csv_folder / f"{split}.csv"
		if not csv_path.exists():
			continue

		df = pd.read_csv(csv_path)
		if "cif" not in df.columns:
			raise ValueError(f"Expected 'cif' column in {csv_path}")

		data_list: list[Data | dict[str, np.ndarray]] = []
		lengths_by_natoms: dict[int, list[np.ndarray]] = {}

		for cif_text in df["cif"].tolist():
			sample = process_cif(cif_text)
			n_atoms = int(sample.pos.shape[0])
			if 0 < max_atoms < n_atoms:
				continue

			# The original KLDM preprocessing stores length statistics per atom count
			# for later lattice normalization.
			lengths_by_natoms.setdefault(n_atoms, []).append(sample.lengths.view(-1).numpy())

			if fmt == "pyg":
				data_list.append(sample)
			else:
				data_list.append(
					{
						"pos": sample.pos.numpy(),
						"h": sample.h.numpy(),
						"lengths": sample.lengths.view(-1).numpy(),
						"angles": sample.angles.view(-1).numpy(),
					}
				)

		torch.save(data_list, output_folder / f"{split}.pt")

		loc_scale: dict[int, list[list[float]]] = {}
		for n_atoms, lengths_list in lengths_by_natoms.items():
			log_lengths = np.log(np.asarray(lengths_list, dtype=np.float64))
			loc = np.mean(log_lengths, axis=0).tolist()
			scale = np.std(log_lengths, axis=0).tolist()
			loc_scale[int(n_atoms)] = [loc, scale]

		write_json(output_folder / f"{split}_loc_scale.json", loc_scale)
		summary[split] = len(data_list)

	return summary


def preprocess(
	data_path: Path = Path("data/mp_20"),
	output_folder: Path = Path("data/mp_20"),
) -> dict[str, int]:
	"""Convenience wrapper for MP-20 conversion to tensors."""
	return preprocess_csv(csv_folder=data_path, output_folder=output_folder, fmt="pyg")


def ensure_mp20_tensors(data_dir: str | Path, splits: Iterable[str] = ("train", "val", "test")) -> None:
	"""Create `.pt` MP-20 splits if only CSV files are present."""
	data_dir = Path(data_dir)
	missing = [split for split in splits if not (data_dir / f"{split}.pt").exists()]
	if missing:
		preprocess_csv(csv_folder=data_dir, output_folder=data_dir, splits=missing, fmt="pyg")


def load_mp20_split(data_dir: str | Path, split: str) -> list[Data]:
	"""Load one MP-20 tensor split, preprocessing from CSV if needed."""
	data_dir = Path(data_dir)
	ensure_mp20_tensors(data_dir, splits=(split,))
	path = data_dir / f"{split}.pt"
	raw = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(raw, list):
		raise TypeError(f"Expected list of samples in {path}, got {type(raw)!r}")
	return [sample_to_data(sample) for sample in raw]



def main(data_dir: Path = Path("data/mp_20")) -> None:
	"""Small example: convert MP-20 CSVs and print pandas schema."""
	summary = preprocess(data_path=data_dir, output_folder=data_dir)
	print(f"Saved tensor datasets to: {data_dir}")
	print(f"Conversion summary: {summary}")



if __name__ == "__main__":
	typer.run(main)
