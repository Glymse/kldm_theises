from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from mattergen.common.data.dataset import BaseDataset, CrystalDataset
from pymatgen.core.periodic_table import Element


def filter_elements(dataset: CrystalDataset, exclude_elements: Iterable[int | str]) -> CrystalDataset:
    """Exclude structures containing any atomic number in `exclude_elements`."""
    forbidden_Z = set()
    for element in exclude_elements:
        if isinstance(element, int):
            forbidden_Z.add(element)
        elif isinstance(element, str):
            forbidden_Z.add(Element(element).Z)
        else:
            raise TypeError(f"Unsupported type in exclude_elements: {type(element)}")

    forbidden_mask = np.isin(dataset.atomic_numbers, list(forbidden_Z))

    # Sum forbidden atoms per structure
    # len(num_atoms) == number of structures
    counts_per_structure = np.add.reduceat(forbidden_mask, dataset.index_offset)
    mask = counts_per_structure == 0
    indices = np.nonzero(mask)[0]
    return dataset.subset(list(indices))


def filter_energy_above_hull(dataset: BaseDataset, threshold: float = 0.1) -> BaseDataset:
    """Filter out structures with energy above the hull greater than threshold."""
    if "energy_above_hull" not in dataset.properties:
        return dataset

    energies = dataset.properties["energy_above_hull"]
    indices = np.where(energies <= threshold)[0]
    return dataset.subset(list(indices))
