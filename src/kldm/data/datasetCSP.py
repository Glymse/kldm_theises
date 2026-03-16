from __future__ import annotations

from typing import Callable, Optional, Sequence

import chemparse
import torch
from ase.data import atomic_numbers
from torch_geometric.data import Data

from kldm.data.dataset import Dataset


class DatasetCSP(Dataset):
    """Formula-conditioned dataset used for CSP diffusion.

    This task only fixes composition. Positions are placeholders so the batch
    still satisfies the KLDM interface before the reverse process samples a
    crystal structure.

    In other words, this dataset is suitable as a CSP conditioning container
    for sampling, not as a supervised source of clean crystal targets.
    """

    def __init__(
        self,
        formulas: Sequence[str],
        n_samples_per_formula: int = 1,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        super().__init__(transform=transform)
        self.parsed_formulas: list[dict[str, float]] = []
        for formula in formulas:
            parsed = chemparse.parse_formula(formula)
            for _ in range(int(n_samples_per_formula)):
                self.parsed_formulas.append(parsed)

    def __len__(self) -> int:
        return len(self.parsed_formulas)

    def _get_raw_sample(self, idx: int) -> Data:
        formula = self.parsed_formulas[idx]
        atom_numbers = [
            atomic_numbers[symbol]
            for symbol, count in formula.items()
            for _ in range(int(count))
        ]
        h = torch.tensor(atom_numbers, dtype=torch.long)
        # CSP starts from composition only, matching the original KLDM setup.
        return Data(pos=torch.randn(h.shape[0], 3), h=h)


SampleDatasetCSP = DatasetCSP
