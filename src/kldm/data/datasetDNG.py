from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from torch_geometric.data import Data

from kldm.data.convertToTensor import load_mp20_split
from kldm.data.dataset import Dataset


class DatasetDNG(Dataset):
    """De-novo dataset backed by a real MP-20 tensor split.

    Unlike CSP, DNG trains from complete crystals, so each sample carries the
    observed clean state `(pos, h, lengths, angles)` before graph fields are
    attached by the abstract dataset.

    Atomic species stay as integer atomic numbers here. That matches KLDM when
    `diffusion_h` is discrete, but continuous/analog-bits species diffusion
    would need an extra encoding step before calling the diffusion module.
    """

    def __init__(
        self,
        path: str | Path,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        super().__init__(transform=transform)
        split_path = Path(path)
        self.data = load_mp20_split(split_path.parent, split_path.stem)

    def __len__(self) -> int:
        return len(self.data)

    def _get_raw_sample(self, idx: int) -> Data:
        return self.data[idx].clone()
