from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Data

from kldm.data.data import preprocess_csv
from kldm.data.dataset import Dataset


class MyDataset(Dataset):
    """Load one KLDM-preprocessed split from MP-20."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        super().__init__(path=str(self.data_path))

    def __getitem__(self, index: int) -> Data:
        return super().__getitem__(index)

    @classmethod
    def from_split(cls, data_folder: str | Path, split: str) -> "MyDataset":
        return cls(Path(data_folder) / f"{split}.pt")

    @staticmethod
    def preprocess(data_path: str | Path, output_folder: str | Path | None = None) -> None:
        data_path = Path(data_path)
        output_folder = data_path if output_folder is None else Path(output_folder)
        if output_folder != data_path:
            raise ValueError(
                "Preprocessing writes outputs next to the CSV files, so output_folder must match data_path."
            )
        preprocess_csv(
            csv_folder=output_folder,
            output_folder=output_folder,
            splits=("train", "val", "test"),
            fmt="pyg",
        )
