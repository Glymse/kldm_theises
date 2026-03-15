from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from kldm.data import MyDataset
from kldm.data.data import preprocess_csv
from kldm.data.utils import read_json


def test_my_dataset():
    """Dataset should construct and follow the PyTorch Dataset interface."""
    split_path = Path("data/mp_20/train.pt")
    if not split_path.exists():
        pytest.skip("Processed split not found: data/mp_20/train.pt")

    dataset = MyDataset(split_path)
    assert isinstance(dataset, Dataset)


def test_my_dataset_loads_processed_split_shapes():
    """A processed MP-20 sample should expose the expected tensor fields/shapes."""
    split_path = Path("data/mp_20/train.pt")
    if not split_path.exists():
        pytest.skip("Processed split not found: data/processed/mp_20/train.pt")

    dataset = MyDataset(split_path)
    assert len(dataset) > 0

    sample = dataset[0]
    assert hasattr(sample, "pos")
    assert hasattr(sample, "h")
    assert hasattr(sample, "lengths")
    assert hasattr(sample, "angles")

    assert sample.pos.ndim == 2
    assert sample.pos.shape[1] == 3
    assert sample.h.ndim == 1
    assert sample.h.shape[0] == sample.pos.shape[0]
    assert tuple(sample.lengths.shape) == (1, 3)
    assert tuple(sample.angles.shape) == (1, 3)


def test_preprocess_csv_smoke(tmp_path: Path):
    """Smoke test preprocessing on a tiny subset and verify artifacts."""
    source_csv = Path("data/mp_20/train.csv")
    if not source_csv.exists():
        pytest.skip("Source MP-20 CSV not found: data/mp_20/train.csv")

    source_df = pd.read_csv(source_csv)
    if "cif" not in source_df.columns:
        pytest.skip("Source MP-20 CSV is missing required 'cif' column")

    subset_df = source_df[["cif"]].head(8)
    input_dir = tmp_path / "mini_csv"
    output_dir = tmp_path / "mini_processed"
    input_dir.mkdir(parents=True, exist_ok=True)

    subset_df.to_csv(input_dir / "train.csv", index=False)

    preprocess_csv(
        csv_folder=input_dir,
        output_folder=output_dir,
        splits=("train",),
        fmt="numpy",
        max_atoms=40,
    )

    split_pt = output_dir / "train.pt"
    split_stats = output_dir / "train_loc_scale.json"

    assert split_pt.exists()
    assert split_stats.exists()

    data_list = torch.load(split_pt, map_location="cpu", weights_only=False)
    assert isinstance(data_list, list)
    assert len(data_list) > 0

    stats = read_json(split_stats)
    assert isinstance(stats, dict)
