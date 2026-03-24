from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import requests
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CrystalDataset, CrystalDatasetBuilder, DatasetTransform
from mattergen.common.data.transform import Transform
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS
from pymatgen.symmetry.groups import SpaceGroup
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data"


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Resolve the dataset root.

    If no root is provided, always use the workspace-level `data/` directory
    instead of a path relative to the caller's current working directory.
    """
    return DEFAULT_DATA_ROOT if root is None else Path(root).expanduser()


# Inspired by torchvision-style dataset wrappers: keep a small class responsible for
# download, cache discovery, and handing off to MatterGen's builder.
class CrystalDatasetWrapper(Dataset):
    """Dataset class for loading MatterGen-backed crystal structures."""

    dataset_name: str
    url: str
    properties_map: dict[str, str]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: list[Transform] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
        download: bool = False,
    ) -> None:
        if not isinstance(split, str) or split not in ["train", "val", "test"]:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        self.root = Path(root).expanduser()
        self.split = split
        self.transforms = transforms if transforms is not None else []
        self.dataset_transforms = dataset_transforms if dataset_transforms is not None else []

        if download:
            self.download()

        if not self._check_exists_raw():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._df_raw, self.properties = self._prepare_df()
        self.data: CrystalDataset = self._build()

    @staticmethod
    def collate_fn(samples: list[ChemGraph]) -> Batch:
        return Batch.from_data_list(samples)

    def _prepare_df(self) -> tuple[pd.DataFrame, list[str]]:
        df = pd.read_csv(self.raw_folder / f"{self.split}.csv")
        df = df.rename(columns=self.properties_map)

        space_group_map = {
            i: SpaceGroup.from_int_number(i).symbol
            for i in range(1, len(SpaceGroup.full_sg_mapping) + 1)
        }
        if "space_group" in df.columns:
            df["space_group"] = df["space_group"].map(space_group_map)

        properties = list(set(df.columns) & set(PROPERTY_SOURCE_IDS))
        return df, properties

    def _build(self) -> CrystalDataset:
        processed_path = self.processed_folder / f"{self.split}"

        if not self._check_exists_processed():
            if processed_path.exists():
                shutil.rmtree(processed_path)
            self.processed_folder.mkdir(parents=True, exist_ok=True)
            builder = CrystalDatasetBuilder.from_csv(
                csv_path=str(self.raw_folder / f"{self.split}.csv"),
                cache_path=str(processed_path),
                transforms=self.transforms,
            )

            for prop in self.properties:
                if prop not in builder.property_names:
                    values = self._df_raw[prop].to_numpy()
                    data_dict = dict(zip(builder.structure_id, values, strict=False))
                    builder.add_property_to_cache(prop, data_dict)
        else:
            builder = CrystalDatasetBuilder.from_cache_path(
                cache_path=str(processed_path),
                transforms=self.transforms,
                properties=self.properties,
            )

        return builder.build(dataset_class=CrystalDataset, dataset_transforms=self.dataset_transforms)

    def __getitem__(self, index: int) -> ChemGraph:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def df(self) -> pd.DataFrame:
        if len(self.data) != len(self._df_raw):
            mask = self._df_raw["material_id"].isin(self.data.structure_id)
            return self._df_raw[mask].reset_index(drop=True)
        return self._df_raw

    @property
    def raw_folder(self) -> Path:
        return Path(self.root, self.dataset_name, "raw")

    @property
    def processed_folder(self) -> Path:
        return Path(self.root, self.dataset_name, "processed")

    def _check_exists_raw(self) -> bool:
        return (self.raw_folder / f"{self.split}.csv").exists()

    def _check_exists_processed(self) -> bool:
        processed_path = self.processed_folder / f"{self.split}"
        if not processed_path.exists():
            return False

        required_arrays = [
            "atomic_numbers.npy",
            "cell.npy",
            "num_atoms.npy",
            "pos.npy",
            "structure_id.npy",
        ]
        if any(not (processed_path / filename).exists() for filename in required_arrays):
            return False

        required_properties = [f"{prop}.json" for prop in self.properties]
        if any(not (processed_path / filename).exists() for filename in required_properties):
            return False

        return True

    def download(self) -> None:
        if self._check_exists_raw():
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)
        response = requests.get(url=self.url + f"{self.split}.csv", stream=True, timeout=40)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        output_file = self.raw_folder / f"{self.split}.csv"

        with (
            output_file.open("wb") as handle,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {self.dataset_name} {self.split} dataset") as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    pbar.update(len(chunk))

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of samples: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        if len(self.transforms) > 0:
            body.append(f"Transforms: {self.transforms}")
        if len(self.dataset_transforms) > 0:
            body.append(f"Dataset Transforms: {self.dataset_transforms}")
        return "\n".join([head] + [" " * 4 + line for line in body])


class Carbon24(CrystalDatasetWrapper):
    dataset_name = "carbon_24"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/carbon_24/"
    properties_map = {
        "energy_per_atom": "formation_energy_per_atom",
        "spacegroup.number": "space_group",
    }


class MP20(CrystalDatasetWrapper):
    dataset_name = "mp_20"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/"
    properties_map = {
        "formation_energy_per_atom": "formation_energy_per_atom",
        "band_gap": "dft_band_gap",
        "e_above_hull": "energy_above_hull",
        "spacegroup.number": "space_group",
    }


class MPTS52(CrystalDatasetWrapper):
    dataset_name = "mpts_52"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mpts_52/"
    properties_map = {
        "energy_above_hull": "energy_above_hull",
        "formation_energy_per_atom": "formation_energy_per_atom",
    }


class Perov5(CrystalDatasetWrapper):
    dataset_name = "perov_5"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/"
    properties_map = {
        "heat_all": "formation_energy_per_atom",
        "ind_gap": "dft_band_gap",
        "spacegroup.number": "space_group",
    }
