from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_json(json_path: str | Path):
    path = Path(json_path)
    with path.open(encoding="utf-8", mode="r") as file_obj:
        return json.load(file_obj)


def save_json(json_dict: dict, json_path: str | Path, sort_keys: bool = False) -> None:
    serializable: dict = {}
    for key, value in json_dict.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
            continue

        if isinstance(value, tuple):
            serializable[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            continue

        serializable[key] = value

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8", mode="w") as file_obj:
        json.dump(serializable, file_obj, sort_keys=sort_keys)
