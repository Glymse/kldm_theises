from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        print(torch.backends.mps.is_available())
        return torch.device("mps")

    print("mps: ", torch.backends.mps.is_available())
    return torch.device("cpu")


def should_pin_memory(device: torch.device) -> bool:
    """Pinned host memory is only useful for CUDA transfers."""
    return device.type == "cuda"
