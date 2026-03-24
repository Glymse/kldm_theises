"""
Utility functions for KLDM.

This module provides:
- Configuration management (YAML/JSON)
- Logging setup
- Device/model utilities
- Visualization helpers
"""

from __future__ import annotations

from typing import Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import yaml

import torch
from torch import nn
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


# ============================================================================
# Device & Model Utilities
# ============================================================================

def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available in this PyTorch build. Falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU")

    return device


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def log_model_info(model: nn.Module, name: str = "Model"):
    """Log model architecture and parameters."""
    logger.info(f"\n{name} Architecture:")
    logger.info("=" * 80)
    logger.info(model)

    params = count_parameters(model)
    logger.info(f"\nParameters:")
    logger.info(f"  Total: {params['total']:,}")
    logger.info(f"  Trainable: {params['trainable']:,}")
    logger.info(f"  Frozen: {params['frozen']:,}")


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")



def batch_to_device(batch: Data | Batch, device: torch.device) -> Data | Batch:
    """Move batch to device."""
    if isinstance(batch, Batch):
        batch = batch.to(device)
    else:
        batch = batch.to(device)
    return batch


def get_batch_properties(batch: Data | Batch) -> dict[str, Any]:
    """Get batch statistics."""
    props = {}

    if hasattr(batch, "pos"):
        props["n_atoms"] = batch.pos.shape[0]

    if hasattr(batch, "num_graphs"):
        props["n_graphs"] = batch.num_graphs

    if hasattr(batch, "lengths"):
        props["avg_lattice_param"] = batch.lengths.mean().item()

    if hasattr(batch, "h"):
        unique_elements = torch.unique(batch.h)
        props["unique_elements"] = unique_elements.tolist()

    return props


def get_structure_summary(struct: Data) -> str:
    """Get summary of structure properties."""
    summary = []

    if hasattr(struct, "pos"):
        summary.append(f"  Atoms: {struct.pos.shape[0]}")

    if hasattr(struct, "h"):
        elements, counts = torch.unique(struct.h, return_counts=True)
        formula = "".join(f"{int(e)}{int(c)}" for e, c in zip(elements, counts))
        summary.append(f"  Composition: {formula}")

    if hasattr(struct, "lengths"):
        a, b, c = struct.lengths.squeeze()
        summary.append(f"  Lattice: a={a:.2f} b={b:.2f} c={c:.2f}")

    if hasattr(struct, "angles"):
        alpha, beta, gamma = struct.angles.squeeze()
        summary.append(f"  Angles: α={alpha:.1f}° β={beta:.1f}° γ={gamma:.1f}°")

    return "\n".join(summary)
