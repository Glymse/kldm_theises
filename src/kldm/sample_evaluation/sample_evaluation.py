from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from kldm.data.transform import ContinuousIntervalLattice, DEFAULT_ATOMIC_VOCAB

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = StructureMatcher = SpacegroupAnalyzer = None


@dataclass
class CSPReconstructionResult:
    valid: bool
    match: bool
    rmse: float | None
    predicted_structure: Any
    target_structure: Any
    formula: str | None = None


def _require_pymatgen() -> None:
    if Structure is None or Lattice is None or Element is None or StructureMatcher is None:
        raise ImportError("sample_evaluation requires pymatgen in the active environment.")


def _to_2d_tensor(x: torch.Tensor | list[float] | list[list[float]]) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _default_lattice_transform() -> ContinuousIntervalLattice:
    return ContinuousIntervalLattice(standardize=False)


def _standardize_structure(structure: Structure | None) -> Structure | None:
    if structure is None or SpacegroupAnalyzer is None:
        return structure
    try:
        standardized = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        return standardized if standardized is not None else structure
    except Exception:
        return structure


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def decode_atom_types(
    a: torch.Tensor | list[int] | list[list[float]],
    species_vocab: Optional[list[int]] = None,
) -> tuple[list[int], list[str]]:
    _require_pymatgen()
    species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB
    a_tensor = torch.as_tensor(a)

    if a_tensor.ndim == 1:
        atomic_numbers = [int(x) for x in a_tensor.tolist()]
    elif a_tensor.ndim == 2:
        indices = a_tensor.argmax(dim=-1).tolist()
        atomic_numbers = [int(species_vocab[int(i)]) for i in indices]
    else:
        raise ValueError(f"Expected atom tensor with ndim 1 or 2, got shape {tuple(a_tensor.shape)}")

    species = [Element.from_Z(z).symbol for z in atomic_numbers]
    return atomic_numbers, species


def decode_lattice(
    l: torch.Tensor | list[float] | list[list[float]],
    n_atoms: int,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lattice_transform = lattice_transform or _default_lattice_transform()
    l_tensor = _to_2d_tensor(l)
    lengths, angles_rad = lattice_transform.invert_to_lengths_angles(l=l_tensor, num_atoms=n_atoms)
    return lengths.squeeze(0), torch.rad2deg(angles_rad.squeeze(0))


def _angles_define_physical_cell(angles_deg: torch.Tensor) -> bool:
    alpha, beta, gamma = [float(x) for x in angles_deg.tolist()]
    return (
        0.0 < alpha < 180.0
        and 0.0 < beta < 180.0
        and 0.0 < gamma < 180.0
        and alpha + beta > gamma
        and alpha + gamma > beta
        and beta + gamma > alpha
        and alpha + beta + gamma < 360.0
    )


def build_structure_from_sample(
    f: torch.Tensor | list[list[float]],
    l: torch.Tensor | list[float] | list[list[float]],
    a: torch.Tensor | list[int] | list[list[float]],
    *,
    species_vocab: Optional[list[int]] = None,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
    sort_structure: bool = True,
) -> Structure:
    _require_pymatgen()
    lattice_transform = lattice_transform or _default_lattice_transform()

    frac_coords = _to_2d_tensor(f)
    if frac_coords.shape[-1] != 3:
        raise ValueError(f"Expected fractional coordinates with last dim 3, got shape {tuple(frac_coords.shape)}")

    n_atoms = int(frac_coords.shape[0])
    _, species = decode_atom_types(a=a, species_vocab=species_vocab)
    lengths, angles_deg = decode_lattice(l=l, n_atoms=n_atoms, lattice_transform=lattice_transform)

    if not torch.isfinite(frac_coords).all():
        raise ValueError("Predicted fractional coordinates contain non-finite values.")
    if not torch.isfinite(lengths).all() or not torch.isfinite(angles_deg).all():
        raise ValueError("Decoded lattice contains non-finite values.")
    if not (lengths > 0.0).all():
        raise ValueError("Decoded lattice contains non-positive lengths.")
    if not _angles_define_physical_cell(angles_deg):
        raise ValueError("Decoded lattice angles do not define a physical cell.")

    structure = Structure(
        lattice=Lattice.from_parameters(
            a=float(lengths[0].item()),
            b=float(lengths[1].item()),
            c=float(lengths[2].item()),
            alpha=float(angles_deg[0].item()),
            beta=float(angles_deg[1].item()),
            gamma=float(angles_deg[2].item()),
        ),
        species=species,
        coords=(frac_coords % 1.0).detach().cpu().tolist(),
        coords_are_cartesian=False,
    )
    return structure.get_sorted_structure() if sort_structure else structure


def validity_structure(
    structure: Structure,
    cutoff: float = 0.5,
    max_length: float = 40.0,
    min_volume: float = 0.1,
) -> bool:
    try:
        distance_matrix = np.asarray(structure.distance_matrix, dtype=float)
    except Exception:
        return False

    distance_matrix = distance_matrix + np.diag(np.ones(distance_matrix.shape[0]) * (cutoff + 10.0))
    return not (
        distance_matrix.min() < cutoff
        or structure.volume < min_volume
        or max(structure.lattice.abc) > max_length
    )


def prepare_visualization_pair(
    predicted_structure: Structure | None,
    target_structure: Structure | None,
    *,
    standardize: bool = True,
) -> tuple[Structure | None, Structure | None]:
    """Prepare a visualization-only structure pair.

    The reconstruction metric is computed elsewhere. For pictures/CIF export we:
    1. align the prediction to look like the target using StructureMatcher
    2. optionally standardize both structures for cleaner display

    This keeps visualization close to facit's matcher behavior while preserving
    the newer paired export flow.
    """
    if predicted_structure is None or target_structure is None:
        return predicted_structure, target_structure

    predicted_vis = predicted_structure
    target_vis = target_structure

    try:
        matcher = StructureMatcher(primitive_cell=True)
        matched_pred = matcher.get_s2_like_s1(target_vis, predicted_vis)
        if matched_pred is not None:
            predicted_vis = matched_pred
    except Exception:
        pass

    if standardize:
        predicted_vis = _standardize_structure(predicted_vis)
        target_vis = _standardize_structure(target_vis)

    return predicted_vis, target_vis


def evaluate_csp_reconstruction(
    *,
    pred_f: torch.Tensor | list[list[float]],
    pred_l: torch.Tensor | list[float] | list[list[float]],
    pred_a: torch.Tensor | list[int] | list[list[float]],
    target_f: torch.Tensor | list[list[float]],
    target_l: torch.Tensor | list[float] | list[list[float]],
    target_a: torch.Tensor | list[int] | list[list[float]],
    species_vocab: Optional[list[int]] = None,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
) -> CSPReconstructionResult:
    lattice_transform = lattice_transform or _default_lattice_transform()

    try:
        pred_structure = build_structure_from_sample(
            f=pred_f,
            l=pred_l,
            a=pred_a,
            species_vocab=species_vocab,
            lattice_transform=lattice_transform,
        )
    except Exception:
        return CSPReconstructionResult(False, False, None, None, None, None)

    try:
        target_structure = build_structure_from_sample(
            f=target_f,
            l=target_l,
            a=target_a,
            species_vocab=species_vocab,
            lattice_transform=lattice_transform,
        )
    except Exception:
        return CSPReconstructionResult(
            False,
            False,
            None,
            pred_structure,
            None,
            pred_structure.composition.formula if pred_structure is not None else None,
        )

    valid = validity_structure(pred_structure)
    match = False
    rmse = None
    if valid:
        try:
            matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
            rms = matcher.get_rms_dist(pred_structure, target_structure)
            match = rms is not None
            if rms is not None:
                rmse = float(rms[0])
        except Exception:
            match = False

    return CSPReconstructionResult(
        valid=valid,
        match=match,
        rmse=rmse,
        predicted_structure=pred_structure,
        target_structure=target_structure,
        formula=pred_structure.composition.formula if pred_structure is not None else None,
    )


def aggregate_csp_reconstruction_metrics(results: list[CSPReconstructionResult]) -> dict[str, Any]:
    if not results:
        return {"num_samples": 0, "valid": None, "match_rate": None, "rmse": None}

    valid_values = [1.0 if result.valid else 0.0 for result in results]
    match_values = [1.0 if result.match else 0.0 for result in results]
    rmses = [float(result.rmse) for result in results if result.rmse is not None]

    return {
        "num_samples": len(results),
        "valid": float(sum(valid_values) / len(valid_values)),
        "match_rate": float(sum(match_values) / len(match_values)),
        "rmse": _mean_or_none(rmses),
    }
