from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kldm.data.transform import ContinuousIntervalLattice, DEFAULT_ATOMIC_VOCAB

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = StructureMatcher = SpacegroupAnalyzer = None


TensorLike = torch.Tensor | list[float] | list[int] | list[list[float]]


@dataclass
class CSPReconstructionResult:
    valid: bool
    match: bool
    rmse: float | None
    predicted_structure: Any
    target_structure: Any
    formula: str | None = None


def _require_pymatgen() -> None:
    if None in (Element, Lattice, Structure, StructureMatcher):
        raise ImportError("sample_evaluation requires pymatgen.")


def _default_lattice_transform() -> ContinuousIntervalLattice:
    return ContinuousIntervalLattice(standardize=False)


def _to_2d_tensor(x: TensorLike) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor




def decode_atom_types(
    a: TensorLike,
    species_vocab: list[int] | None = None,
) -> tuple[list[int], list[str]]:
    _require_pymatgen()

    species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB
    atom_tensor = torch.as_tensor(a)

    if atom_tensor.ndim == 1:
        atomic_numbers = [int(value) for value in atom_tensor.tolist()]
    else:
        indices = atom_tensor.argmax(dim=-1).tolist()
        atomic_numbers = [int(species_vocab[int(index)]) for index in indices]

    species = [Element.from_Z(z).symbol for z in atomic_numbers]
    return atomic_numbers, species

# Reverse the lattice transformations
def decode_lattice(
    l: TensorLike,
    n_atoms: int,
    lattice_transform: ContinuousIntervalLattice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del n_atoms

    lattice_transform = lattice_transform or _default_lattice_transform()
    lengths, angles_rad = lattice_transform.invert_to_lengths_angles(l=_to_2d_tensor(l))
    return lengths.squeeze(0), torch.rad2deg(angles_rad.squeeze(0))


def build_structure_from_sample(
    f: TensorLike,
    l: TensorLike,
    a: TensorLike,
    *,
    species_vocab: list[int] | None = None,
    lattice_transform: ContinuousIntervalLattice | None = None,
) -> Structure:
    _require_pymatgen()

    frac_coords = _to_2d_tensor(f)
    if frac_coords.shape[-1] != 3:
        raise ValueError(
            f"Expected fractional coordinates with last dim 3, got shape {tuple(frac_coords.shape)}"
        )
    if not torch.isfinite(frac_coords).all():
        raise ValueError("Fractional coordinates contain non-finite values.")

    _, species = decode_atom_types(a=a, species_vocab=species_vocab)
    lengths, angles_deg = decode_lattice(
        l=l,
        n_atoms=int(frac_coords.shape[0]),
        lattice_transform=lattice_transform,
    )

    # Reject obviously broken lattice predictions before constructing pymatgen objects.
    if not torch.isfinite(lengths).all() or not torch.isfinite(angles_deg).all():
        raise ValueError("Decoded lattice contains non-finite values.")
    if not (lengths > 0.0).all():
        raise ValueError("Decoded lattice contains non-positive lengths.")

    structure = Structure(
        lattice=Lattice.from_parameters(
            a=float(lengths[0]),
            b=float(lengths[1]),
            c=float(lengths[2]),
            alpha=float(angles_deg[0]),
            beta=float(angles_deg[1]),
            gamma=float(angles_deg[2]),
        ),
        species=species,
        coords=(frac_coords % 1.0).detach().cpu().tolist(),
        coords_are_cartesian=False,
    )
    return structure.get_sorted_structure()


def validity_structure(structure: Structure, cutoff: float = 0.5) -> bool:
    try:
        distance_matrix = np.asarray(structure.distance_matrix, dtype=float)
    except Exception:
        return False

    # Ignore self-distances by shifting the diagonal above the cutoff.
    distance_matrix = distance_matrix + np.diag(
        np.ones(distance_matrix.shape[0]) * (cutoff + 10.0)
    )

    return not (
        distance_matrix.min() < cutoff
        or structure.volume < 0.1
        or max(structure.lattice.abc) > 40.0
    )


def prepare_visualization_pair(
    predicted_structure: Structure | None,
    target_structure: Structure | None,
) -> tuple[Structure | None, Structure | None]:
    if predicted_structure is None or target_structure is None:
        return predicted_structure, target_structure

    # Align prediction to the target for easier side-by-side viewing.
    try:
        matched = StructureMatcher().get_s2_like_s1(target_structure, predicted_structure)
        if matched is not None:
            predicted_structure = matched
    except Exception:
        pass

    if SpacegroupAnalyzer is None:
        return predicted_structure, target_structure

    # Standardize both cells for cleaner exported CIFs and ASE renders.
    try:
        standardized = SpacegroupAnalyzer(predicted_structure).get_conventional_standard_structure()
        if standardized is not None:
            predicted_structure = standardized
    except Exception:
        pass

    try:
        standardized = SpacegroupAnalyzer(target_structure).get_conventional_standard_structure()
        if standardized is not None:
            target_structure = standardized
    except Exception:
        pass

    return predicted_structure, target_structure


def evaluate_csp_reconstruction(
    *,
    pred_f: TensorLike,
    pred_l: TensorLike,
    pred_a: TensorLike,
    target_f: TensorLike,
    target_l: TensorLike,
    target_a: TensorLike,
    species_vocab: list[int] | None = None,
    lattice_transform: ContinuousIntervalLattice | None = None,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
) -> CSPReconstructionResult:
    lattice_transform = lattice_transform or _default_lattice_transform()

    try:
        predicted = build_structure_from_sample(
            pred_f,
            pred_l,
            pred_a,
            species_vocab=species_vocab,
            lattice_transform=lattice_transform,
        )
    except Exception:
        return CSPReconstructionResult(False, False, None, None, None, None)

    try:
        target = build_structure_from_sample(
            target_f,
            target_l,
            target_a,
            species_vocab=species_vocab,
            lattice_transform=lattice_transform,
        )
    except Exception:
        return CSPReconstructionResult(
            False,
            False,
            None,
            predicted,
            None,
            predicted.composition.formula,
        )

    valid = validity_structure(predicted)
    match = False
    rmse = None

    if valid:
        try:
            rms = StructureMatcher(
                stol=stol,
                angle_tol=angle_tol,
                ltol=ltol,
            ).get_rms_dist(predicted, target)
            match = rms is not None
            if rms is not None:
                rmse = float(rms[0])
        except Exception:
            pass

    return CSPReconstructionResult(
        valid=valid,
        match=match,
        rmse=rmse,
        predicted_structure=predicted,
        target_structure=target,
        formula=predicted.composition.formula,
    )


def aggregate_csp_reconstruction_metrics(
    results: list[CSPReconstructionResult],
) -> dict[str, Any]:
    if not results:
        return {"num_samples": 0, "valid": None, "match_rate": None, "rmse": None}

    valid_values = [float(result.valid) for result in results]
    match_values = [float(result.match) for result in results]
    rmse_values = [float(result.rmse) for result in results if result.rmse is not None]

    return {
        "num_samples": len(results),
        "valid": float(sum(valid_values) / len(valid_values)),
        "match_rate": float(sum(match_values) / len(match_values)),
        "rmse": None if not rmse_values else float(sum(rmse_values) / len(rmse_values)),
    }
