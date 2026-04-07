from __future__ import annotations

from dataclasses import dataclass
import itertools
import warnings
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np
import torch

from kldm.data.transform import ContinuousIntervalLattice, DEFAULT_ATOMIC_VOCAB

try:
    from pymatgen.analysis.local_env import MinimumDistanceNN
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
except ImportError:  # pragma: no cover - depends on local environment
    Element = Lattice = Structure = MinimumDistanceNN = StructureMatcher = None

try:
    import smact
    from smact.screening import pauling_test
except ImportError:  # pragma: no cover - depends on local environment
    smact = None
    pauling_test = None

try:
    from ase.data import chemical_symbols
except ImportError:  # pragma: no cover - depends on local environment
    chemical_symbols = None

try:
    from matminer.featurizers.composition.composite import ElementProperty
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
except ImportError:  # pragma: no cover - depends on local environment
    ElementProperty = None
    CrystalNNFingerprint = None


CrystalNNFP = CrystalNNFingerprint.from_preset("ops") if CrystalNNFingerprint is not None else None
CompFP = ElementProperty.from_preset("magpie") if ElementProperty is not None else None


@dataclass
class SampleEvaluationResult:
    is_valid: bool
    metrics: dict[str, Any]
    basic_validity: dict[str, Any]
    geometric_sanity: dict[str, Any]
    chemical_sanity: dict[str, Any]
    stability_proxy: dict[str, Any]
    decoded_species: list[str]
    decoded_atomic_numbers: list[int]
    decoded_lengths: list[float]
    decoded_angles_deg: list[float]
    structure: Any = None


def validity_structure(structure: Structure, cutoff: float = 0.5, max_length: float = 40.0, min_volume: float = 0.1) -> bool:
    try:
        distance_matrix = structure.distance_matrix
    except Exception as exc:
        warnings.warn(f"In the structure validity the following error occurred: {exc}")
        return False

    distance_matrix = distance_matrix + np.diag(np.ones(distance_matrix.shape[0]) * (cutoff + 10.0))
    return not (
        distance_matrix.min() < cutoff
        or structure.volume < min_volume
        or max(structure.lattice.abc) > max_length
    )


def _validity_composition(
    unique_symbols: list[str],
    count: list[int],
    *,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool | None:
    if smact is None:
        return None

    space = smact.element_dictionary(unique_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    if len(set(unique_symbols)) == 1:
        return True

    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in unique_symbols]
        if all(is_metal_list):
            return True

    threshold = max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, _ = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test and pauling_test is not None:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_ok = True
            else:
                electroneg_ok = True
            if electroneg_ok:
                return True
    return False


def validity_composition(structure: Structure) -> bool | None:
    if chemical_symbols is None:
        return None

    counter = Counter(structure.atomic_numbers)
    unique_symbols: list[str] = []
    count: list[int] = []
    try:
        for n, c in counter.items():
            unique_symbols.append(chemical_symbols[n])
            count.append(c)
        return _validity_composition(unique_symbols, count)
    except Exception:
        return False


def fp_composition(structure: Structure) -> Optional[list[float]]:
    if CompFP is None:
        return None
    return list(CompFP.featurize(structure.composition))


def fp_structural(structure: Structure) -> Optional[list[float]]:
    if CrystalNNFP is None:
        return None
    try:
        fp_struct = np.array([CrystalNNFP.featurize(structure, i) for i in range(len(structure))])
        return list(fp_struct.mean(axis=0))
    except Exception:
        return None


def _require_pymatgen() -> None:
    if Structure is None or Lattice is None or Element is None:
        raise ImportError(
            "sample_evaluation requires pymatgen. Install it in the active environment "
            "before running structure validity checks."
        )


def _to_2d_tensor(x: torch.Tensor | list[float] | list[list[float]]) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def decode_atom_types(
    a: torch.Tensor | list[int] | list[list[float]],
    species_vocab: Optional[list[int]] = None,
) -> tuple[list[int], list[str]]:
    species_vocab = species_vocab or DEFAULT_ATOMIC_VOCAB
    a_tensor = torch.as_tensor(a)

    if a_tensor.ndim == 1:
        atomic_numbers = [int(x) for x in a_tensor.tolist()]
    elif a_tensor.ndim == 2:
        indices = a_tensor.argmax(dim=-1).tolist()
        atomic_numbers = [int(species_vocab[int(i)]) for i in indices]
    else:
        raise ValueError(f"Expected atom tensor with ndim 1 or 2, got shape {tuple(a_tensor.shape)}")

    species: list[str] = []
    for z in atomic_numbers:
        try:
            species.append(Element.from_Z(z).symbol)
        except Exception:
            species.append(f"INVALID_{z}")

    return atomic_numbers, species


def decode_lattice(
    l: torch.Tensor | list[float] | list[list[float]],
    n_atoms: int,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lattice_transform = lattice_transform or ContinuousIntervalLattice()
    l_tensor = _to_2d_tensor(l)

    if l_tensor.shape[-1] != 6:
        raise ValueError(f"Expected lattice tensor with last dim 6, got shape {tuple(l_tensor.shape)}")

    raw_lengths = l_tensor[..., :3].squeeze(0)
    raw_angles = l_tensor[..., 3:].squeeze(0)

    lengths = raw_lengths.clone()
    if n_atoms in lattice_transform.lengths_loc_scale:
        loc, scale = lattice_transform.lengths_loc_scale[n_atoms]
        lengths = lengths * scale.to(lengths.device) + loc.to(lengths.device)
    lengths = torch.exp(lengths)
    if lattice_transform.normalize_lengths_by_num_atoms:
        lengths = lengths * (n_atoms ** (1.0 / 3.0))

    angles = raw_angles.clone()
    if lattice_transform.angles_loc_scale is not None:
        loc, scale = lattice_transform.angles_loc_scale
        angles = angles * scale.to(angles.device) + loc.to(angles.device)
    angles_rad = torch.atan(angles) + torch.pi / 2
    angles_deg = torch.rad2deg(angles_rad)

    return lengths, angles_deg


def _angles_define_physical_cell(angles_deg: torch.Tensor) -> bool:
    alpha, beta, gamma = [float(x) for x in angles_deg.tolist()]
    if not (0.0 < alpha < 180.0 and 0.0 < beta < 180.0 and 0.0 < gamma < 180.0):
        return False
    return (alpha + beta > gamma) and (alpha + gamma > beta) and (beta + gamma > alpha) and (
        alpha + beta + gamma < 360.0
    )


def _single_sample_percentage(flag: Any) -> Optional[float]:
    if flag is None:
        return None
    return 100.0 if bool(flag) else 0.0


def _compute_rmsd_angstrom(
    structure: Optional[Structure],
    relaxed_structure: Optional[Structure],
) -> Optional[float]:
    if structure is None or relaxed_structure is None or StructureMatcher is None:
        return None
    try:
        matcher = StructureMatcher(primitive_cell=False, scale=True)
        rmsd = matcher.get_rms_dist(structure, relaxed_structure)
        if rmsd is None:
            return None
        return float(rmsd[0])
    except Exception:
        return None


def evaluate_sample(
    f: torch.Tensor | list[list[float]],
    l: torch.Tensor | list[float] | list[list[float]],
    a: torch.Tensor | list[int] | list[list[float]],
    *,
    species_vocab: Optional[list[int]] = None,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
    min_distance_reject: float = 0.8,
    min_distance_warn: float = 1.2,
    min_volume: float = 1e-3,
    allow_oxi_guess: bool = True,
    relax_fn: Optional[Callable[[Any], dict[str, Any]]] = None,
    reference_structure: Optional[Structure] = None,
) -> SampleEvaluationResult:
    """
    Evaluate a sampled crystal in layers:
    1. basic validity
    2. structure construction
    3. geometric sanity
    4. chemical sanity
    5. optional stability proxy via `relax_fn`

    Notes:
    - `f` is expected to be fractional coordinates in [0, 1).
    - `l` is expected to be the model-space 6D lattice representation used by
      `ContinuousIntervalLattice`: [log_lengths, tan(angle - pi/2)].
    - `a` may be either one-hot/logits [N, C] or already-decoded atomic numbers [N].
    """
    _require_pymatgen()

    frac_coords = _to_2d_tensor(f)
    if frac_coords.shape[-1] != 3:
        raise ValueError(f"Expected fractional coordinates with last dim 3, got shape {tuple(frac_coords.shape)}")

    n_atoms = int(frac_coords.shape[0])
    atomic_numbers, species = decode_atom_types(a=a, species_vocab=species_vocab)
    lengths, angles_deg = decode_lattice(l=l, n_atoms=n_atoms, lattice_transform=lattice_transform)
    lengths_finite = bool(torch.isfinite(lengths).all().item())
    angles_finite = bool(torch.isfinite(angles_deg).all().item())
    physical_angles = _angles_define_physical_cell(angles_deg) if angles_finite else False

    basic_validity = {
        "fractional_coords_in_unit_cell": bool(((frac_coords >= 0.0) & (frac_coords < 1.0)).all().item()),
        "num_atoms_match": bool(len(atomic_numbers) == n_atoms),
        "all_species_valid": all(not s.startswith("INVALID_") for s in species),
        "finite_lengths": lengths_finite,
        "finite_angles": angles_finite,
        "positive_lengths": bool(lengths_finite and (lengths > 0.0).all().item()),
        "sensible_angles": bool(angles_finite and ((angles_deg > 0.0) & (angles_deg < 180.0)).all().item()),
        "physical_angle_combination": physical_angles,
    }

    structure = None
    build_error = None
    geometric_sanity: dict[str, Any] = {}
    chemical_sanity: dict[str, Any] = {
        "formula": None,
        "composition_valid": None,
        "oxidation_state_guesses": None,
        "oxi_guess_error": None,
        "composition_fingerprint": None,
        "structure_fingerprint": None,
    }
    stability_proxy: dict[str, Any] = {
        "relaxed": False,
        "relaxation_result": None,
        "relaxation_error": None,
    }

    try:
        lattice = Lattice.from_parameters(
            a=float(lengths[0].item()),
            b=float(lengths[1].item()),
            c=float(lengths[2].item()),
            alpha=float(angles_deg[0].item()),
            beta=float(angles_deg[1].item()),
            gamma=float(angles_deg[2].item()),
        )
        basic_validity["finite_volume"] = bool(torch.isfinite(torch.tensor(lattice.volume)).item())
        basic_validity["non_negligible_volume"] = bool(
            basic_validity["finite_volume"] and lattice.volume > min_volume
        )
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=frac_coords.detach().cpu().tolist(),
            coords_are_cartesian=False,
        )
    except Exception as exc:
        build_error = str(exc)
        basic_validity["finite_volume"] = False
        basic_validity["non_negligible_volume"] = False

    if structure is not None:
        try:
            dmat = structure.distance_matrix
            nonzero = dmat[dmat > 1e-8]
            min_dist = float(nonzero.min()) if len(nonzero) else None
            nn_summary: list[dict[str, Any]] = []
            nn = MinimumDistanceNN()
            for i, site in enumerate(structure):
                try:
                    neighs = nn.get_nn_info(structure, i)
                    nn_summary.append({"site": i, "species": str(site.specie), "neighbors": len(neighs)})
                except Exception as exc:
                    nn_summary.append({"site": i, "species": str(site.specie), "neighbors": None, "error": str(exc)})

            geometric_sanity = {
                "volume": float(structure.volume),
                "min_interatomic_distance": min_dist,
                "reject_too_close": bool(min_dist is not None and min_dist < min_distance_reject),
                "warn_short_contact": bool(min_dist is not None and min_dist < min_distance_warn),
                "structure_valid": validity_structure(structure, cutoff=min_distance_reject),
                "nn_summary": nn_summary,
            }

            chemical_sanity["formula"] = structure.composition.formula
            chemical_sanity["composition_valid"] = validity_composition(structure)
            if allow_oxi_guess:
                try:
                    chemical_sanity["oxidation_state_guesses"] = structure.composition.oxi_state_guesses()
                except Exception as exc:
                    chemical_sanity["oxi_guess_error"] = str(exc)
            chemical_sanity["composition_fingerprint"] = fp_composition(structure)
            chemical_sanity["structure_fingerprint"] = fp_structural(structure)

            if relax_fn is not None:
                try:
                    stability_proxy["relaxation_result"] = relax_fn(structure)
                    stability_proxy["relaxed"] = True
                except Exception as exc:
                    stability_proxy["relaxation_error"] = str(exc)
        except Exception as exc:
            geometric_sanity = {
                "volume": float(structure.volume),
                "min_interatomic_distance": None,
                "reject_too_close": True,
                "warn_short_contact": False,
                "nn_summary": [],
                "distance_matrix_error": str(exc),
            }

    all_basic = all(bool(v) for v in basic_validity.values())
    geometry_ok = structure is not None and not geometric_sanity.get("reject_too_close", False) and bool(
        geometric_sanity.get("structure_valid", structure is not None)
    )
    composition_valid = chemical_sanity.get("composition_valid")
    chemistry_ok = basic_validity["all_species_valid"] and structure is not None and (
        composition_valid is not False
    )

    metrics = {
        "validity_structure": geometric_sanity.get("structure_valid"),
        "validity_composition": chemical_sanity.get("composition_valid"),
        "min_interatomic_distance": geometric_sanity.get("min_interatomic_distance"),
        "volume": geometric_sanity.get("volume"),
        "warn_short_contact": geometric_sanity.get("warn_short_contact"),
        "fp_composition_available": chemical_sanity.get("composition_fingerprint") is not None,
        "fp_structural_available": chemical_sanity.get("structure_fingerprint") is not None,
        "rmsd_angstrom": None,
        "avg_above_hull_ev_atom": None,
        "stable_percent": None,
        "sun_percent": None,
    }

    relaxation_result = stability_proxy.get("relaxation_result")
    if isinstance(relaxation_result, dict):
        relaxed_structure = relaxation_result.get("relaxed_structure", reference_structure)
        rmsd_angstrom = _compute_rmsd_angstrom(structure, relaxed_structure)
        e_above_hull = relaxation_result.get("energy_above_hull")
        if e_above_hull is None:
            e_above_hull = relaxation_result.get("e_above_hull")
        stable_flag = relaxation_result.get("stable")
        if stable_flag is None and e_above_hull is not None:
            stable_flag = float(e_above_hull) <= 0.1
        sun_flag = relaxation_result.get("sun")
        if sun_flag is None:
            sun_flag = relaxation_result.get("structure_unrelaxed_and_novel")

        metrics["rmsd_angstrom"] = rmsd_angstrom
        metrics["avg_above_hull_ev_atom"] = None if e_above_hull is None else float(e_above_hull)
        metrics["stable_percent"] = _single_sample_percentage(stable_flag)
        metrics["sun_percent"] = _single_sample_percentage(sun_flag)

    if build_error is not None:
        geometric_sanity["build_error"] = build_error

    return SampleEvaluationResult(
        is_valid=all_basic and geometry_ok and chemistry_ok,
        metrics=metrics,
        basic_validity=basic_validity,
        geometric_sanity=geometric_sanity,
        chemical_sanity=chemical_sanity,
        stability_proxy=stability_proxy,
        decoded_species=species,
        decoded_atomic_numbers=atomic_numbers,
        decoded_lengths=[float(x) for x in lengths.detach().cpu().tolist()],
        decoded_angles_deg=[float(x) for x in angles_deg.detach().cpu().tolist()],
        structure=structure,
    )
