from __future__ import annotations

from dataclasses import dataclass
import itertools
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from kldm.data import resolve_data_root
from kldm.data.dataset import MP20
from kldm.data.transform import (
    ContinuousIntervalLattice,
    DEFAULT_ATOMIC_VOCAB,
    FACIT_ANGLES_LOC_SCALE,
)

try:
    from pymatgen.analysis.local_env import MinimumDistanceNN
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element, Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:  # pragma: no cover
    Element = Lattice = Structure = MinimumDistanceNN = StructureMatcher = AseAtomsAdaptor = None

try:
    import smact
    from smact.screening import pauling_test
except ImportError:  # pragma: no cover
    smact = None
    pauling_test = None

try:
    from ase.data import chemical_symbols
except ImportError:  # pragma: no cover
    chemical_symbols = None

try:
    from matminer.featurizers.composition.composite import ElementProperty
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
except ImportError:  # pragma: no cover
    ElementProperty = None
    CrystalNNFingerprint = None


CrystalNNFP = CrystalNNFingerprint.from_preset("ops") if CrystalNNFingerprint is not None else None
CompFP = ElementProperty.from_preset("magpie") if ElementProperty is not None else None


@dataclass
class SampleEvaluationResult:
    is_valid: bool
    metrics: dict[str, Any]
    metric_status: dict[str, str]
    basic_validity: dict[str, Any]
    geometric_sanity: dict[str, Any]
    chemical_sanity: dict[str, Any]
    stability_proxy: dict[str, Any]
    decoded_species: list[str]
    decoded_atomic_numbers: list[int]
    decoded_lengths: list[float]
    decoded_angles_deg: list[float]
    structure: Any = None


@dataclass
class CSPReconstructionResult:
    valid: bool
    match: bool
    rmse: Optional[float]
    predicted_structure: Any
    target_structure: Any
    formula: Optional[str] = None


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


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


def _default_lattice_transform() -> ContinuousIntervalLattice:
    data_root = resolve_data_root()
    cache_file = Path(data_root) / MP20.dataset_name / "train_loc_scale.json"
    return ContinuousIntervalLattice(
        standardize=True,
        cache_file=cache_file,
        angles_loc_scale=FACIT_ANGLES_LOC_SCALE,
    )


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
    lattice_transform = lattice_transform or _default_lattice_transform()
    l_tensor = _to_2d_tensor(l)

    if l_tensor.shape[-1] != 6:
        raise ValueError(f"Expected lattice tensor with last dim 6, got shape {tuple(l_tensor.shape)}")

    lengths, angles_rad = lattice_transform.invert_to_lengths_angles(
        l=l_tensor,
        num_atoms=n_atoms,
    )
    lengths = lengths.squeeze(0)
    angles_rad = angles_rad.squeeze(0)
    angles_deg = torch.rad2deg(angles_rad)
    return lengths, angles_deg


def _angles_define_physical_cell(angles_deg: torch.Tensor) -> bool:
    alpha, beta, gamma = [float(x) for x in angles_deg.tolist()]
    if not (0.0 < alpha < 180.0 and 0.0 < beta < 180.0 and 0.0 < gamma < 180.0):
        return False
    return (
        (alpha + beta > gamma)
        and (alpha + gamma > beta)
        and (beta + gamma > alpha)
        and (alpha + beta + gamma < 360.0)
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
        raise ValueError("Decoded lattice contains non-finite lengths or angles.")
    if not (lengths > 0.0).all():
        raise ValueError("Decoded lattice contains non-positive lengths.")
    if not _angles_define_physical_cell(angles_deg):
        raise ValueError("Decoded lattice angles do not define a physical cell.")

    lattice = Lattice.from_parameters(
        a=float(lengths[0].item()),
        b=float(lengths[1].item()),
        c=float(lengths[2].item()),
        alpha=float(angles_deg[0].item()),
        beta=float(angles_deg[1].item()),
        gamma=float(angles_deg[2].item()),
    )
    structure = Structure(
        lattice=lattice,
        species=species,
        coords=(frac_coords % 1.0).detach().cpu().tolist(),
        coords_are_cartesian=False,
    )
    if sort_structure:
        structure = structure.get_sorted_structure()
    return structure


def structures_from_tensors(
    tensors: dict[str, torch.Tensor],
    ptr: torch.Tensor,
    *,
    species_vocab: Optional[list[int]] = None,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
) -> list[Structure | None]:
    _require_pymatgen()
    lattice_transform = lattice_transform or _default_lattice_transform()

    pos = tensors["pos"]
    h = tensors["h"]
    l = tensors["l"]

    ptr_list = ptr.detach().cpu().tolist()
    structures: list[Structure | None] = []

    for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr_list[:-1], ptr_list[1:])):
        try:
            structure = build_structure_from_sample(
                f=pos[start_idx:end_idx],
                l=l[graph_idx],
                a=h[start_idx:end_idx],
                species_vocab=species_vocab,
                lattice_transform=lattice_transform,
            )
        except Exception:
            structure = None
        structures.append(structure)

    return structures


def structures_from_batch(
    batch,
    *,
    species_vocab: Optional[list[int]] = None,
    lattice_transform: Optional[ContinuousIntervalLattice] = None,
) -> list[Structure | None]:
    tensors = {"h": batch.h, "pos": batch.pos, "l": batch.l}
    return structures_from_tensors(
        tensors=tensors,
        ptr=batch.ptr,
        species_vocab=species_vocab,
        lattice_transform=lattice_transform,
    )


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


class CSPMetrics:
    def __init__(
        self,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        ltol: float = 0.3,
    ) -> None:
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.valid: list[int] = []
        self.match: list[int] = []
        self.rmse: list[float] = []

    def __call__(self, input_s: list[Structure | None], target_s: list[Structure | None]) -> None:
        self.update(input_s, target_s)

    def update(self, input_s: list[Structure | None], target_s: list[Structure | None]) -> None:
        assert len(input_s) == len(target_s)

        for si, st in zip(input_s, target_s):
            v, m = 0, 0
            if si is not None and st is not None:
                v = int(validity_structure(si))
                if v:
                    rms = self.matcher.get_rms_dist(si, st)
                    m = int(rms is not None)
                    if rms is not None:
                        self.rmse.append(float(rms[0]))  # only for valid + matching

            self.valid.append(v)
            self.match.append(m)

    def summarize(self) -> dict[str, float | None]:
        return {
            "valid": safe_divide(sum(self.valid), len(self.valid)),
            "match_rate": safe_divide(sum(self.match), len(self.match)),
            "rmse": safe_divide(sum(self.rmse), len(self.rmse)),
        }

    def reset(self) -> None:
        self.valid = []
        self.match = []
        self.rmse = []

    @property
    def details(self) -> dict[str, list[float] | list[int]]:
        return {
            "valid": self.valid,
            "match": self.match,
            "rmse": self.rmse,
        }


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
        return CSPReconstructionResult(
            valid=False,
            match=False,
            rmse=None,
            predicted_structure=None,
            target_structure=None,
            formula=None,
        )

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
            valid=False,
            match=False,
            rmse=None,
            predicted_structure=pred_structure,
            target_structure=None,
            formula=pred_structure.composition.formula if pred_structure is not None else None,
        )

    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    valid = validity_structure(pred_structure)
    match = False
    rmse = None

    if valid:
        try:
            rms = matcher.get_rms_dist(pred_structure, target_structure)
            match = rms is not None
            if rms is not None:
                rmse = float(rms[0])
        except Exception:
            match = False
            rmse = None

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
        return {
            "num_samples": 0,
            "valid": None,
            "match_rate": None,
            "rmse": None,
        }

    valid_values = [1.0 if r.valid else 0.0 for r in results]
    match_values = [1.0 if r.match else 0.0 for r in results]
    rmses = [r.rmse for r in results if r.rmse is not None]

    return {
        "num_samples": len(results),
        "valid": float(sum(valid_values) / len(valid_values)),
        "match_rate": float(sum(match_values) / len(match_values)),
        "rmse": None if not rmses else float(sum(rmses) / len(rmses)),
    }


def make_mattersim_relax_fn(
    *,
    steps: int = 500,
    optimizer: str = "BFGS",
    cell_filter: str = "ExpCellFilter",
    constrain_symmetry: bool = True,
    potential_load_path: Optional[str] = None,
    device: Optional[str] = None,
    hull_fn: Optional[Callable[[Structure, Optional[float]], Optional[float]]] = None,
    novelty_fn: Optional[Callable[[Structure], Optional[bool]]] = None,
) -> Callable[[Structure], dict[str, Any]]:
    if AseAtomsAdaptor is None:
        raise ImportError("pymatgen's ASE adaptor is required to use make_mattersim_relax_fn().")

    try:
        from mattersim.applications.relax import Relaxer
        from mattersim.forcefield.potential import MatterSimCalculator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MatterSim is not installed in the active environment, so a MatterSim-backed "
            "relax_fn cannot be created."
        ) from exc

    calculator_kwargs: dict[str, Any] = {}
    if potential_load_path is not None:
        calculator_kwargs["potential_load_path"] = potential_load_path
    if device is not None:
        calculator_kwargs["device"] = device

    relaxer = Relaxer(
        optimizer=optimizer,
        filter=cell_filter,
        constrain_symmetry=constrain_symmetry,
    )
    adaptor = AseAtomsAdaptor()

    def relax_fn(structure: Structure) -> dict[str, Any]:
        atoms = adaptor.get_atoms(structure)
        atoms.calc = MatterSimCalculator(**calculator_kwargs)

        try:
            relaxed = relaxer.relax(atoms, steps=steps, verbose=False)
        except TypeError:
            try:
                relaxed = relaxer.relax(atoms, steps=steps, logfile=None)
            except TypeError:
                relaxed = relaxer.relax(atoms, steps=steps)

        relax_debug = {
            "type": type(relaxed).__name__,
            "module": type(relaxed).__module__,
        }

        relaxed_atoms = None
        if hasattr(relaxed, "get_positions"):
            relaxed_atoms = relaxed
        elif isinstance(relaxed, dict):
            relax_debug["keys"] = sorted(relaxed.keys())
            for key in ("final_atoms", "atoms", "relaxed_atoms"):
                if key in relaxed and hasattr(relaxed[key], "get_positions"):
                    relaxed_atoms = relaxed[key]
                    break
            if relaxed_atoms is None:
                relaxed_structure = relaxed.get("relaxed_structure")
                if isinstance(relaxed_structure, Structure):
                    energy_per_atom = relaxed.get("energy_per_atom")
                    if energy_per_atom is None and relaxed.get("energy") is not None:
                        energy_per_atom = float(relaxed["energy"]) / len(relaxed_structure)
                    energy_above_hull = hull_fn(relaxed_structure, energy_per_atom) if hull_fn is not None else None
                    stable = energy_above_hull <= 0.1 if energy_above_hull is not None else None
                    novel = novelty_fn(relaxed_structure) if novelty_fn is not None else None
                    unique = True
                    sun = None if stable is None or novel is None else bool(stable and unique and novel)
                    return {
                        "relaxed_structure": relaxed_structure,
                        "energy_per_atom": energy_per_atom,
                        "energy_above_hull": energy_above_hull,
                        "stable": stable,
                        "unique": unique,
                        "novel": novel,
                        "sun": sun,
                        "_relax_debug": relax_debug,
                    }
        else:
            public_attrs = [name for name in dir(relaxed) if not name.startswith("_")]
            relax_debug["public_attrs"] = public_attrs[:100]
            for key in ("final_atoms", "atoms", "relaxed_atoms"):
                candidate = getattr(relaxed, key, None)
                if candidate is not None and hasattr(candidate, "get_positions"):
                    relaxed_atoms = candidate
                    break
            if relaxed_atoms is None:
                relaxed_structure = getattr(relaxed, "relaxed_structure", None)
                if isinstance(relaxed_structure, Structure):
                    total_energy = getattr(relaxed, "energy", None)
                    energy_per_atom = getattr(relaxed, "energy_per_atom", None)
                    if energy_per_atom is None and total_energy is not None:
                        energy_per_atom = float(total_energy) / len(relaxed_structure)
                    energy_above_hull = hull_fn(relaxed_structure, energy_per_atom) if hull_fn is not None else None
                    stable = energy_above_hull <= 0.1 if energy_above_hull is not None else None
                    novel = novelty_fn(relaxed_structure) if novelty_fn is not None else None
                    unique = True
                    sun = None if stable is None or novel is None else bool(stable and unique and novel)
                    return {
                        "relaxed_structure": relaxed_structure,
                        "energy_per_atom": energy_per_atom,
                        "energy_above_hull": energy_above_hull,
                        "stable": stable,
                        "unique": unique,
                        "novel": novel,
                        "sun": sun,
                        "_relax_debug": relax_debug,
                    }

        if relaxed_atoms is None:
            raise RuntimeError(
                "MatterSim relaxation returned an unsupported object; could not extract relaxed atoms. "
                f"debug={relax_debug}"
            )

        relaxed_structure = adaptor.get_structure(relaxed_atoms)
        total_energy = None
        try:
            total_energy = float(relaxed_atoms.get_potential_energy())
        except Exception:
            total_energy = None
        energy_per_atom = None if total_energy is None else total_energy / len(relaxed_structure)
        energy_above_hull = hull_fn(relaxed_structure, energy_per_atom) if hull_fn is not None else None
        stable = energy_above_hull <= 0.1 if energy_above_hull is not None else None
        novel = novelty_fn(relaxed_structure) if novelty_fn is not None else None
        unique = True
        sun = None if stable is None or novel is None else bool(stable and unique and novel)

        return {
            "relaxed_structure": relaxed_structure,
            "energy_per_atom": energy_per_atom,
            "energy_above_hull": energy_above_hull,
            "stable": stable,
            "unique": unique,
            "novel": novel,
            "sun": sun,
            "_relax_debug": relax_debug,
        }

    return relax_fn


def aggregate_csp_metrics(results: list[SampleEvaluationResult]) -> dict[str, Any]:
    if not results:
        return {
            "num_samples": 0,
            "validity_structure_percent": None,
            "validity_composition_percent": None,
            "rmsd_angstrom": None,
            "avg_above_hull_ev_atom": None,
            "stable_percent": None,
            "sun_percent": None,
        }

    def _mean_optional(values: list[Optional[float]]) -> Optional[float]:
        present = [float(v) for v in values if v is not None]
        if not present:
            return None
        return float(sum(present) / len(present))

    validity_structure_vals = [_single_sample_percentage(r.metrics.get("validity_structure")) for r in results]
    validity_composition_vals = [_single_sample_percentage(r.metrics.get("validity_composition")) for r in results]

    return {
        "num_samples": len(results),
        "validity_structure_percent": _mean_optional(validity_structure_vals),
        "validity_composition_percent": _mean_optional(validity_composition_vals),
        "rmsd_angstrom": _mean_optional([r.metrics.get("rmsd_angstrom") for r in results]),
        "avg_above_hull_ev_atom": _mean_optional([r.metrics.get("avg_above_hull_ev_atom") for r in results]),
        "stable_percent": _mean_optional([_single_sample_percentage(r.metrics.get("stable_flag")) for r in results]),
        "sun_percent": _mean_optional(
            [
                _single_sample_percentage(
                    r.stability_proxy.get("relaxation_result", {}).get("sun")
                    if isinstance(r.stability_proxy.get("relaxation_result"), dict)
                    else None
                )
                for r in results
            ]
        ),
    }


def aggregate_dng_metrics(results: list[SampleEvaluationResult]) -> dict[str, Any]:
    return aggregate_csp_metrics(results)


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
    _require_pymatgen()
    lattice_transform = lattice_transform or _default_lattice_transform()

    frac_coords = _to_2d_tensor(f)
    if frac_coords.shape[-1] != 3:
        raise ValueError(f"Expected fractional coordinates with last dim 3, got shape {tuple(frac_coords.shape)}")

    n_atoms = int(frac_coords.shape[0])
    atomic_numbers, species = decode_atom_types(a=a, species_vocab=species_vocab)
    lengths, angles_deg = decode_lattice(l=l, n_atoms=n_atoms, lattice_transform=lattice_transform)
    lengths_finite = bool(torch.isfinite(lengths).all().item())
    angles_finite = bool(torch.isfinite(angles_deg).all().item())
    physical_angles = _angles_define_physical_cell(angles_deg) if angles_finite else False

    frac_coords_wrapped = frac_coords % 1.0

    basic_validity = {
        "fractional_coords_in_unit_cell": bool(((frac_coords_wrapped >= 0.0) & (frac_coords_wrapped < 1.0)).all().item()),
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
            coords=frac_coords_wrapped.detach().cpu().tolist(),
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
    chemistry_ok = basic_validity["all_species_valid"] and structure is not None

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
        "stable_flag": None,
        "novel_flag": None,
    }
    metric_status = {
        "rmsd_angstrom": "unavailable: requires relaxed_structure from relax_fn or a reference structure",
        "avg_above_hull_ev_atom": "unavailable: requires energy_above_hull/e_above_hull from relax_fn or hull backend",
        "stable_flag": "unavailable: requires stable flag or energy_above_hull from relax_fn",
        "novel_flag": "unavailable: requires novelty backend outside per-sample evaluation",
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
        novel_flag = relaxation_result.get("novel")

        metrics["rmsd_angstrom"] = rmsd_angstrom
        metrics["avg_above_hull_ev_atom"] = None if e_above_hull is None else float(e_above_hull)
        metrics["stable_flag"] = None if stable_flag is None else bool(stable_flag)
        metrics["novel_flag"] = None if novel_flag is None else bool(novel_flag)
        metric_status["rmsd_angstrom"] = "ok" if rmsd_angstrom is not None else metric_status["rmsd_angstrom"]
        if rmsd_angstrom is None and relaxed_structure is not None:
            metric_status["rmsd_angstrom"] = (
                "unavailable: relaxed_structure exists but StructureMatcher could not compute RMSD"
            )
        metric_status["avg_above_hull_ev_atom"] = (
            "ok" if metrics["avg_above_hull_ev_atom"] is not None else metric_status["avg_above_hull_ev_atom"]
        )
        metric_status["stable_flag"] = (
            "ok" if metrics["stable_flag"] is not None else metric_status["stable_flag"]
        )
        metric_status["novel_flag"] = (
            "ok" if metrics["novel_flag"] is not None else metric_status["novel_flag"]
        )

    if build_error is not None:
        geometric_sanity["build_error"] = build_error

    return SampleEvaluationResult(
        is_valid=all_basic and geometry_ok and chemistry_ok,
        metrics=metrics,
        metric_status=metric_status,
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
