from __future__ import annotations

import itertools
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import CSPTask, resolve_data_root
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    aggregate_csp_reconstruction_metrics,
    build_structure_from_sample,
    decode_atom_types,
    decode_lattice,
    evaluate_csp_reconstruction,
    validity_structure,
)


def print_invalid_sample_diagnostic(
    *,
    sample_idx: int,
    pred_f: torch.Tensor,
    pred_l: torch.Tensor,
    pred_a: torch.Tensor,
) -> None:
    print(f"\nInvalid CSP sample {sample_idx + 1:03d}")
    print("pos shape:", tuple(pred_f.shape))
    print("l shape:", tuple(pred_l.shape))
    print("h shape:", tuple(pred_a.shape))
    print("First 3 sampled fractional coordinates:")
    print(pred_f[:3])

    try:
        atomic_numbers, species = decode_atom_types(pred_a)
        lengths, angles_deg = decode_lattice(pred_l, n_atoms=int(pred_f.shape[0]))
        print("decoded_species:", species)
        print("decoded_atomic_numbers:", atomic_numbers)
        print("decoded_lengths:", [float(x) for x in lengths.detach().cpu().tolist()])
        print("decoded_angles_deg:", [float(x) for x in angles_deg.detach().cpu().tolist()])
    except Exception as exc:
        print("decode_error:", str(exc))
        return

    try:
        structure = build_structure_from_sample(f=pred_f, l=pred_l, a=pred_a)
    except Exception as exc:
        print("structure_build_error:", str(exc))
        return

    try:
        dmat = torch.as_tensor(structure.distance_matrix, dtype=torch.float32)
        nonzero = dmat[dmat > 1e-8]
        min_dist = None if nonzero.numel() == 0 else float(nonzero.min().item())
    except Exception as exc:
        print("distance_matrix_error:", str(exc))
        min_dist = None

    print("structure_valid:", validity_structure(structure))
    print("volume:", float(structure.volume))
    print("min_interatomic_distance:", min_dist)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )

    model = ModelKLDM(device=device).to(device)
    max_samples = 10
    n_steps = 1000
    reconstruction_results = []
    oracle_lattice_results = []
    oracle_coordinate_results = []
    invalid_diagnostics_printed = 0
    template_iter = itertools.cycle(loader)

    for sample_idx in range(max_samples):
        batch = next(template_iter)
        batch = batch.to(device)
        pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
            n_steps=n_steps,
            batch=batch,
        )

        try:
            result = evaluate_csp_reconstruction(
                pred_f=pos_t,
                pred_l=l_t[0],
                pred_a=h_t,
                target_f=batch.pos,
                target_l=batch.l[0],
                target_a=batch.h,
            )
        except Exception as exc:
            print(f"[sample {sample_idx:03d}] evaluation_error: {exc}")
            continue

        try:
            oracle_lattice_result = evaluate_csp_reconstruction(
                pred_f=pos_t,
                pred_l=batch.l[0],
                pred_a=h_t,
                target_f=batch.pos,
                target_l=batch.l[0],
                target_a=batch.h,
            )
            oracle_lattice_results.append(oracle_lattice_result)
        except Exception as exc:
            print(f"[sample {sample_idx:03d}] oracle_lattice_error: {exc}")

        try:
            oracle_coordinate_result = evaluate_csp_reconstruction(
                pred_f=batch.pos,
                pred_l=l_t[0],
                pred_a=h_t,
                target_f=batch.pos,
                target_l=batch.l[0],
                target_a=batch.h,
            )
            oracle_coordinate_results.append(oracle_coordinate_result)
        except Exception as exc:
            print(f"[sample {sample_idx:03d}] oracle_coordinate_error: {exc}")

        reconstruction_results.append(result)
        atom_type_index = h_t if h_t.ndim == 1 else h_t.argmax(dim=-1)

        if not result.match:
            if invalid_diagnostics_printed < 3:
                print_invalid_sample_diagnostic(
                    sample_idx=sample_idx,
                    pred_f=pos_t,
                    pred_l=l_t[0],
                    pred_a=h_t,
                )
                print("valid:", result.valid)
                print("match:", result.match)
                print("RMSE:", result.rmse)
                invalid_diagnostics_printed += 1
            continue

        print(f"\nMatching CSP sample {sum(r.match for r in reconstruction_results):03d} / {sample_idx + 1:03d}")
        print("pos shape:", tuple(pos_t.shape))
        print("v shape:", tuple(v_t.shape))
        print("l shape:", tuple(l_t.shape))
        print("h shape:", tuple(h_t.shape))
        print("atom_type_index shape:", tuple(atom_type_index.shape))
        print("First 3 sampled fractional coordinates:")
        print(pos_t[:3])
        print("Predicted atom type indices:")
        print(atom_type_index)
        print("Sampled lattice:")
        print(l_t)
        print("valid:", result.valid)
        print("match:", result.match)
        print("RMSE:", result.rmse)
        print("formula:", result.formula)

    summary = aggregate_csp_reconstruction_metrics(reconstruction_results)
    oracle_lattice_summary = aggregate_csp_reconstruction_metrics(oracle_lattice_results)
    oracle_coordinate_summary = aggregate_csp_reconstruction_metrics(oracle_coordinate_results)
    print("\nSampling summary")
    print("generated_samples:", max_samples)
    print("valid_samples:", sum(r.valid for r in reconstruction_results))
    print("matching_samples:", sum(r.match for r in reconstruction_results))
    print("MR:", summary["match_rate"])
    print("RMSE:", summary["rmse"])
    print("csp_metrics:", summary)
    print("\nOracle-lattice summary")
    print("valid_samples:", sum(r.valid for r in oracle_lattice_results))
    print("matching_samples:", sum(r.match for r in oracle_lattice_results))
    print("MR:", oracle_lattice_summary["match_rate"])
    print("RMSE:", oracle_lattice_summary["rmse"])
    print("csp_metrics:", oracle_lattice_summary)
    print("\nOracle-coordinate summary")
    print("valid_samples:", sum(r.valid for r in oracle_coordinate_results))
    print("matching_samples:", sum(r.match for r in oracle_coordinate_results))
    print("MR:", oracle_coordinate_summary["match_rate"])
    print("RMSE:", oracle_coordinate_summary["rmse"])
    print("csp_metrics:", oracle_coordinate_summary)


if __name__ == "__main__":
    main()
