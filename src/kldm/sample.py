from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import DNGTask, resolve_data_root
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import evaluate_sample


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()

    loader = DNGTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    model = ModelKLDM(device=device).to(device)
    pos_t, v_t, l_t, h_t = model.sample_DNG_algorithm3(
        n_steps=1000,
        batch=batch,
    )

    atom_type_index = h_t.argmax(dim=-1)

    print("Sampled one DNG crystal")
    print("pos shape:", tuple(pos_t.shape))
    print("v shape:", tuple(v_t.shape))
    print("l shape:", tuple(l_t.shape))
    print("h shape:", tuple(h_t.shape))
    print("atom_type_index shape:", tuple(atom_type_index.shape))

    print("\nFirst 3 sampled fractional coordinates:")
    print(pos_t[:3])

    print("\nPredicted atom type indices:")
    print(atom_type_index)

    print("\nSampled lattice:")
    print(l_t)

    print("\nSample evaluation")
    try:
        result = evaluate_sample(
            f=pos_t,
            l=l_t[0],
            a=h_t,
        )

        print("is_valid:", result.is_valid)
        print("metrics:", result.metrics)
        print("RMSD [A]:", result.metrics["rmsd_angstrom"])
        print("AVG. ABOVE HULL [eV/atom]:", result.metrics["avg_above_hull_ev_atom"])
        print("STABLE [%]:", result.metrics["stable_percent"])
        print("S.U.N. [%]:", result.metrics["sun_percent"])
        print("decoded_species:", result.decoded_species)
        print("decoded_atomic_numbers:", result.decoded_atomic_numbers)
        print("decoded_lengths:", result.decoded_lengths)
        print("decoded_angles_deg:", result.decoded_angles_deg)
        print("basic_validity:", result.basic_validity)
        print("geometric_sanity:", result.geometric_sanity)
        print(
            "chemical_sanity:",
            {
                "formula": result.chemical_sanity["formula"],
                "composition_valid": result.chemical_sanity["composition_valid"],
                "oxidation_state_guesses": result.chemical_sanity["oxidation_state_guesses"],
                "oxi_guess_error": result.chemical_sanity["oxi_guess_error"],
                "composition_fingerprint_available": result.chemical_sanity["composition_fingerprint"] is not None,
                "structure_fingerprint_available": result.chemical_sanity["structure_fingerprint"] is not None,
            },
        )
    except Exception as exc:
        print("evaluation_error:", exc)


if __name__ == "__main__":
    main()
