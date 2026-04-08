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
    aggregate_csp_metrics,
    evaluate_sample,
    make_mattersim_relax_fn,
)


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
    relax_fn = None
    try:
        relax_fn = make_mattersim_relax_fn()
    except Exception as exc:
        print("relax_fn_status:", exc)

    max_samples = 100
    n_steps = 1000
    valid_results = []
    all_results = []
    template_iter = itertools.cycle(loader)

    for sample_idx in range(max_samples):
        batch = next(template_iter)
        batch = batch.to(device)
        pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
            n_steps=n_steps,
            batch=batch,
        )

        try:
            result = evaluate_sample(
                f=pos_t,
                l=l_t[0],
                a=h_t,
                relax_fn=relax_fn,
            )
        except Exception as exc:
            print(f"[sample {sample_idx:03d}] evaluation_error: {exc}")
            continue

        all_results.append(result)
        if not result.is_valid:
            continue

        valid_results.append(result)
        atom_type_index = h_t if h_t.ndim == 1 else h_t.argmax(dim=-1)

        print(f"\nValid CSP sample {len(valid_results):03d} / {sample_idx + 1:03d}")
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
        print("metrics:", result.metrics)
        print("RMSD [A]:", result.metrics["rmsd_angstrom"])
        print("AVG. ABOVE HULL [eV/atom]:", result.metrics["avg_above_hull_ev_atom"])
        print("STABLE:", result.metrics["stable_flag"])
        print("NOVEL:", result.metrics["novel_flag"])
        print("metric_status:", result.metric_status)
        print(
            "stability_proxy:",
            {
                "relaxed": result.stability_proxy.get("relaxed"),
                "relaxation_error": result.stability_proxy.get("relaxation_error"),
                "relaxation_result_keys": (
                    sorted(result.stability_proxy["relaxation_result"].keys())
                    if isinstance(result.stability_proxy.get("relaxation_result"), dict)
                    else None
                ),
                "relax_debug": (
                    result.stability_proxy["relaxation_result"].get("_relax_debug")
                    if isinstance(result.stability_proxy.get("relaxation_result"), dict)
                    else None
                ),
            },
        )
        print("decoded_species:", result.decoded_species)
        print("decoded_lengths:", result.decoded_lengths)
        print("decoded_angles_deg:", result.decoded_angles_deg)
        print("chemical_formula:", result.chemical_sanity["formula"])

    print("\nSampling summary")
    print("generated_samples:", max_samples)
    print("valid_samples:", len(valid_results))
    print("all_metrics:", aggregate_csp_metrics(all_results))
    print("valid_metrics:", aggregate_csp_metrics(valid_results))


if __name__ == "__main__":
    main()
