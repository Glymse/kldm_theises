from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import CSPTask, resolve_data_root
from kldm.diffusionModels.TDMdev import TrivialisedDiffusionDev
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    aggregate_csp_reconstruction_metrics,
    build_structure_from_sample,
    decode_atom_types,
    decode_lattice,
    evaluate_csp_reconstruction,
    validity_structure,
)

#Here is how we run this sampling code.
# uv run src/kldm/sample.py --model "artifacts/HPC/checkpoints/checkpoint_epoch_800.pt" --max-samples 5 --n-steps 1000

DEFAULT_MODEL_SPEC = "artifacts/HPC/checkpoint_epoch_*.pt"


def resolve_checkpoint_path(model_spec: str) -> Path:
    path = Path(model_spec).expanduser()
    if any(ch in model_spec for ch in "*?[]"):
        matches = [Path(match) for match in glob.glob(str(path))]
        if not matches:
            raise FileNotFoundError(f"No checkpoints matched model spec: {model_spec}")

        def sort_key(candidate: Path) -> tuple[int, float, str]:
            match = re.search(r"_epoch_(\d+)", candidate.stem)
            epoch = int(match.group(1)) if match is not None else -1
            mtime = candidate.stat().st_mtime
            return epoch, mtime, str(candidate)

        return max(matches, key=sort_key)

    if not path.exists():
        fallback_paths = [path]
        if path.parent.name == "checkpoints":
            fallback_paths.append(path.parent.parent / path.name)
        fallback_paths.append(Path("artifacts") / "HPC" / path.name)

        for candidate in fallback_paths:
            if candidate.exists():
                return candidate

        if path.name.startswith("checkpoint_epoch_"):
            search_roots = [path.parent, path.parent.parent, Path("artifacts") / "HPC"]
            for root in search_roots:
                if not root.exists():
                    continue
                direct_match = root / path.name
                if direct_match.exists():
                    return direct_match
                matches = sorted(Path(match) for match in glob.glob(str(root / path.name)))
                if matches:
                    return matches[-1]

        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


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

def print_matching_sample(
    *,
    sample_idx: int,
    pred_f: torch.Tensor,
    pred_v: torch.Tensor,
    pred_l: torch.Tensor,
    pred_a: torch.Tensor,
    result,
) -> None:
    atom_type_index = pred_a if pred_a.ndim == 1 else pred_a.argmax(dim=-1)

    print(f"\nMatching CSP sample {sample_idx + 1:03d}")
    print("pos shape:", tuple(pred_f.shape))
    print("v shape:", tuple(pred_v.shape))
    print("l shape:", tuple(pred_l.shape))
    print("h shape:", tuple(pred_a.shape))
    print("atom_type_index shape:", tuple(atom_type_index.shape))
    print("First 3 sampled fractional coordinates:")
    print(pred_f[:3])
    print("Predicted atom type indices:")
    print(atom_type_index)
    print("Sampled lattice:")
    print(pred_l)
    print("valid:", result.valid)
    print("match:", result.match)
    print("RMSE:", result.rmse)
    print("formula:", result.formula)


def sample_validation_csp(
    *,
    model: ModelKLDM,
    loader,
    checkpoint_path: Path,
    n_steps: int,
    max_graphs: int | None,
) -> dict[str, float | int | None]:
    reconstruction_results = []
    num_graphs_seen = 0

    model.eval()

    for batch in loader:
        batch = batch.to(model.device)
        with torch.no_grad():
            pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
                n_steps=n_steps,
                batch=batch,
                checkpoint_path=str(checkpoint_path),
            )

        ptr = batch.ptr.tolist()
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            sample_idx = num_graphs_seen
            try:
                result = evaluate_csp_reconstruction(
                    pred_f=pos_t[start:end],
                    pred_l=l_t[graph_idx],
                    pred_a=h_t[start:end],
                    target_f=batch.pos[start:end],
                    target_l=batch.l[graph_idx],
                    target_a=batch.h[start:end],
                )
            except Exception as exc:
                print(f"[sample {sample_idx:15d}] evaluation_error: {exc}")
                continue

            reconstruction_results.append(result)

            if result.match:
                print_matching_sample(
                    sample_idx=sample_idx,
                    pred_f=pos_t[start:end],
                    pred_v=v_t[start:end],
                    pred_l=l_t[graph_idx],
                    pred_a=h_t[start:end],
                    result=result,
                )
            else:
                print_invalid_sample_diagnostic(
                    sample_idx=sample_idx,
                    pred_f=pos_t[start:end],
                    pred_l=l_t[graph_idx],
                    pred_a=h_t[start:end],
                )
                print("valid:", result.valid)
                print("match:", result.match)
                print("RMSE:", result.rmse)

            num_graphs_seen += 1
            if max_graphs is not None and num_graphs_seen >= max_graphs:
                break
        if max_graphs is not None and num_graphs_seen >= max_graphs:
            break

    if not reconstruction_results:
        raise RuntimeError("Validation sampling produced no reconstruction results.")

    summary = aggregate_csp_reconstruction_metrics(reconstruction_results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample validation CSPs with a KLDM checkpoint and report metrics.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SPEC,
        help="Checkpoint path or glob for an artifacts/HPC checkpoint, for example artifacts/HPC/checkpoint_epoch_*.pt",
    )
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of Algorithm 3 sampling steps.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Validation dataloader batch size. Match trainHPC validation for closest parity.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Maximum number of validation CSPs to sample. Use 0 or a negative value to sample the full loader.",
    )
    parser.add_argument(
        "--use-sigma-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sigma_norm behavior in TDMdev. Enabled by default to match trainHPC sampling.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()
    checkpoint_path = resolve_checkpoint_path(args.model)

    print(f"using checkpoint: {checkpoint_path}")

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        download=True,
    )

    sampling_tdm = TrivialisedDiffusionDev(
        eps=1e-6,
        n_lambdas=512 if device.type == "cuda" else 128,
        lambda_num_batches=32 if device.type == "cuda" else 8,
        k_wn_score=13,
        n_sigmas=2000 if device.type == "cuda" else 512,
        compute_sigma_norm=bool(args.use_sigma_norm),
    )
    model = ModelKLDM(device=device, diffusion_v=sampling_tdm).to(device)
    max_graphs = None if args.max_samples <= 0 else args.max_samples
    summary = sample_validation_csp(
        model=model,
        loader=loader,
        checkpoint_path=checkpoint_path,
        n_steps=args.n_steps,
        max_graphs=max_graphs,
    )

    print("\nValidation summary")
    print("model:", str(checkpoint_path))
    print("val/valid:", summary["valid"])
    print("val/match:", summary["match_rate"])
    print("val/match_rate:", summary["match_rate"])
    print("val/rmse:", summary["rmse"])
    print("csp_metrics:", summary)


if __name__ == "__main__":
    main()
