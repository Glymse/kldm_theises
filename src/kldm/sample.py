from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import CSPTask, resolve_data_root
from kldm.diffusionModels.TDMdev import TrivialisedDiffusionDev
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    aggregate_csp_reconstruction_metrics,
    evaluate_csp_reconstruction,
)

# Here is how we run this sampling evaluation code.
# uv run src/kldm/sample.py --model "artifacts/HPC/checkpoints/checkpoint_epoch_800.pt" --num-total-samples 1000 --samples-per-target 25 --n-steps 1000

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


def _match_rate_at_n(grouped_results: list[list[Any]]) -> float | None:
    if not grouped_results:
        return None
    hits = [1.0 if any(result.match for result in group) else 0.0 for group in grouped_results]
    return float(sum(hits) / len(hits))


def _rmse_at_n(grouped_results: list[list[Any]]) -> float | None:
    if not grouped_results:
        return None

    best_rmses: list[float] = []
    for group in grouped_results:
        group_rmses = [float(result.rmse) for result in group if result.rmse is not None]
        if group_rmses:
            best_rmses.append(min(group_rmses))

    if not best_rmses:
        return None
    return float(sum(best_rmses) / len(best_rmses))


def evaluate_sampling_csp(
    *,
    model: ModelKLDM,
    loader,
    checkpoint_path: Path,
    n_steps: int,
    num_targets: int,
    samples_per_target: int,
    progress_every: int = 1,
) -> dict[str, float | int | None]:
    if num_targets <= 0:
        raise ValueError("num_targets must be positive.")
    if samples_per_target <= 0:
        raise ValueError("samples_per_target must be positive.")

    grouped_results: list[list[Any]] = [[] for _ in range(num_targets)]
    all_results = []

    model.eval()

    for repeat_idx in range(samples_per_target):
        if progress_every > 0 and ((repeat_idx + 1) % progress_every == 0 or repeat_idx == 0):
            print(
                f"sampling repeat {repeat_idx + 1}/{samples_per_target} "
                f"over {num_targets} targets",
                flush=True,
            )

        target_offset = 0
        for batch in loader:
            if target_offset >= num_targets:
                break
            batch = batch.to(model.device)

            with torch.no_grad():
                pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
                    n_steps=n_steps,
                    batch=batch,
                    checkpoint_path=str(checkpoint_path),
                )

            ptr = batch.ptr.tolist()
            for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
                target_idx = target_offset + graph_idx
                if target_idx >= num_targets:
                    break
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
                    print(
                        f"[target {target_idx:03d} repeat {repeat_idx + 1:02d}] "
                        f"evaluation_error: {exc}",
                        flush=True,
                    )
                    continue

                grouped_results[target_idx].append(result)
                all_results.append(result)

            target_offset += batch.num_graphs

    if not all_results:
        raise RuntimeError("Sampling evaluation produced no reconstruction results.")

    summary_at_1 = aggregate_csp_reconstruction_metrics(all_results)
    summary_at_n = {
        "match_rate": _match_rate_at_n(grouped_results),
        "rmse": _rmse_at_n(grouped_results),
    }

    return {
        "num_targets": int(num_targets),
        "samples_per_target": int(samples_per_target),
        "num_total_samples": int(len(all_results)),
        "valid": summary_at_1["valid"],
        "match_rate": summary_at_1["match_rate"],
        "rmse": summary_at_1["rmse"],
        f"match_rate@{samples_per_target}": summary_at_n["match_rate"],
        f"rmse@{samples_per_target}": summary_at_n["rmse"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CSP sampling with repeated draws from a KLDM checkpoint.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SPEC,
        help="Checkpoint path or glob for an artifacts/HPC checkpoint, for example artifacts/HPC/checkpoint_epoch_*.pt",
    )
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of Algorithm 3 sampling steps.")
    parser.add_argument("--batch-size", type=int, default=256, help="Validation dataloader batch size.")
    parser.add_argument(
        "--num-total-samples",
        type=int,
        default=1000,
        help="Total number of sampled structures to evaluate.",
    )
    parser.add_argument(
        "--samples-per-target",
        type=int,
        default=25,
        help="How many independent samples to draw per target structure.",
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
        eps=1e-3,
        n_lambdas=512 if device.type == "cuda" else 128,
        lambda_num_batches=32 if device.type == "cuda" else 8,
        k_wn_score=13,
        n_sigmas=2000 if device.type == "cuda" else 512,
        compute_sigma_norm=bool(args.use_sigma_norm),
    )
    model = ModelKLDM(device=device, diffusion_v=sampling_tdm).to(device)
    if args.num_total_samples <= 0:
        raise ValueError("--num-total-samples must be positive.")
    if args.samples_per_target <= 0:
        raise ValueError("--samples-per-target must be positive.")
    if args.num_total_samples % args.samples_per_target != 0:
        raise ValueError("--num-total-samples must be divisible by --samples-per-target.")

    num_targets = args.num_total_samples // args.samples_per_target

    summary = evaluate_sampling_csp(
        model=model,
        loader=loader,
        checkpoint_path=checkpoint_path,
        n_steps=args.n_steps,
        num_targets=num_targets,
        samples_per_target=args.samples_per_target,
    )

    print("\nSampling evaluation summary")
    print("model:", str(checkpoint_path))
    print("num_targets:", summary["num_targets"])
    print("samples_per_target:", summary["samples_per_target"])
    print("num_total_samples:", summary["num_total_samples"])
    print("valid:", summary["valid"])
    print("match_rate:", summary["match_rate"])
    print("rmse:", summary["rmse"])
    print(f"match_rate@{args.samples_per_target}:", summary[f"match_rate@{args.samples_per_target}"])
    print(f"rmse@{args.samples_per_target}:", summary[f"rmse@{args.samples_per_target}"])
    print("summary:", summary)


if __name__ == "__main__":
    main()
