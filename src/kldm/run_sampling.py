from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from kldm.run_experiment import load_experiment_config

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/run_sampling.py") from exc

from kldm.utils.device import get_default_device, should_pin_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a KLDM checkpoint with sampling metrics.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


def make_fixed_subset(dataset, subset_size: int | None, seed: int) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def create_loader(config: dict[str, Any]):
    from kldm.data import CSPTask, resolve_data_root

    dataset_cfg = dict(config.get("dataset", {}) or {})
    sampling_eval_cfg = dict(config.get("sampling_eval", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})

    task_name = str(dataset_cfg.get("task", "csp")).lower()
    if task_name != "csp":
        raise ValueError(f"Unsupported task '{task_name}'. Only CSP is supported in run_sampling.")

    task = CSPTask(
        dataset_name=str(dataset_cfg.get("name", "mp20")),
        lattice_parameterization=str(model_cfg.get("lattice_parameterization", "eps")),
    )
    root = resolve_data_root(dataset_cfg.get("root"))
    split = str(sampling_eval_cfg.get("split", "test"))
    batch_size = int(sampling_eval_cfg.get("batch_size", dataset_cfg.get("batch_size", 128)))
    num_workers = int(dataset_cfg.get("num_workers", 1))
    pin_memory = bool(dataset_cfg.get("pin_memory", should_pin_memory(get_default_device())))

    dataset_full = task.fit_dataset(root=root, split=split, download=True)
    dataset = make_fixed_subset(
        dataset_full,
        subset_size=sampling_eval_cfg.get("num_targets"),
        seed=int(sampling_eval_cfg.get("subset_seed", 123)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset_full.collate_fn,
    )
    return loader, task.make_lattice_transform(root=root, download=True)


def _best_of_k(results):
    valid = any(result.valid for result in results)
    matched = [result for result in results if result.match and result.rmse is not None]
    if matched:
        return matched[min(range(len(matched)), key=lambda idx: float(matched[idx].rmse))]
    return next((result for result in results if result.valid), results[0])


def format_metric(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "na"
    return format(value, fmt)


def evaluate_sampling_at_k(
    *,
    model,
    loader,
    device: torch.device,
    sampler_cfg: dict[str, Any],
    lattice_transform,
    samples_per_target: int,
) -> dict[str, Any]:
    from kldm.sample_evaluation.sample_evaluation import (
        aggregate_csp_reconstruction_metrics,
        evaluate_csp_reconstruction,
    )

    model.eval()
    at_1_results = []
    at_k_results = []

    for batch in loader:
        batch = batch.to(device)
        per_graph_results = [[] for _ in range(batch.num_graphs)]

        for _sample_idx in range(samples_per_target):
            with torch.no_grad():
                method = str(sampler_cfg.get("method", "em"))
                if method == "pc":
                    sample_fn = model.sample_CSP_algorithm4
                else:
                    sample_fn = model.sample_CSP_algorithm3

                sample_kwargs = {
                    "n_steps": int(sampler_cfg.get("n_steps", 1000)),
                    "batch": batch,
                    "t_start": float(sampler_cfg.get("t_start", 1.0)),
                    "t_final": float(sampler_cfg.get("t_final", 1e-3)),
                }
                if method == "pc":
                    sample_kwargs.update(
                        {
                            "tau": float(sampler_cfg.get("tau", 0.25)),
                        }
                    )
                    sample_kwargs["n_correction_steps"] = int(sampler_cfg.get("n_correction_steps", 1))
                pos_t, _v_t, l_t, h_t = sample_fn(**sample_kwargs)

            ptr = batch.ptr.tolist()
            for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
                result = evaluate_csp_reconstruction(
                    pred_f=pos_t[start_idx:end_idx],
                    pred_l=l_t[graph_idx],
                    pred_a=h_t[start_idx:end_idx],
                    target_f=batch.pos[start_idx:end_idx],
                    target_l=batch.l[graph_idx],
                    target_a=batch.atomic_numbers[start_idx:end_idx],
                    lattice_transform=lattice_transform,
                )
                per_graph_results[graph_idx].append(result)

        for graph_results in per_graph_results:
            at_1_results.append(graph_results[0])
            at_k_results.append(_best_of_k(graph_results))

    at_1_summary = aggregate_csp_reconstruction_metrics(at_1_results)
    at_k_summary = aggregate_csp_reconstruction_metrics(at_k_results)

    return {
        "at_1_summary": at_1_summary,
        "at_k_summary": at_k_summary,
        "at_1_rmses": [result.rmse for result in at_1_results if result.rmse is not None],
        "at_k_rmses": [result.rmse for result in at_k_results if result.rmse is not None],
        "at_1_matches": [int(result.match) for result in at_1_results],
        "at_k_matches": [int(result.match) for result in at_k_results],
    }


def main() -> None:
    args = parse_args()
    from kldm.utils.model_loader import build_model, load_checkpoint

    config_path, config = load_experiment_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Path not found: {checkpoint_path}")

    experiment_name = str(config.get("experiment_name") or config_path.stem)
    sampler_cfg = dict(config["sampler"])
    sampling_eval_cfg = dict(config.get("sampling_eval", {}) or {})
    samples_per_target = int(sampling_eval_cfg.get("samples_per_target", 20))
    at_k_label = f"@{samples_per_target}"

    device = get_default_device()
    loader, lattice_transform = create_loader(config)
    model = build_model(config=config, device=device)
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        prefer_ema_weights=True,
    )

    run = wandb.init(
        project="mp_20_sampling",
        name=f"EVAL_{experiment_name}",
        config={
            "experiment_name": experiment_name,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "samples_per_target": samples_per_target,
            "sampler": sampler_cfg,
            "sampling_eval": sampling_eval_cfg,
        },
    )

    summary = evaluate_sampling_at_k(
        model=model,
        loader=loader,
        device=device,
        sampler_cfg=sampler_cfg,
        lattice_transform=lattice_transform,
        samples_per_target=samples_per_target,
    )

    at_1 = summary["at_1_summary"]
    at_k = summary["at_k_summary"]
    log_data = {
        "@1/valid": at_1["valid"],
        "@1/match_rate": at_1["match_rate"],
        "@1/rmse": at_1["rmse"],
        f"{at_k_label}/valid": at_k["valid"],
        f"{at_k_label}/match_rate": at_k["match_rate"],
        f"{at_k_label}/rmse": at_k["rmse"],
    }
    if summary["at_1_rmses"]:
        log_data["@1/rmse_hist"] = wandb.Histogram(summary["at_1_rmses"])
    if summary["at_k_rmses"]:
        log_data[f"{at_k_label}/rmse_hist"] = wandb.Histogram(summary["at_k_rmses"])
    log_data["@1/match_hist"] = wandb.Histogram(summary["at_1_matches"])
    log_data[f"{at_k_label}/match_hist"] = wandb.Histogram(summary["at_k_matches"])
    wandb.log(log_data)

    print(
        f"@1 valid={format_metric(at_1['valid'], '.4f')} "
        f"match_rate={format_metric(at_1['match_rate'], '.4f')} "
        f"rmse={format_metric(at_1['rmse'], '.6f')}",
        flush=True,
    )
    print(
        f"{at_k_label} valid={format_metric(at_k['valid'], '.4f')} "
        f"match_rate={format_metric(at_k['match_rate'], '.4f')} "
        f"rmse={format_metric(at_k['rmse'], '.6f')}",
        flush=True,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
