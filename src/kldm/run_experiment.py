from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import signal
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from kldm.utils.device import get_default_device, should_pin_memory
from kldm.utils.time import sample_times

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/run_experiment.py") from exc


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = WORKSPACE_ROOT / "artifacts" / "HPC" / "checkpoints" / "experiments"
TIME_LOWER_BOUND = 1e-3
STOP_REQUESTED = False


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a KLDM experiment from an exact YAML config path.")
    parser.add_argument(
        "--config",
        required=True,
        help="Exact path to the experiment YAML file.",
    )
    return parser.parse_args()


def resolve_config_path(config_name: str) -> Path:
    candidate = Path(config_name).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def resolve_relative_path(base_path: Path, path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_path.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Referenced config file not found: {candidate}")
    return candidate


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    sampler_ref = config.get("sampler_config")
    has_sampler_block = "sampler" in config
    if sampler_ref is not None and has_sampler_block:
        raise ValueError("Use either 'sampler_config' or 'sampler' in the experiment config, not both.")
    if sampler_ref is not None:
        sampler_path = resolve_relative_path(config_path, str(sampler_ref))
        config["sampler"] = load_yaml(sampler_path)
        config["sampler_config"] = str(sampler_path)
    if not isinstance(config.get("sampler"), dict):
        raise ValueError("Experiment config must define a sampler mapping or a sampler_config file.")
    return config


def make_fixed_subset(dataset, subset_size: int | None, seed: int) -> Any:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def create_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, Any]:
    from kldm.data import CSPTask, resolve_data_root

    dataset_cfg = dict(config.get("dataset", {}) or {})
    validation_cfg = dict(config.get("validation", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})


    task = CSPTask(
        dataset_name=str(dataset_cfg.get("name", "mp20")),
        lattice_parameterization=str(model_cfg.get("lattice_parameterization", "eps")),
    )

    root = resolve_data_root(dataset_cfg.get("root"))
    batch_size = int(dataset_cfg.get("batch_size", 256))
    num_workers = int(dataset_cfg.get("num_workers", 1))
    pin_memory = bool(dataset_cfg.get("pin_memory", should_pin_memory(get_default_device())))

    train_loader = task.dataloader(
        root=root,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        download=True,
    )

    val_dataset_full = task.fit_dataset(root=root, split="val", download=True)
    val_dataset = make_fixed_subset(
        val_dataset_full,
        subset_size=validation_cfg.get("subset_size"),
        seed=int(validation_cfg.get("subset_seed", 123)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_dataset_full.collate_fn,
    )
    return train_loader, val_loader, task.make_lattice_transform(root=root, download=True)


def train_epoch(
    *,
    model,
    loader,
    optimizer: torch.optim.Optimizer,
    ema,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss_v = 0.0
    total_loss_l = 0.0
    total_nodes = 0
    total_graphs = 0

    for batch in loader:
        if STOP_REQUESTED:
            break

        batch = batch.to(device)
        # Pick one shared noise time per material: t ~ Uniform(eps, 1).
        # KLDM later expands this to lattice-level and atom/node-level views.
        t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.algorithm2_loss(batch=batch, t=t_graph, debug=False)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model, current_epoch=epoch)

        total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
        total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
        total_nodes += int(batch.pos.shape[0])
        total_graphs += int(batch.num_graphs)

    if total_nodes == 0 or total_graphs == 0:
        raise RuntimeError("Training stopped before any batches were processed.")

    loss_v = total_loss_v / total_nodes
    loss_l = total_loss_l / total_graphs
    return {
        "loss_v": loss_v,
        "loss_l": loss_l,
        "loss_weighted": loss_v + loss_l,
    }


def evaluate_loss(
    *,
    model,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss_v = 0.0
    total_loss_l = 0.0
    total_nodes = 0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        # Validation uses the same one-time-per-material sampling as training.
        # This keeps coordinates, velocities, and lattice at one common noise level.
        t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)
        with torch.no_grad():
            _, metrics = model.algorithm2_loss(batch=batch, t=t_graph, debug=False)

        total_loss_v += float(metrics["loss_v"]) * int(batch.pos.shape[0])
        total_loss_l += float(metrics["loss_l"]) * int(batch.num_graphs)
        total_nodes += int(batch.pos.shape[0])
        total_graphs += int(batch.num_graphs)

    if total_nodes == 0 or total_graphs == 0:
        raise RuntimeError("Validation loader is empty.")

    loss_v = total_loss_v / total_nodes
    loss_l = total_loss_l / total_graphs
    return {
        "loss_v": loss_v,
        "loss_l": loss_l,
        "loss_weighted": loss_v + loss_l,
    }


def run_sampling_evaluation(
    *,
    model,
    loader,
    device: torch.device,
    sampler_cfg: dict[str, Any],
    lattice_transform,
    max_graphs: int | None,
) -> dict[str, float | int | None]:
    from kldm.sample_evaluation.sample_evaluation import (
        aggregate_csp_reconstruction_metrics,
        evaluate_csp_reconstruction,
    )

    model.eval()
    results = []
    num_graphs_seen = 0

    for batch in loader:
        batch = batch.to(device)
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
            results.append(result)
            num_graphs_seen += 1
            if max_graphs is not None and num_graphs_seen >= max_graphs:
                break
        if max_graphs is not None and num_graphs_seen >= max_graphs:
            break

    summary = aggregate_csp_reconstruction_metrics(results)
    return {
        "valid": summary.get("valid"),
        "match_rate": summary.get("match_rate"),
        "rmse": summary.get("rmse"),
        "num_samples": summary.get("num_samples"),
    }


def should_stop(run) -> bool:
    if STOP_REQUESTED:
        return True
    if run is None:
        return False
    for attr in ("stopped", "_stopped"):
        value = getattr(run, attr, None)
        if isinstance(value, bool) and value:
            return True
    return False


def build_run_name() -> str:
    now = datetime.now()
    return f"trial_{now.strftime('%Y%m%d')}"


def format_metric(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "na"
    return format(value, fmt)


def checkpoint_dir(config: dict[str, Any], experiment_name: str) -> Path:
    checkpoint_cfg = dict(config.get("checkpoint", {}) or {})
    root = checkpoint_cfg.get("dir")
    if root is None:
        return CHECKPOINTS_ROOT / experiment_name
    return Path(str(root)).expanduser()


def save_named_checkpoint(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    ema,
    config: dict[str, Any],
    experiment_name: str,
    epoch: int,
    metrics: dict[str, float],
    filename: str,
) -> Path:
    from kldm.utils.model_loader import save_checkpoint

    output_dir = checkpoint_dir(config=config, experiment_name=experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
        output_path=output_path,
        config=config,
        epoch=epoch,
        metrics=metrics,
    )
    for candidate in output_dir.glob("*.pt"):
        if candidate.name not in {"last.pt", "final.pt"}:
            candidate.unlink(missing_ok=True)
    return output_path


def save_wandb_checkpoint(path: Path) -> None:
    if path.exists():
        wandb.save(str(path), policy="now")


def main() -> None:
    args = parse_args()
    from kldm.utils.model_loader import build_training_components, load_checkpoint

    config_path = resolve_config_path(args.config)
    config = load_experiment_config(config_path)

    experiment_name = str(config.get("experiment_name") or config_path.stem)
    sampler_cfg = dict(config["sampler"])
    logging_cfg = dict(config.get("logging", {}) or {})
    validation_cfg = dict(config.get("validation", {}) or {})
    checkpoint_cfg = dict(config.get("checkpoint", {}) or {})

    train_every_epochs = int(logging_cfg.get("train_every_epochs", 1))
    validate_every_epochs = int(validation_cfg.get("every_n_epochs", 100))
    if train_every_epochs <= 0 or validate_every_epochs <= 0:
        raise ValueError("Logging and validation intervals must be positive integers.")

    device = get_default_device()
    train_loader, val_loader, lattice_transform = create_loaders(config)
    components = build_training_components(config=config, device=device)
    model = components.model
    optimizer = components.optimizer
    ema = components.ema
    start_epoch = 0

    resume_from = checkpoint_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_relative_path(config_path, str(resume_from))
        checkpoint = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            ema=ema,
            device=device,
            prefer_ema_weights=False,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"resumed checkpoint={resume_path} epoch={start_epoch}", flush=True)

    run = wandb.init(
        project=experiment_name,
        name=build_run_name(),
        config=config | {"start_epoch": start_epoch},
    )

    print(f"run_experiment config={config_path}", flush=True)
    print(f"device={device.type} experiment={experiment_name}", flush=True)
    print(f"sampler={sampler_cfg}", flush=True)

    epoch = start_epoch + 1
    try:
        while not should_stop(run):
            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                ema=ema,
                device=device,
                epoch=epoch,
            )

            if epoch % train_every_epochs == 0:
                train_log = {
                    "epoch": epoch,
                    "train/loss_v": train_metrics["loss_v"],
                    "train/loss_l": train_metrics["loss_l"],
                    "train/loss_weighted": train_metrics["loss_weighted"],
                }
                wandb.log(train_log, step=epoch)
                print(
                    f"epoch={epoch:04d} train_loss_weighted={train_metrics['loss_weighted']:.6f} "
                    f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f})",
                    flush=True,
                )

            if epoch % validate_every_epochs == 0 and not should_stop(run):
                ema_val = bool(validation_cfg.get("ema_val", validation_cfg.get("use_ema", False)))
                use_ema = ema_val and ema is not None and ema.num_updates > 0
                context = ema.average_parameters(model) if use_ema else nullcontext()
                model_label = "EMA model" if use_ema else "current model"

                print(f"epoch={epoch:04d} entering validation with {model_label}", flush=True)
                with context:
                    val_loss_metrics = evaluate_loss(
                        model=model,
                        loader=val_loader,
                        device=device,
                    )
                    val_sample_metrics = run_sampling_evaluation(
                        model=model,
                        loader=val_loader,
                        device=device,
                        sampler_cfg=sampler_cfg,
                        lattice_transform=lattice_transform,
                        max_graphs=validation_cfg.get("sampling_max_graphs"),
                    )

                merged_metrics = {
                    "loss_v": val_loss_metrics["loss_v"],
                    "loss_l": val_loss_metrics["loss_l"],
                    "loss_weighted": val_loss_metrics["loss_weighted"],
                    "valid": val_sample_metrics["valid"],
                    "match_rate": val_sample_metrics["match_rate"],
                    "rmse": val_sample_metrics["rmse"],
                }
                wandb.log(
                    {
                        "epoch": epoch,
                        "val/loss_v": merged_metrics["loss_v"],
                        "val/loss_l": merged_metrics["loss_l"],
                        "val/loss_weighted": merged_metrics["loss_weighted"],
                        "val/valid": merged_metrics["valid"],
                        "val/match_rate": merged_metrics["match_rate"],
                        "val/rmse": merged_metrics["rmse"],
                    },
                    step=epoch,
                )
                checkpoint_path = save_named_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    config=config,
                    experiment_name=experiment_name,
                    epoch=epoch,
                    metrics=merged_metrics,
                    filename="last.pt",
                )
                if bool(logging_cfg.get("wandb_checkpoints", True)):
                    save_wandb_checkpoint(checkpoint_path)
                print(
                    f"validation_epoch={epoch:04d} val_loss_weighted={merged_metrics['loss_weighted']:.6f} "
                    f"(loss_v={merged_metrics['loss_v']:.6f}, loss_l={merged_metrics['loss_l']:.6f}) "
                    f"valid={format_metric(merged_metrics['valid'], '.4f')} "
                    f"match_rate={format_metric(merged_metrics['match_rate'], '.4f')} "
                    f"rmse={format_metric(merged_metrics['rmse'], '.6f')}",
                    flush=True,
                )
                print(f"checkpoint_saved={checkpoint_path}", flush=True)

            epoch += 1
    except KeyboardInterrupt:
        print("run_experiment interrupted", flush=True)
    finally:
        final_metrics = {"final_epoch": float(max(epoch - 1, start_epoch))}
        final_path = save_named_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            config=config,
            experiment_name=experiment_name,
            epoch=max(epoch - 1, start_epoch),
            metrics=final_metrics,
            filename="final.pt",
        )
        if bool(logging_cfg.get("wandb_checkpoints", True)):
            save_wandb_checkpoint(final_path)
        wandb.finish()


if __name__ == "__main__":
    main()
