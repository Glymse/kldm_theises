from __future__ import annotations

import argparse
import csv
import errno
import os
from pathlib import Path
import signal
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from kldm.utils.device import get_default_device, should_pin_memory
from kldm.utils.ema import ExponentialMovingAverage
from kldm.utils.model_loader import load_checkpoint, save_checkpoint
from kldm.utils.time import sample_times

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/trainHPC.py") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


STOP_REQUESTED = False
TIME_LOWER_BOUND = 1e-3
VAL_SUBSET_SEED = 123
LOADER_NUM_WORKERS = 1
NODE_AVERAGED_METRICS = {
    "loss_v",
    "target_v_abs_mean",
    "target_v_norm_mean",
    "pred_v_abs_mean",
    "pred_v_norm_mean",
    "v_t_abs_mean",
    "f_t_abs_mean",
    "r_t_abs_mean",
    "score_v_abs_mean",
}

GRAPH_AVERAGED_METRICS = {
    "loss_l",
    "pred_l_abs_mean",
    "target_l_abs_mean",
}

BATCH_AVERAGED_METRICS = {
    "grad_norm",
    "score_network_grad_norm",
    "out_v_grad_norm",
    "out_l_grad_norm",
}


def _request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)


def should_stop(run) -> bool:
    if STOP_REQUESTED:
        return True

    for attr in ("stopped", "_stopped"):
        value = getattr(run, attr, None)
        if isinstance(value, bool) and value:
            return True

    return False


def make_fixed_subset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def aggregate_epoch_metrics(
    batch_metrics: list[tuple[dict[str, float], float, float]],
) -> dict[str, float]:
    if not batch_metrics:
        raise RuntimeError("Cannot aggregate empty metric list.")

    totals_node: dict[str, float] = {}
    totals_graph: dict[str, float] = {}
    totals_batch: dict[str, float] = {}
    totals_other_graph: dict[str, float] = {}
    total_nodes = 0.0
    total_graphs = 0.0
    total_batches = 0.0

    for metrics, num_nodes, num_graphs in batch_metrics:
        total_nodes += num_nodes
        total_graphs += num_graphs
        total_batches += 1.0

        for key, value in metrics.items():
            if key == "loss":
                continue
            if key in NODE_AVERAGED_METRICS:
                totals_node[key] = totals_node.get(key, 0.0) + float(value) * num_nodes
            elif key in GRAPH_AVERAGED_METRICS:
                totals_graph[key] = totals_graph.get(key, 0.0) + float(value) * num_graphs
            elif key in BATCH_AVERAGED_METRICS:
                totals_batch[key] = totals_batch.get(key, 0.0) + float(value)
            else:
                totals_other_graph[key] = totals_other_graph.get(key, 0.0) + float(value) * num_graphs

    aggregated: dict[str, float] = {}
    if total_nodes > 0.0:
        for key, total in totals_node.items():
            aggregated[key] = total / total_nodes
    if total_graphs > 0.0:
        for key, total in totals_graph.items():
            aggregated[key] = total / total_graphs
        for key, total in totals_other_graph.items():
            aggregated[key] = total / total_graphs
    if total_batches > 0.0:
        for key, total in totals_batch.items():
            aggregated[key] = total / total_batches

    aggregated["loss_v"] = aggregated.get("loss_v", 0.0)
    aggregated["loss_l"] = aggregated.get("loss_l", 0.0)
    aggregated["loss"] = aggregated["loss_v"] + aggregated["loss_l"]
    return aggregated


#####
#####
#####


def validation_step(
    model: ModelKLDM,
    batch,
    device: torch.device,
    debug: bool = False,
) -> dict[str, float]:
    model.eval()
    batch = batch.to(device)

    # Pick one shared noise time per material: t ~ Uniform(eps, 1).
    # KLDM later expands this to lattice-level and atom/node-level views.
    t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

    with torch.no_grad():
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            debug=debug,
        )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


def train_epoch(
    model: ModelKLDM,
    loader,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    device: torch.device,
    epoch: int,
    debug: bool = False,
) -> dict[str, float]:
    model.train()
    batch_metrics: list[tuple[dict[str, float], float, float]] = []

    for batch in loader:
        if STOP_REQUESTED:
            break

        batch = batch.to(device)
        num_graphs = float(batch.num_graphs)
        num_nodes = float(batch.pos.shape[0])
        # Pick one shared noise time per material: t ~ Uniform(eps, 1).
        # This is the noise level used by Algorithm 1 for the whole crystal state.
        t_graph = sample_times(batch, lower_bound=TIME_LOWER_BOUND)

        optimizer.zero_grad()
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            debug=debug,
        )
        loss.backward()
        if debug:
            total_grad_sq = 0.0
            score_network_grad_sq = 0.0
            out_v_grad_sq = 0.0
            out_l_grad_sq = 0.0
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad_sq = float(param.grad.detach().pow(2).sum().item())
                total_grad_sq += grad_sq
                if name.startswith("score_network"):
                    score_network_grad_sq += grad_sq
                if name.startswith("score_network.out_v"):
                    out_v_grad_sq += grad_sq
                if name.startswith("score_network.out_l"):
                    out_l_grad_sq += grad_sq
            metrics = dict(metrics)
            metrics.update(
                {
                    "grad_norm": total_grad_sq ** 0.5,
                    "score_network_grad_norm": score_network_grad_sq ** 0.5,
                    "out_v_grad_norm": out_v_grad_sq ** 0.5,
                    "out_l_grad_norm": out_l_grad_sq ** 0.5,
                }
            )
        optimizer.step()
        ema.update(model, current_epoch=epoch)
        batch_metrics.append(
            ({key: float(value) for key, value in metrics.items()}, num_nodes, num_graphs)
        )

    if not batch_metrics:
        raise RuntimeError("Training loader is empty or interrupted before any batch was processed.")
    return aggregate_epoch_metrics(batch_metrics)


def evaluate_loss(
    model: ModelKLDM,
    loader,
    device: torch.device,
    max_graphs: int | None = None,
    debug: bool = False,
) -> dict[str, float]:
    num_graphs_seen = 0.0
    batch_metrics: list[tuple[dict[str, float], float, float]] = []

    for batch in loader:
        metrics = validation_step(
            model=model,
            batch=batch,
            device=device,
            debug=debug,
        )
        num_graphs = float(batch.num_graphs)
        num_nodes = float(batch.pos.shape[0])
        batch_metrics.append((metrics, num_nodes, num_graphs))
        num_graphs_seen += num_graphs
        if max_graphs is not None and num_graphs_seen >= max_graphs:
            break

    if not batch_metrics:
        raise RuntimeError("Validation loader is empty; cannot compute epoch metrics.")
    return aggregate_epoch_metrics(batch_metrics)


def is_no_space_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return exc.errno == errno.ENOSPC
    message = str(exc).lower()
    return "no space left on device" in message


def resolve_artifacts_root() -> Path:
    override = os.environ.get("KLDM_ARTIFACTS_DIR")
    if override:
        return Path(override).expanduser()

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        return Path(tmpdir) / "kldm_artifacts"

    return Path("artifacts") / "HPC"


def export_model_checkpoint(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
) -> None:
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
        output_path=output_path,
        config=config,
        epoch=epoch,
        metrics={},
    )


def try_export_model_checkpoint(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
) -> bool:
    try:
        export_model_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            output_path=output_path,
            config=config,
            epoch=epoch,
        )
        return True
    except Exception as exc:
        if is_no_space_error(exc):
            print(
                f"warning: skipping checkpoint export due to no space left: {output_path}",
                flush=True,
            )
            return False
        raise


def export_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_export_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> bool:
    try:
        export_csv(rows=rows, output_path=output_path, fieldnames=fieldnames)
        return True
    except Exception as exc:
        if is_no_space_error(exc):
            print(
                f"warning: skipping csv export due to no space left: {output_path}",
                flush=True,
            )
            return False
        raise


def maybe_plot_training_metrics(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="Val loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_loss_v"] for row in rows], marker="o", label="Train loss_v")
    axes[1].plot(epochs, [row["val_loss_v"] for row in rows], marker="o", label="Val loss_v")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Velocity loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [row["train_loss_l"] for row in rows], marker="o", label="Train loss_l")
    axes[2].plot(epochs, [row["val_loss_l"] for row in rows], marker="o", label="Val loss_l")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Lattice loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def maybe_plot_sampling_metrics(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not rows:
        return None

    rows = [row for row in rows if row.get("valid") == row.get("valid")]
    if not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(epochs, [row["valid"] for row in rows], marker="o", label="valid")
    axes[0].plot(epochs, [row["match_rate"] for row in rows], marker="o", label="match_rate")
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Validation sampling metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["rmse"] for row in rows], marker="o", label="rmse")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="val/loss")
    axes[2].plot(epochs, [row["val_loss_v"] for row in rows], marker="o", label="val/loss_v")
    axes[2].plot(epochs, [row["val_loss_l"] for row in rows], marker="o", label="val/loss_l")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Validation losses at checkpoint epochs")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
def run_sampling_evaluation(
    model: ModelKLDM,
    loader,
    device: torch.device,
    n_steps: int,
    max_graphs: int | None = None,
) -> dict[str, float | int]:
    from kldm.sample_evaluation.sample_evaluation import (
        aggregate_csp_reconstruction_metrics,
        evaluate_csp_reconstruction,
    )

    reconstruction_results = []
    num_graphs_seen = 0

    model.eval()

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
                n_steps=n_steps,
                batch=batch,
            )

        ptr = batch.ptr.tolist()
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            result = evaluate_csp_reconstruction(
                pred_f=pos_t[start:end],
                pred_l=l_t[graph_idx],
                pred_a=h_t[start:end],
                target_f=batch.pos[start:end],
                target_l=batch.l[graph_idx],
                target_a=batch.atomic_numbers[start:end],
            )
            reconstruction_results.append(result)
            num_graphs_seen += 1

            if max_graphs is not None and num_graphs_seen >= max_graphs:
                break
        if max_graphs is not None and num_graphs_seen >= max_graphs:
            break

    if not reconstruction_results:
        raise RuntimeError("Validation sampling produced no reconstruction results.")

    summary = aggregate_csp_reconstruction_metrics(reconstruction_results)
    return {
        "num_samples": int(summary["num_samples"]),
        "valid": float("nan") if summary["valid"] is None else float(summary["valid"]),
        "match_rate": float("nan") if summary["match_rate"] is None else float(summary["match_rate"]),
        "rmse": float("nan") if summary["rmse"] is None else float(summary["rmse"]),
    }

def should_use_ema_for_sampling(
    ema: ExponentialMovingAverage | None,
    *,
    current_epoch: int,
    force_ema: bool,
) -> bool:
    if ema is None:
        return False
    if ema.num_updates <= 0:
        return False
    return bool(force_ema)


def maybe_resume(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    if checkpoint_path is None:
        return 0, {}

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        device=device,
        prefer_ema_weights=False,
    )
    start_epoch = int(checkpoint.get("epoch", 0))
    checkpoint_config = checkpoint.get("config", {})

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {start_epoch}")

    return start_epoch, checkpoint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-ended KLDM HPC training with checkpointed sampling validation."
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to continue from, e.g. artifacts/HPC/checkpoints/checkpoint_epoch_100.pt",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=100,
        help="Run validation, save checkpoint, and plot every N epochs.",
    )
    parser.add_argument(
        "--loss-every",
        type=int,
        default=1,
        help="Deprecated. Train losses are recorded every epoch.",
    )
    parser.add_argument(
        "--sampling-samples",
        type=int,
        default=25,
        help="Deprecated. Facit-style validation sampling now uses the validation subset instead.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=1000,
        help="Number of Algorithm 3 steps for each sampled structure.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training and loss-validation batch size.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of the training split to use, e.g. 0.05 for 5%% of train data.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. Defaults to facitKLDM's 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-12,
        help="AdamW weight decay. Defaults to facitKLDM's 1e-12.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional safety cap on epochs. Omit to run until stopped.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay used for the shadow model during HPC training.",
    )
    parser.add_argument(
        "--ema-start",
        type=int,
        default=500,
        help="Epoch threshold after which EMA starts updating, matching facitKLDM semantics.",
    )
    parser.add_argument(
        "--sampling-force-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force EMA weights for validation sampling. "
            "Defaults to false, so validation and sampling stay on the live model unless "
            "you explicitly enable EMA."
        ),
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable detailed KLDM/TDM convergence diagnostics in logs and W&B.",
    )
    return parser.parse_args()


def train() -> None:
    from kldm.data import CSPTask, resolve_data_root
    from kldm.kldm import ModelKLDM

    args = parse_args()
    if not (0.0 < args.subset <= 1.0):
        raise ValueError("--subset must be in the interval (0, 1].")

    device = get_default_device()
    root = resolve_data_root()
    print(f"starting trainHPC root={root} device={device.type}", flush=True)

    config = {
        "task": "CSP",
        "batch_size": args.batch_size,
        "subset": args.subset,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "validate_every": args.validate_every,
        "loss_every": args.loss_every,
        "sampling_samples": args.sampling_samples,
        "sampling_steps": args.sampling_steps,
        "val_subset_graphs": 1024,
        "val_subset_seed": VAL_SUBSET_SEED,
        "load_from_checkpoint": args.load_from_checkpoint,
        "max_epochs": args.max_epochs,
        "ema_decay": args.ema_decay,
        "ema_start": args.ema_start,
        "sampling_force_ema": args.sampling_force_ema,
        "num_workers": LOADER_NUM_WORKERS,
        "pin_memory": should_pin_memory(device),
        "dev": args.dev,
        "score_network": {
            "hidden_dim": 512,
            "time_dim": 256,
            "num_layers": 6,
            "num_freqs": 128,
            "ln": True,
            "h_dim": 100,
            "smooth": False,
            "pred_v": True,
            "pred_l": True,
            "pred_h": False,
            "zero_cog": True,
        },
    }

    csp_task = CSPTask()
    train_dataset_full = csp_task.fit_dataset(
        root=root,
        split="train",
        download=True,
    )
    train_subset_size = max(1, int(len(train_dataset_full) * config["subset"]))
    train_dataset = make_fixed_subset(
        train_dataset_full,
        subset_size=train_subset_size,
        seed=VAL_SUBSET_SEED,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=train_dataset_full.collate_fn,
    )

    val_dataset_full = csp_task.fit_dataset(
        root=root,
        split="val",
        download=True,
    )
    val_dataset = make_fixed_subset(
        val_dataset_full,
        subset_size=config["val_subset_graphs"],
        seed=config["val_subset_seed"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=val_dataset_full.collate_fn,
    )

    print("constructed train/val loaders", flush=True)

    model = ModelKLDM(
        device=device,
        eps=1e-6,
        tdm_n_sigmas=2000 if device.type == "cuda" else 512,
        score_network_kwargs=config["score_network"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        amsgrad=True,
        foreach=device.type == "cuda",
        weight_decay=config["weight_decay"],
    )
    ema = ExponentialMovingAverage(
        model=model,
        decay=config["ema_decay"],
        start_epoch=config["ema_start"],
    )
    print(f"constructed model and optimizer tdm={type(model.tdm).__name__}", flush=True)

    start_epoch, resume_config = maybe_resume(
        model=model,
        optimizer=optimizer,
        ema=ema,
        checkpoint_path=args.load_from_checkpoint,
        device=device,
    )

    run_name = (
        f"csp_hpc_continue_from_{start_epoch}"
        if args.load_from_checkpoint is not None
        else "csp_hpc_open_ended"
    )

    run = wandb.init(
        project="kldm-csp-hpc",
        config=config | {"resume_config": resume_config, "start_epoch": start_epoch},
        name=run_name,
    )
    print(
        f"wandb initialized run_name={run_name} stopped={getattr(run, 'stopped', None)} "
        f"_stopped={getattr(run, '_stopped', None)} state={getattr(run, 'state', None)}",
        flush=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    artifacts_dir = resolve_artifacts_root()
    checkpoints_dir = artifacts_dir / "checkpoints"
    history_path = artifacts_dir / "training_history.csv"
    sampling_history_path = artifacts_dir / "sampling_history.csv"
    training_plot_path = artifacts_dir / "training_metrics.png"
    sampling_plot_path = artifacts_dir / "sampling_metrics.png"
    print(f"using artifacts_dir={artifacts_dir}", flush=True)

    history: list[dict[str, Any]] = []
    epoch = start_epoch

    try:
        while True:
            if args.max_epochs is not None and epoch >= args.max_epochs:
                print(f"Reached max_epochs={args.max_epochs}. Stopping.")
                break

            if should_stop(run):
                print("Stop requested. Finishing cleanly.")
                break

            epoch += 1
            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                ema=ema,
                device=device,
                epoch=epoch,
                debug=args.dev,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_loss_v": train_metrics["loss_v"],
                "train_loss_l": train_metrics["loss_l"],
                "val_loss": float("nan"),
                "val_loss_v": float("nan"),
                "val_loss_l": float("nan"),
                "valid": float("nan"),
                "match_rate": float("nan"),
                "rmse": float("nan"),
            }
            history.append(row)

            if args.dev:
                print(
                    f"epoch={epoch:04d} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"(v={train_metrics['loss_v']:.6f}, l={train_metrics['loss_l']:.6f}) "
                    f"target_v_abs={train_metrics['target_v_abs_mean']:.6f} "
                    f"pred_v_abs={train_metrics['pred_v_abs_mean']:.6f} "
                    f"grad=[total={train_metrics['grad_norm']:.3f},v={train_metrics['out_v_grad_norm']:.3f},l={train_metrics['out_l_grad_norm']:.3f}]"
                )
            else:
                print(
                    f"epoch={epoch:04d} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f})"
                )

            should_run_sampling = (epoch % config["validate_every"] == 0)

            train_log_payload = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/loss_v": train_metrics["loss_v"],
                "train/loss_l": train_metrics["loss_l"],
            }
            if args.dev:
                train_log_payload.update(
                    {
                        "train/target_v_abs_mean": train_metrics["target_v_abs_mean"],
                        "train/target_v_norm_mean": train_metrics["target_v_norm_mean"],
                        "train/pred_v_abs_mean": train_metrics["pred_v_abs_mean"],
                        "train/pred_v_norm_mean": train_metrics["pred_v_norm_mean"],
                        "train/v_t_abs_mean": train_metrics["v_t_abs_mean"],
                        "train/f_t_abs_mean": train_metrics["f_t_abs_mean"],
                        "train/r_t_abs_mean": train_metrics["r_t_abs_mean"],
                        "train/score_v_abs_mean": train_metrics["score_v_abs_mean"],
                        "train/pred_l_abs_mean": train_metrics["pred_l_abs_mean"],
                        "train/target_l_abs_mean": train_metrics["target_l_abs_mean"],
                        "train/grad_norm": train_metrics["grad_norm"],
                        "train/score_network_grad_norm": train_metrics["score_network_grad_norm"],
                        "train/out_v_grad_norm": train_metrics["out_v_grad_norm"],
                        "train/out_l_grad_norm": train_metrics["out_l_grad_norm"],
                    }
                )
            wandb.log(train_log_payload)
            try_export_csv(
                rows=history,
                output_path=history_path,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_loss_v",
                    "train_loss_l",
                    "val_loss",
                    "val_loss_v",
                    "val_loss_l",
                    "valid",
                    "match_rate",
                    "rmse",
                ],
            )

            if not should_run_sampling:
                continue

            phase_bits = []
            phase_bits.append("validation")
            phase_bits.append("sampling")
            print(
                f"epoch={epoch:04d} entering {' + '.join(phase_bits)}",
                flush=True,
            )

            use_ema_sampling = should_use_ema_for_sampling(
                ema=ema,
                current_epoch=epoch,
                force_ema=config["sampling_force_ema"],
            )
            if use_ema_sampling:
                print(
                    f"epoch={epoch:04d} validation + sampling with EMA model",
                    flush=True,
                )
                with ema.average_parameters(model):
                    print(
                        f"epoch={epoch:04d} starting validation loss pass",
                        flush=True,
                    )
                    val_metrics = evaluate_loss(
                        model=model,
                        loader=val_loader,
                        device=device,
                        max_graphs=config["val_subset_graphs"],
                        debug=args.dev,
                    )
                    print(
                        f"epoch={epoch:04d} finished validation loss pass",
                        flush=True,
                    )
                    print(
                        f"epoch={epoch:04d} starting sampling evaluation",
                        flush=True,
                    )
                    sampling_metrics = run_sampling_evaluation(
                        model=model,
                        loader=val_loader,
                        device=device,
                        n_steps=config["sampling_steps"],
                        max_graphs=config["val_subset_graphs"],
                    )
            else:
                print(
                    f"epoch={epoch:04d} validation + sampling with current model",
                    flush=True,
                )
                print(
                    f"epoch={epoch:04d} starting validation loss pass",
                    flush=True,
                )
                val_metrics = evaluate_loss(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_graphs=config["val_subset_graphs"],
                    debug=args.dev,
                )
                print(
                    f"epoch={epoch:04d} finished validation loss pass",
                    flush=True,
                )
                print(
                    f"epoch={epoch:04d} starting sampling evaluation",
                    flush=True,
                )
                sampling_metrics = run_sampling_evaluation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    n_steps=config["sampling_steps"],
                    max_graphs=config["val_subset_graphs"],
                )
            print(
                f"epoch={epoch:04d} finished sampling evaluation",
                flush=True,
            )

            row["val_loss"] = val_metrics["loss"]
            row["val_loss_v"] = val_metrics["loss_v"]
            row["val_loss_l"] = val_metrics["loss_l"]
            row["valid"] = sampling_metrics["valid"]
            row["match_rate"] = sampling_metrics["match_rate"]
            row["rmse"] = sampling_metrics["rmse"]

            if args.dev:
                print(
                    f"validation_epoch={epoch:04d} "
                    f"val_loss={val_metrics['loss']:.6f} "
                    f"(loss_v={val_metrics['loss_v']:.6f}, loss_l={val_metrics['loss_l']:.6f}) "
                    f"valid={sampling_metrics['valid']:.4f} "
                    f"match_rate={sampling_metrics['match_rate']:.4f} "
                    f"rmse={sampling_metrics['rmse']:.6f}"
                )
            else:
                print(
                    f"validation_epoch={epoch:04d} "
                    f"val_loss={val_metrics['loss']:.6f} "
                    f"(loss_v={val_metrics['loss_v']:.6f}, loss_l={val_metrics['loss_l']:.6f}) "
                    f"valid={sampling_metrics['valid']:.4f} "
                    f"match_rate={sampling_metrics['match_rate']:.4f} "
                    f"rmse={sampling_metrics['rmse']:.6f}"
                )

            print(
                f"checkpoint_epoch={epoch:04d} "
                f"valid={100.0 * sampling_metrics['valid']:.2f}% "
                f"match_rate={100.0 * sampling_metrics['match_rate']:.2f}% "
                f"mean_rmse={sampling_metrics['rmse']:.6f}"
            )

            checkpoint_path = checkpoints_dir / "last.pt"
            print(
                f"epoch={epoch:04d} exporting checkpoint",
                flush=True,
            )
            checkpoint_written = try_export_model_checkpoint(
                model=model,
                optimizer=optimizer,
                ema=ema,
                output_path=checkpoint_path,
                config=config,
                epoch=epoch,
            )
            if checkpoint_written:
                for candidate in checkpoints_dir.glob("checkpoint_epoch_*.pt"):
                    candidate.unlink(missing_ok=True)

            val_log_payload = {
                "epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/loss_v": val_metrics["loss_v"],
                "val/loss_l": val_metrics["loss_l"],
                "val/valid": sampling_metrics["valid"],
                "val/match_rate": sampling_metrics["match_rate"],
                "val/rmse": sampling_metrics["rmse"],
            }
            if args.dev:
                val_log_payload.update(
                    {
                        "val/target_v_abs_mean": val_metrics["target_v_abs_mean"],
                        "val/target_v_norm_mean": val_metrics["target_v_norm_mean"],
                        "val/pred_v_abs_mean": val_metrics["pred_v_abs_mean"],
                        "val/pred_v_norm_mean": val_metrics["pred_v_norm_mean"],
                        "val/v_t_abs_mean": val_metrics["v_t_abs_mean"],
                        "val/f_t_abs_mean": val_metrics["f_t_abs_mean"],
                        "val/r_t_abs_mean": val_metrics["r_t_abs_mean"],
                        "val/score_v_abs_mean": val_metrics["score_v_abs_mean"],
                        "val/pred_l_abs_mean": val_metrics["pred_l_abs_mean"],
                        "val/target_l_abs_mean": val_metrics["target_l_abs_mean"],
                    }
                )
            wandb.log(val_log_payload)

            try_export_csv(
                rows=history,
                output_path=history_path,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_loss_v",
                    "train_loss_l",
                    "val_loss",
                    "val_loss_v",
                    "val_loss_l",
                    "valid",
                    "match_rate",
                    "rmse",
                ],
            )
            try_export_csv(
                rows=[
                    {
                        "epoch": row["epoch"],
                        "val_loss": row["val_loss"],
                        "val_loss_v": row["val_loss_v"],
                        "val_loss_l": row["val_loss_l"],
                        "valid": row["valid"],
                        "match_rate": row["match_rate"],
                        "rmse": row["rmse"],
                    }
                    for row in history
                    if row["valid"] == row["valid"]
                ],
                output_path=sampling_history_path,
                fieldnames=[
                    "epoch",
                    "val_loss",
                    "val_loss_v",
                    "val_loss_l",
                    "valid",
                    "match_rate",
                    "rmse",
                ],
            )
            print(
                f"epoch={epoch:04d} writing training metrics plot",
                flush=True,
            )
            try:
                maybe_plot_training_metrics(history, training_plot_path)
            except Exception as exc:
                if is_no_space_error(exc):
                    print(
                        f"epoch={epoch:04d} warning: skipping training plot due to no space left",
                        flush=True,
                    )
                else:
                    raise
            print(
                f"epoch={epoch:04d} finished training metrics plot",
                flush=True,
            )
            print(
                f"epoch={epoch:04d} writing sampling metrics plot",
                flush=True,
            )
            try:
                maybe_plot_sampling_metrics(history, sampling_plot_path)
            except Exception as exc:
                if is_no_space_error(exc):
                    print(
                        f"epoch={epoch:04d} warning: skipping sampling plot due to no space left",
                        flush=True,
                    )
                else:
                    raise
            print(
                f"epoch={epoch:04d} finished sampling metrics plot",
                flush=True,
            )

            if run is not None and checkpoint_written:
                try:
                    print(
                        f"epoch={epoch:04d} saving last checkpoint to wandb",
                        flush=True,
                    )
                    wandb.save(str(checkpoint_path), policy="now")
                    print(
                        f"epoch={epoch:04d} finished wandb sync for validation",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"epoch={epoch:04d} warning: wandb artifact/save failed: {exc}",
                        flush=True,
                    )

            print(
                f"epoch={epoch:04d} validation block complete",
                flush=True,
            )

    finally:
        final_checkpoint_path = checkpoints_dir / "final.pt"
        final_checkpoint_written = try_export_model_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            output_path=final_checkpoint_path,
            config=config | {"completed_epochs": epoch},
            epoch=epoch,
        )

        if run is not None:
            if final_checkpoint_written and final_checkpoint_path.exists():
                wandb.save(str(final_checkpoint_path), policy="now")
            wandb.finish()


if __name__ == "__main__":
    train()
