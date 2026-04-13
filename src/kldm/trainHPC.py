from __future__ import annotations

import argparse
import csv
import itertools
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import CSPTask, resolve_data_root
from kldm.distribution.uniform import sample_uniform
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    aggregate_csp_reconstruction_metrics,
    evaluate_csp_reconstruction,
)

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/trainHPC.py") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


STOP_REQUESTED = False


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


class ExponentialMovingAverage:
    def __init__(
        self,
        model: ModelKLDM,
        decay: float = 0.999,
        start_step: int = 500,
    ) -> None:
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.num_updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: ModelKLDM) -> None:
        self.num_updates += 1

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.num_updates <= self.start_step:
                self.shadow[name].copy_(param.detach())
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.start_step = int(state_dict.get("start_step", self.start_step))
        self.num_updates = int(state_dict.get("num_updates", 0))
        self.shadow = {
            name: tensor.clone()
            for name, tensor in state_dict.get("shadow", {}).items()
        }

    def ema_model_state_dict(self, model: ModelKLDM) -> dict[str, torch.Tensor]:
        ema_state = clean_model_state_dict(model)
        for name, tensor in self.shadow.items():
            if name in ema_state:
                ema_state[name] = tensor.clone()
        return ema_state

    @contextmanager
    def average_parameters(self, model: ModelKLDM):
        if not self.shadow:
            yield
            return

        self.backup = {}
        try:
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow:
                    continue
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
            yield
        finally:
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
            self.backup = {}

#####
#####
#####


def validation_step(
    model: ModelKLDM,
    batch,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batch = batch.to(device)

    t_graph = sample_uniform(
        lb=model.diffusion_l.eps,
        size=(batch.num_graphs, 1),
        device=device,
    )

    with torch.no_grad():
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=1.0,
        )

    return {
        "loss": float(loss),
        "loss_v": float(metrics["loss_v"]),
        "loss_l": float(metrics["loss_l"]),
    }


def train_epoch(
    model: ModelKLDM,
    loader,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0}
    num_batches = 0

    for batch in loader:
        if STOP_REQUESTED:
            break

        batch = batch.to(device)
        t_graph = sample_uniform(
            lb=model.diffusion_l.eps,
            size=(batch.num_graphs, 1),
            device=device,
        )

        optimizer.zero_grad()
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=1.0,
        )
        loss.backward()
        optimizer.step()
        ema.update(model)

        running["loss"] += float(loss)
        running["loss_v"] += float(metrics["loss_v"])
        running["loss_l"] += float(metrics["loss_l"])
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training loader is empty or interrupted before any batch was processed.")

    for key in running:
        running[key] /= num_batches
    return running


def evaluate_loss(
    model: ModelKLDM,
    loader,
    device: torch.device,
) -> dict[str, float]:
    totals = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0}
    num_batches = 0

    for batch in loader:
        metrics = validation_step(model=model, batch=batch, device=device)
        for key in totals:
            totals[key] += metrics[key]
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation loader is empty; cannot compute epoch metrics.")

    for key in totals:
        totals[key] /= num_batches
    return totals


def clean_model_state_dict(model: ModelKLDM) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("_cached_sampling_score_network")
    }


def export_model_checkpoint(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": clean_model_state_dict(model),
            "ema_model_state_dict": None if ema is None else ema.ema_model_state_dict(model),
            "ema_state_dict": None if ema is None else ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_path,
    )


def export_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_training_metrics(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train total loss")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="Val total loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Algorithm 2 total loss")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=True)

    axes[0].plot(epochs, [row["valid_percentage"] for row in rows], marker="o", label="Valid %")
    axes[0].plot(epochs, [row["match_rate_percentage"] for row in rows], marker="o", label="Match rate %")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Sample validity and match rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["mean_rmse"] for row in rows], marker="o", label="Mean RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Sample reconstruction mean RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        [row["oracle_lattice_valid_percentage"] for row in rows],
        marker="o",
        label="Oracle lattice valid %",
    )
    axes[2].plot(
        epochs,
        [row["oracle_lattice_match_rate_percentage"] for row in rows],
        marker="o",
        label="Oracle lattice match rate %",
    )
    axes[2].plot(
        epochs,
        [row["oracle_coordinate_valid_percentage"] for row in rows],
        marker="o",
        label="Oracle coordinate valid %",
    )
    axes[2].plot(
        epochs,
        [row["oracle_coordinate_match_rate_percentage"] for row in rows],
        marker="o",
        label="Oracle coordinate match rate %",
    )
    axes[2].set_ylabel("Percent")
    axes[2].set_title("Oracle validity and match rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(
        epochs,
        [row["oracle_lattice_mean_rmse"] for row in rows],
        marker="o",
        label="Oracle lattice mean RMSE",
    )
    axes[3].plot(
        epochs,
        [row["oracle_coordinate_mean_rmse"] for row in rows],
        marker="o",
        label="Oracle coordinate mean RMSE",
    )
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("RMSE")
    axes[3].set_title("Oracle mean RMSE")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_sampling_evaluation(
    model: ModelKLDM,
    loader,
    device: torch.device,
    num_samples: int,
    n_steps: int,
) -> dict[str, float | int]:
    reconstruction_results = []
    oracle_lattice_results = []
    oracle_coordinate_results = []
    template_iter = itertools.cycle(loader)

    model.eval()

    for _ in range(num_samples):
        batch = next(template_iter).to(device)

        with torch.no_grad():
            pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
                n_steps=n_steps,
                batch=batch,
            )

        result = evaluate_csp_reconstruction(
            pred_f=pos_t,
            pred_l=l_t[0],
            pred_a=h_t,
            target_f=batch.pos,
            target_l=batch.l[0],
            target_a=batch.h,
        )
        oracle_lattice_result = evaluate_csp_reconstruction(
            pred_f=pos_t,
            pred_l=batch.l[0],
            pred_a=h_t,
            target_f=batch.pos,
            target_l=batch.l[0],
            target_a=batch.h,
        )
        oracle_coordinate_result = evaluate_csp_reconstruction(
            pred_f=batch.pos,
            pred_l=l_t[0],
            pred_a=h_t,
            target_f=batch.pos,
            target_l=batch.l[0],
            target_a=batch.h,
        )

        reconstruction_results.append(result)
        oracle_lattice_results.append(oracle_lattice_result)
        oracle_coordinate_results.append(oracle_coordinate_result)

    summary = aggregate_csp_reconstruction_metrics(reconstruction_results)
    oracle_lattice_summary = aggregate_csp_reconstruction_metrics(oracle_lattice_results)
    oracle_coordinate_summary = aggregate_csp_reconstruction_metrics(oracle_coordinate_results)

    def count_valid(results) -> int:
        return sum(1 for r in results if r.valid)

    def count_match(results) -> int:
        return sum(1 for r in results if r.match)

    def percentage(count: int) -> float:
        return 100.0 * float(count) / float(num_samples)

    def nan_if_none(x: float | None) -> float:
        return float("nan") if x is None else float(x)

    valid_true_count = count_valid(reconstruction_results)
    match_true_count = count_match(reconstruction_results)
    oracle_lattice_valid_true_count = count_valid(oracle_lattice_results)
    oracle_lattice_match_true_count = count_match(oracle_lattice_results)
    oracle_coordinate_valid_true_count = count_valid(oracle_coordinate_results)
    oracle_coordinate_match_true_count = count_match(oracle_coordinate_results)

    return {
        "num_samples": num_samples,
        "valid_true_count": valid_true_count,
        "valid_percentage": percentage(valid_true_count),
        "match_true_count": match_true_count,
        "match_rate_percentage": percentage(match_true_count),
        "mean_rmse": nan_if_none(summary["rmse"]),
        "oracle_lattice_valid_true_count": oracle_lattice_valid_true_count,
        "oracle_lattice_valid_percentage": percentage(oracle_lattice_valid_true_count),
        "oracle_lattice_match_true_count": oracle_lattice_match_true_count,
        "oracle_lattice_match_rate_percentage": percentage(oracle_lattice_match_true_count),
        "oracle_lattice_mean_rmse": nan_if_none(oracle_lattice_summary["rmse"]),
        "oracle_coordinate_valid_true_count": oracle_coordinate_valid_true_count,
        "oracle_coordinate_valid_percentage": percentage(oracle_coordinate_valid_true_count),
        "oracle_coordinate_match_true_count": oracle_coordinate_match_true_count,
        "oracle_coordinate_match_rate_percentage": percentage(oracle_coordinate_match_true_count),
        "oracle_coordinate_mean_rmse": nan_if_none(oracle_coordinate_summary["rmse"]),
    }


def maybe_resume(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    if checkpoint_path is None:
        return 0, {}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cleaned_state_dict = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("_cached_sampling_score_network")
    }
    model.load_state_dict(cleaned_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
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
        default=50,
        help="Record and plot train/val losses every N epochs.",
    )
    parser.add_argument(
        "--sampling-samples",
        type=int,
        default=25,
        help="Number of CSP samples to generate at each validation epoch.",
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
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
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
        help="Number of optimizer steps before EMA switches from copying to decayed updates.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()
    print(f"starting trainHPC root={root} device={device.type}", flush=True)

    config = {
        "task": "CSP",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
        "validate_every": args.validate_every,
        "loss_every": args.loss_every,
        "sampling_samples": args.sampling_samples,
        "sampling_steps": args.sampling_steps,
        "load_from_checkpoint": args.load_from_checkpoint,
        "max_epochs": args.max_epochs,
        "ema_decay": args.ema_decay,
        "ema_start": args.ema_start,
    }

    train_loader = CSPTask().dataloader(
        root=root,
        split="train",
        batch_size=config["batch_size"],
        shuffle=True,
        download=True,
    )
    #Make sure to use the same validation subset over the validation epochs.
    #TODO:

    val_loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=config["batch_size"],
        shuffle=False,
        download=True,
    )

    sample_loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    print("constructed train/val/sample loaders", flush=True)

    model = ModelKLDM(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(
        model=model,
        decay=config["ema_decay"],
        start_step=config["ema_start"],
    )
    lambda_table = getattr(model.tdm, "_lambda_v_table", None)
    if isinstance(lambda_table, torch.Tensor):
        print(
            "lambda_v_table_stats "
            f"min={float(lambda_table.min()):.6f} "
            f"mean={float(lambda_table.mean()):.6f} "
            f"max={float(lambda_table.max()):.6f}",
            flush=True,
        )
    print("constructed model and optimizer", flush=True)

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
    wandb.define_metric("sampling/*", step_metric="epoch")
    wandb.define_metric("oracle_lattice/*", step_metric="epoch")
    wandb.define_metric("oracle_coordinate/*", step_metric="epoch")

    artifacts_dir = Path("artifacts") / "HPC"
    checkpoints_dir = artifacts_dir / "checkpoints"
    history_path = artifacts_dir / "training_history.csv"
    sampling_history_path = artifacts_dir / "sampling_history.csv"
    training_plot_path = artifacts_dir / "training_metrics.png"
    sampling_plot_path = artifacts_dir / "sampling_metrics.png"

    history: list[dict[str, Any]] = []
    sampling_history: list[dict[str, Any]] = []
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
            )

            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"(v={train_metrics['loss_v']:.6f}, l={train_metrics['loss_l']:.6f})"
            )

            should_record_loss = (epoch % config["loss_every"] == 0)
            should_run_sampling = (epoch % config["validate_every"] == 0)

            if not should_record_loss and not should_run_sampling:
                continue

            with ema.average_parameters(model):
                val_metrics = evaluate_loss(model=model, loader=val_loader, device=device)

            #Add metric here, ground truth vs predicted.  ASE libary.
            if should_record_loss:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_loss_v": train_metrics["loss_v"],
                        "train_loss_l": train_metrics["loss_l"],
                        "val_loss": val_metrics["loss"],
                        "val_loss_v": val_metrics["loss_v"],
                        "val_loss_l": val_metrics["loss_l"],
                    }
                )

                print(
                    f"validation_epoch={epoch:04d} "
                    f"val_loss={val_metrics['loss']:.6f} "
                    f"(v={val_metrics['loss_v']:.6f}, l={val_metrics['loss_l']:.6f})"
                )

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_metrics["loss"],
                        "train/epoch_loss_v": train_metrics["loss_v"],
                        "train/epoch_loss_l": train_metrics["loss_l"],
                        "val/epoch_loss": val_metrics["loss"],
                        "val/epoch_loss_v": val_metrics["loss_v"],
                        "val/epoch_loss_l": val_metrics["loss_l"],
                    }
                )

                export_csv(
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
                    ],
                )
                maybe_plot_training_metrics(history, training_plot_path)

            if not should_run_sampling:
                continue

            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
            export_model_checkpoint(
                model=model,
                optimizer=optimizer,
                ema=ema,
                output_path=checkpoint_path,
                config=config,
                epoch=epoch,
            )

            with ema.average_parameters(model):
                sampling_metrics = run_sampling_evaluation(
                    model=model,
                    loader=sample_loader,
                    device=device,
                    num_samples=config["sampling_samples"],
                    n_steps=config["sampling_steps"],
                )
            sampling_history.append({"epoch": epoch, **sampling_metrics})

            print(
                f"checkpoint_epoch={epoch:04d} "
                f"valid={sampling_metrics['valid_percentage']:.2f}% "
                f"match_rate={sampling_metrics['match_rate_percentage']:.2f}% "
                f"mean_rmse={sampling_metrics['mean_rmse']:.6f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "sampling/valid_true_count": sampling_metrics["valid_true_count"],
                    "sampling/valid_percentage": sampling_metrics["valid_percentage"],
                    "sampling/match_true_count": sampling_metrics["match_true_count"],
                    "sampling/match_rate_percentage": sampling_metrics["match_rate_percentage"],
                    "sampling/mean_rmse": sampling_metrics["mean_rmse"],
                    "oracle_lattice/valid_true_count": sampling_metrics["oracle_lattice_valid_true_count"],
                    "oracle_lattice/valid_percentage": sampling_metrics["oracle_lattice_valid_percentage"],
                    "oracle_lattice/match_true_count": sampling_metrics["oracle_lattice_match_true_count"],
                    "oracle_lattice/match_rate_percentage": sampling_metrics["oracle_lattice_match_rate_percentage"],
                    "oracle_lattice/mean_rmse": sampling_metrics["oracle_lattice_mean_rmse"],
                    "oracle_coordinate/valid_true_count": sampling_metrics["oracle_coordinate_valid_true_count"],
                    "oracle_coordinate/valid_percentage": sampling_metrics["oracle_coordinate_valid_percentage"],
                    "oracle_coordinate/match_true_count": sampling_metrics["oracle_coordinate_match_true_count"],
                    "oracle_coordinate/match_rate_percentage": sampling_metrics["oracle_coordinate_match_rate_percentage"],
                    "oracle_coordinate/mean_rmse": sampling_metrics["oracle_coordinate_mean_rmse"],
                }
            )

            export_csv(
                rows=sampling_history,
                output_path=sampling_history_path,
                fieldnames=[
                    "epoch",
                    "num_samples",
                    "valid_true_count",
                    "valid_percentage",
                    "match_true_count",
                    "match_rate_percentage",
                    "mean_rmse",
                    "oracle_lattice_valid_true_count",
                    "oracle_lattice_valid_percentage",
                    "oracle_lattice_match_true_count",
                    "oracle_lattice_match_rate_percentage",
                    "oracle_lattice_mean_rmse",
                    "oracle_coordinate_valid_true_count",
                    "oracle_coordinate_valid_percentage",
                    "oracle_coordinate_match_true_count",
                    "oracle_coordinate_match_rate_percentage",
                    "oracle_coordinate_mean_rmse",
                ],
            )

            maybe_plot_sampling_metrics(sampling_history, sampling_plot_path)

            checkpoint_artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
            checkpoint_artifact.add_file(str(checkpoint_path))
            run.log_artifact(checkpoint_artifact)

            if sampling_history_path.exists():
                wandb.save(str(sampling_history_path))
            if sampling_plot_path.exists():
                wandb.save(str(sampling_plot_path))

    finally:
        final_checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch}_final.pt"
        export_model_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            output_path=final_checkpoint_path,
            config=config | {"completed_epochs": epoch},
            epoch=epoch,
        )

        final_artifact = wandb.Artifact("csp_hpc_final", type="model")
        if final_checkpoint_path.exists():
            final_artifact.add_file(str(final_checkpoint_path))
        if history_path.exists():
            final_artifact.add_file(str(history_path))
        if sampling_history_path.exists():
            final_artifact.add_file(str(sampling_history_path))
        if training_plot_path.exists():
            final_artifact.add_file(str(training_plot_path))
        if sampling_plot_path.exists():
            final_artifact.add_file(str(sampling_plot_path))
        run.log_artifact(final_artifact)
        wandb.finish()


if __name__ == "__main__":
    train()
