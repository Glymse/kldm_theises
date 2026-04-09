from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys
from typing import Any

# uv run src/kldm/trainHPC.py --epoch 500
# uv run src/kldm/trainHPC.py --epoch 800 --resume-checkpoint artifacts/checkpoints/train_checkpoint_epoch_500.pt
# --validate-every 100
# --sampling-samples 50
# --sampling-steps 1000
# --batch-size 256
# --lr 1e-3

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
            lambda_t_fn=None,
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
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0}
    num_batches = 0

    for batch in loader:
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
            lambda_t_fn=None,
        )
        loss.backward()
        optimizer.step()

        running["loss"] += float(loss)
        running["loss_v"] += float(metrics["loss_v"])
        running["loss_l"] += float(metrics["loss_l"])
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training loader is empty; cannot compute epoch metrics.")

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


def export_model_checkpoint(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    config: dict[str, Any],
    epoch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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


def maybe_plot_sampling_metrics(rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    if plt is None or not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(epochs, [row["sampling_valid"] for row in rows], marker="o", label="Validity")
    axes[0].plot(epochs, [row["sampling_match_rate"] for row in rows], marker="o", label="MR")
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rmse_values = [row["sampling_rmse"] for row in rows]
    axes[1].plot(epochs, rmse_values, marker="o", color="tab:red")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        epochs,
        [row["oracle_lattice_match_rate"] for row in rows],
        marker="o",
        label="Oracle Lattice MR",
    )
    axes[2].plot(
        epochs,
        [row["oracle_coordinate_match_rate"] for row in rows],
        marker="o",
        label="Oracle Coord MR",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Oracle MR")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_sampling_evaluation(
    model: ModelKLDM,
    loader,
    checkpoint_path: Path,
    device: torch.device,
    num_samples: int,
    n_steps: int,
) -> dict[str, float | int | None]:
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
                checkpoint_path=str(checkpoint_path),
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

    return {
        "num_samples": int(summary["num_samples"]),
        "sampling_valid": summary["valid"],
        "sampling_match_rate": summary["match_rate"],
        "sampling_rmse": summary["rmse"],
        "oracle_lattice_valid": oracle_lattice_summary["valid"],
        "oracle_lattice_match_rate": oracle_lattice_summary["match_rate"],
        "oracle_lattice_rmse": oracle_lattice_summary["rmse"],
        "oracle_coordinate_valid": oracle_coordinate_summary["valid"],
        "oracle_coordinate_match_rate": oracle_coordinate_summary["match_rate"],
        "oracle_coordinate_rmse": oracle_coordinate_summary["rmse"],
    }


def maybe_resume(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    if checkpoint_path is None:
        return 0, {}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    checkpoint_config = checkpoint.get("config", {})

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {start_epoch}")

    return start_epoch, checkpoint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KLDM on the CSP task with validation and sampling every N epochs."
    )
    parser.add_argument(
        "--epoch",
        "--epochs",
        dest="epochs",
        type=int,
        default=100,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=100,
        help="Run loss validation + sampling validation every N epochs.",
    )
    parser.add_argument(
        "--sampling-samples",
        type=int,
        default=50,
        help="Number of CSP samples to generate during each checkpoint validation run.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=1000,
        help="Number of Algorithm 3 sampling steps for checkpoint validation runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()

    config = {
        "task": "CSP",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
        "validate_every": args.validate_every,
        "sampling_samples": args.sampling_samples,
        "sampling_steps": args.sampling_steps,
        "resume_checkpoint": args.resume_checkpoint,
    }

    train_loader = CSPTask().dataloader(
        root=root,
        split="train",
        batch_size=config["batch_size"],
        shuffle=True,
        download=True,
    )
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

    model = ModelKLDM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start_epoch, resume_config = maybe_resume(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume_checkpoint,
        device=device,
    )

    if start_epoch >= config["epochs"]:
        raise ValueError(
            f"Resumed checkpoint is already at epoch {start_epoch}, "
            f"but requested total epochs is only {config['epochs']}."
        )

    run_name = (
        f"csp_hpc_resume_from_{start_epoch}_to_{config['epochs']}"
        if args.resume_checkpoint is not None
        else f"csp_hpc_{config['epochs']}_epochs"
    )

    run = wandb.init(
        project="kldm-csp-hpc",
        config=config | {"resume_config": resume_config, "start_epoch": start_epoch},
        name=run_name,
    )

    artifacts_dir = Path("artifacts")
    checkpoints_dir = artifacts_dir / "checkpoints"
    final_model_path = artifacts_dir / "csp_final_model.pt"
    best_model_path = artifacts_dir / "csp_best_model.pt"
    history_path = artifacts_dir / "csp_hpc_training_history.csv"
    sampling_history_path = artifacts_dir / "csp_hpc_sampling_history.csv"
    sampling_plot_path = artifacts_dir / "csp_hpc_sampling_metrics.png"

    history: list[dict[str, Any]] = []
    sampling_history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    if best_model_path.exists() and args.resume_checkpoint is None:
        best_model_path.unlink()

    for epoch_idx in range(start_epoch, config["epochs"]):
        epoch = epoch_idx + 1

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        should_validate = (epoch % config["validate_every"] == 0) or (epoch == config["epochs"])

        log_data = {
            "epoch": epoch,
            "train/epoch_loss": train_metrics["loss"],
            "train/epoch_loss_v": train_metrics["loss_v"],
            "train/epoch_loss_l": train_metrics["loss_l"],
        }

        print(
            f"epoch={epoch:04d}/{config['epochs']:04d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(v={train_metrics['loss_v']:.6f}, l={train_metrics['loss_l']:.6f})"
        )

        if should_validate:
            val_metrics = evaluate_loss(
                model=model,
                loader=val_loader,
                device=device,
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_loss_v": train_metrics["loss_v"],
                "train_loss_l": train_metrics["loss_l"],
                "val_loss": val_metrics["loss"],
                "val_loss_v": val_metrics["loss_v"],
                "val_loss_l": val_metrics["loss_l"],
            }
            history.append(epoch_record)

            log_data.update(
                {
                    "val/epoch_loss": val_metrics["loss"],
                    "val/epoch_loss_v": val_metrics["loss_v"],
                    "val/epoch_loss_l": val_metrics["loss_l"],
                }
            )

            print(
                f"validation_epoch={epoch:04d} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"(v={val_metrics['loss_v']:.6f}, l={val_metrics['loss_l']:.6f})"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                export_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    output_path=best_model_path,
                    config=config | {"best_val_loss": best_val_loss, "best_epoch": epoch},
                    epoch=epoch,
                )

            checkpoint_path = checkpoints_dir / f"train_checkpoint_epoch_{epoch}.pt"
            export_model_checkpoint(
                model=model,
                optimizer=optimizer,
                output_path=checkpoint_path,
                config=config,
                epoch=epoch,
            )

            checkpoint_artifact = wandb.Artifact(
                f"train_checkpoint_epoch_{epoch}",
                type="model",
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            run.log_artifact(checkpoint_artifact)

            sampling_metrics = run_sampling_evaluation(
                model=model,
                loader=sample_loader,
                checkpoint_path=checkpoint_path,
                device=device,
                num_samples=config["sampling_samples"],
                n_steps=config["sampling_steps"],
            )

            sampling_record = {"epoch": epoch, **sampling_metrics}
            sampling_history.append(sampling_record)

            log_data.update(
                {
                    "sampling/valid": sampling_metrics["sampling_valid"],
                    "sampling/match_rate": sampling_metrics["sampling_match_rate"],
                    "sampling/rmse": sampling_metrics["sampling_rmse"],
                    "oracle_lattice/valid": sampling_metrics["oracle_lattice_valid"],
                    "oracle_lattice/match_rate": sampling_metrics["oracle_lattice_match_rate"],
                    "oracle_lattice/rmse": sampling_metrics["oracle_lattice_rmse"],
                    "oracle_coordinate/valid": sampling_metrics["oracle_coordinate_valid"],
                    "oracle_coordinate/match_rate": sampling_metrics["oracle_coordinate_match_rate"],
                    "oracle_coordinate/rmse": sampling_metrics["oracle_coordinate_rmse"],
                }
            )

            print(
                f"checkpoint_epoch={epoch:04d} "
                f"sample_valid={sampling_metrics['sampling_valid']!s} "
                f"sample_mr={sampling_metrics['sampling_match_rate']!s} "
                f"sample_rmse={sampling_metrics['sampling_rmse']!s}"
            )

        wandb.log(log_data)

    export_model_checkpoint(
        model=model,
        optimizer=optimizer,
        output_path=final_model_path,
        config=config | {"completed_epochs": config["epochs"]},
        epoch=config["epochs"],
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

    export_csv(
        rows=sampling_history,
        output_path=sampling_history_path,
        fieldnames=[
            "epoch",
            "num_samples",
            "sampling_valid",
            "sampling_match_rate",
            "sampling_rmse",
            "oracle_lattice_valid",
            "oracle_lattice_match_rate",
            "oracle_lattice_rmse",
            "oracle_coordinate_valid",
            "oracle_coordinate_match_rate",
            "oracle_coordinate_rmse",
        ],
    )

    plot_path = maybe_plot_sampling_metrics(sampling_history, sampling_plot_path)

    final_artifact = wandb.Artifact("csp_final_model_hpc", type="model")
    final_artifact.add_file(str(final_model_path))

    if best_model_path.exists():
        final_artifact.add_file(str(best_model_path))
    if history_path.exists():
        final_artifact.add_file(str(history_path))
    if sampling_history_path.exists():
        final_artifact.add_file(str(sampling_history_path))
    if plot_path is not None and plot_path.exists():
        final_artifact.add_file(str(plot_path))

    run.log_artifact(final_artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
