from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kldm.data import CSPTask, resolve_data_root
from kldm.distribution.uniform import sample_uniform
from kldm.kldm import ModelKLDM

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/train.py") from exc

def validation_step(
    model: ModelKLDM,
    batch,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batch = batch.to(device)

    t_graph = sample_uniform(lb=model.diffusion_l.eps, size=(batch.num_graphs, 1), device=device)

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
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        t_graph = sample_uniform(lb=model.diffusion_l.eps, size=(batch.num_graphs, 1), device=device)

        optimizer.zero_grad()
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=1.0,
        )
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += float(metrics[key])
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training loader is empty; cannot compute epoch metrics.")

    for key in running:
        running[key] /= num_batches
    return running


def evaluate(
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


def export_history(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_v",
        "train_loss_l",
        "val_loss",
        "val_loss_v",
        "val_loss_l",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def export_final_model(
    model: ModelKLDM,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KLDM on the CSP task.")
    parser.add_argument(
        "--epoch",
        "--epochs",
        dest="epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()

    config = {
        "task": "CSP",
        "epochs": args.epochs,
        "batch_size": 256,
        "lr": 1e-3,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
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

    model = ModelKLDM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    run = wandb.init(
        project="kldm-csp",
        config=config,
        name=f"csp_{config['epochs']}_epochs",
    )

    output_path = Path("artifacts") / "csp_final_model.pt"
    history_path = Path("artifacts") / "csp_training_history.csv"
    best_output_path = Path("artifacts") / "csp_best_model.pt"
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch_idx in range(config["epochs"]):
        epoch = epoch_idx + 1
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(model=model, loader=val_loader, device=device)

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_loss_v": train_metrics["loss_v"],
            "train_loss_l": train_metrics["loss_l"],
            "val_loss": val_metrics["loss"],
            "val_loss_v": val_metrics["loss_v"],
            "val_loss_l": val_metrics["loss_l"],
        }
        history.append(epoch_record)

        log_metrics = {
            "epoch": epoch,
            "train/epoch_loss": train_metrics["loss"],
            "train/epoch_loss_v": train_metrics["loss_v"],
            "train/epoch_loss_l": train_metrics["loss_l"],
            "val/epoch_loss": val_metrics["loss"],
            "val/epoch_loss_v": val_metrics["loss_v"],
            "val/epoch_loss_l": val_metrics["loss_l"],
            "train/loss": train_metrics["loss"],
            "train/loss_v": train_metrics["loss_v"],
            "train/loss_l": train_metrics["loss_l"],
            "val/loss": val_metrics["loss"],
            "val/loss_v": val_metrics["loss_v"],
            "val/loss_l": val_metrics["loss_l"],
        }
        wandb.log(log_metrics)

        print(
            f"epoch={epoch:03d}/{config['epochs']:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(v={train_metrics['loss_v']:.4f}, l={train_metrics['loss_l']:.4f}) "
            f"val_loss={val_metrics['loss']:.6f} "
            f"(v={val_metrics['loss_v']:.4f}, l={val_metrics['loss_l']:.4f}) "
            f"device={device.type}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            export_final_model(
                model=model,
                optimizer=optimizer,
                output_path=best_output_path,
                config=config | {"best_val_loss": best_val_loss, "best_epoch": epoch},
            )

    export_final_model(
        model=model,
        optimizer=optimizer,
        output_path=output_path,
        config=config,
    )
    export_history(history=history, output_path=history_path)
    artifact = wandb.Artifact("csp_final_model", type="model")
    artifact.add_file(str(output_path))
    if best_output_path.exists():
        artifact.add_file(str(best_output_path))
    if history_path.exists():
        artifact.add_file(str(history_path))
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
