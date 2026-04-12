from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

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
    clean_state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("_cached_sampling_score_network")
    }
    torch.save(
        {
            "model_state_dict": clean_state_dict,
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
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Optional batch size override. Defaults to 256 on GPU and 16 on CPU.",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=0,
        help="DataLoader worker count. Defaults to 0 for safer local execution.",
    )
    parser.add_argument(
        "--train-fraction",
        dest="train_fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use, e.g. 0.2 for 20%%.",
    )
    parser.add_argument(
        "--subset-seed",
        dest="subset_seed",
        type=int,
        default=7,
        help="Random seed used when selecting a training subset.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--lambda-precompute-batches",
        dest="lambda_precompute_batches",
        type=int,
        default=None,
        help="Number of train batches used to precompute the centered lambda(t) table.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()
    batch_size = args.batch_size if args.batch_size is not None else (256 if device.type == "cuda" else 16)
    lambda_precompute_batches = (
        args.lambda_precompute_batches
        if args.lambda_precompute_batches is not None
        else (64 if device.type == "cuda" else 16)
    )
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("--train-fraction must be in the interval (0, 1].")

    config = {
        "task": "CSP",
        "epochs": args.epochs,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "train_fraction": args.train_fraction,
        "subset_seed": args.subset_seed,
        "wandb_enabled": not args.no_wandb,
        "lambda_precompute_batches": lambda_precompute_batches,
        "lr": 1e-3,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
    }

    csp_task = CSPTask()
    base_train_dataset = csp_task.fit_dataset(
        root=root,
        split="train",
        download=True,
    )
    train_dataset = base_train_dataset
    if config["train_fraction"] < 1.0:
        subset_size = max(1, int(len(train_dataset) * config["train_fraction"]))
        subset_generator = torch.Generator().manual_seed(config["subset_seed"])
        subset_indices = torch.randperm(len(train_dataset), generator=subset_generator)[:subset_size].tolist()
        train_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=base_train_dataset.collate_fn,
    )
    val_loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        download=True,
    )

    model = ModelKLDM(device=device).to(device)
    print(
        f"precomputing centered lambda_v table from {config['lambda_precompute_batches']} train batches",
        flush=True,
    )
    model.tdm.precompute_lambda_v_table_from_loader(
        loader=train_loader,
        device=device,
        num_batches=config["lambda_precompute_batches"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    run = None
    if config["wandb_enabled"]:
        run = wandb.init(
            project="kldm-csp",
            config=config,
            name=f"csp_{config['epochs']}_epochs",
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    output_path = Path("artifacts") / "csp_final_model.pt"
    history_path = Path("artifacts") / "csp_training_history.csv"
    best_output_path = Path("artifacts") / "csp_best_model.pt"
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    print(
        f"starting training epochs={config['epochs']} batch_size={config['batch_size']} "
        f"lr={config['lr']} device={device.type} train_fraction={config['train_fraction']}",
        flush=True,
    )

    try:
        for epoch_idx in range(config["epochs"]):
            epoch = epoch_idx + 1
            print(f"starting epoch {epoch}/{config['epochs']}", flush=True)

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
            if config["wandb_enabled"]:
                wandb.log(log_metrics)

            print(
                f"epoch={epoch:03d}/{config['epochs']:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"(v={train_metrics['loss_v']:.4f}, l={train_metrics['loss_l']:.4f}) "
                f"val_loss={val_metrics['loss']:.6f} "
                f"(v={val_metrics['loss_v']:.4f}, l={val_metrics['loss_l']:.4f}) "
                f"device={device.type}",
                flush=True,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                export_final_model(
                    model=model,
                    optimizer=optimizer,
                    output_path=best_output_path,
                    config=config | {"best_val_loss": best_val_loss, "best_epoch": epoch},
                )
    finally:
        export_final_model(
            model=model,
            optimizer=optimizer,
            output_path=output_path,
            config=config,
        )
        export_history(history=history, output_path=history_path)
        if config["wandb_enabled"] and run is not None:
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
