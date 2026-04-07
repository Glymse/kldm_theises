from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from kldm.data import DNGTask, resolve_data_root
from kldm.distribution.uniform import sample_uniform
from kldm.kldm import ModelKLDM

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/train.py") from exc


def tdm_paper_lambda(model: ModelKLDM, t_node: torch.Tensor) -> torch.Tensor:
    """Velocity weighting used by the current KLDM training path."""
    return torch.full_like(t_node, model.tdm.time_scaling_T ** 2)


def reconstruct_clean_atom_logits(
    model: ModelKLDM,
    a_t: torch.Tensor,
    pred_eps_a: torch.Tensor,
    t_node: torch.Tensor,
) -> torch.Tensor:
    """Estimate clean atom logits from the noisy atom channel."""
    alpha_t = model.diffusion_a._match_dims(model.diffusion_a.alpha(t_node), a_t)
    sigma_t = model.diffusion_a._match_dims(model.diffusion_a.sigma(t_node), a_t)
    return (a_t - sigma_t * pred_eps_a) / alpha_t


def validation_step(
    model: ModelKLDM,
    batch,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batch = batch.to(device)

    t_graph = sample_uniform(lb=model.diffusion_l.eps, size=(batch.num_graphs, 1), device=device)
    t_node = t_graph[batch.batch].squeeze(-1)

    with torch.no_grad():
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=1.0,
            lambda_a=1.0,
            lambda_t_fn=lambda x: tdm_paper_lambda(model, x),
        )

        (v_t, f_t, l_t, a_t), _ = model.algorithm1_training_targets(batch, t_graph)
        preds = model.score_network(
            t=t_graph,
            pos=f_t,
            v=v_t,
            h=a_t,
            l=l_t,
            node_index=batch.batch,
            edge_node_index=batch.edge_node_index,
        )

        clean_atom_logits = reconstruct_clean_atom_logits(
            model=model,
            a_t=a_t,
            pred_eps_a=preds["h"],
            t_node=t_node,
        )
        pred_labels = clean_atom_logits.argmax(dim=-1)
        true_labels = batch.h.argmax(dim=-1)
        val_accuracy = (pred_labels == true_labels).float().mean()

    return {
        "loss": float(loss),
        "loss_v": float(metrics["loss_v"]),
        "loss_l": float(metrics["loss_l"]),
        "loss_a": float(metrics["loss_a"]),
        "val_accuracy": float(val_accuracy),
    }


def train_epoch(
    model: ModelKLDM,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0, "loss_a": 0.0}

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        t_graph = sample_uniform(lb=model.diffusion_l.eps, size=(batch.num_graphs, 1), device=device)

        optimizer.zero_grad()
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=1.0,
            lambda_a=1.0,
            lambda_t_fn=lambda x: tdm_paper_lambda(model, x),
        )
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += float(metrics[key])

    num_steps = step + 1
    for key in running:
        running[key] /= num_steps
    return running


def evaluate(
    model: ModelKLDM,
    loader,
    device: torch.device,
) -> dict[str, float]:
    totals = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0, "loss_a": 0.0, "val_accuracy": 0.0}

    for step, batch in enumerate(loader):
        metrics = validation_step(model=model, batch=batch, device=device)
        for key in totals:
            totals[key] += metrics[key]

    num_steps = step + 1
    for key in totals:
        totals[key] /= num_steps
    return totals


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


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()

    config = {
        "task": "DNG",
        "epochs": 500,
        "batch_size": 64,
        "lr": 1e-3,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
        "lambda_a": 1.0,
    }

    train_loader = DNGTask().dataloader(
        root=root,
        split="train",
        batch_size=config["batch_size"],
        shuffle=True,
        download=True,
    )
    val_loader = DNGTask().dataloader(
        root=root,
        split="val",
        batch_size=config["batch_size"],
        shuffle=False,
        download=True,
    )

    model = ModelKLDM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    run = wandb.init(
        project="kldm-dng",
        config=config,
        name="dng_500_epochs",
    )

    output_path = Path("artifacts") / "dng_final_model.pt"

    for epoch in range(config["epochs"]):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_metrics = evaluate(model=model, loader=val_loader, device=device)

        log_metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/loss_v": train_metrics["loss_v"],
            "train/loss_l": train_metrics["loss_l"],
            "train/loss_a": train_metrics["loss_a"],
            "val/loss": val_metrics["loss"],
            "val/loss_v": val_metrics["loss_v"],
            "val/loss_l": val_metrics["loss_l"],
            "val/loss_a": val_metrics["loss_a"],
            "val/accuracy": val_metrics["val_accuracy"],
        }
        wandb.log(log_metrics)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(v={train_metrics['loss_v']:.4f}, l={train_metrics['loss_l']:.4f}, a={train_metrics['loss_a']:.4f}) "
            f"val_loss={val_metrics['loss']:.6f} "
            f"(v={val_metrics['loss_v']:.4f}, l={val_metrics['loss_l']:.4f}, a={val_metrics['loss_a']:.4f}) "
            f"val_acc={val_metrics['val_accuracy']:.4f} "
            f"device={device.type}"
        )

    export_final_model(
        model=model,
        optimizer=optimizer,
        output_path=output_path,
        config=config,
    )
    artifact = wandb.Artifact("dng_final_model", type="model")
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
