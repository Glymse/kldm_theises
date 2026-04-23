from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from kldm.data import CSPTask, resolve_data_root
# from kldm.diffusionModels.trivialized_diffusion import TrivialisedDiffusion
from kldm.diffusionModels.TDMdev import TrivialisedDiffusionDev
from kldm.distribution.uniform import sample_uniform
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    CSPMetrics,
    structures_from_batch,
    structures_from_tensors,
)

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/trainHPC.py") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import ase.io
    from ase.visualize.plot import plot_atoms
except ImportError:  # pragma: no cover
    ase = None
    plot_atoms = None

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:  # pragma: no cover
    AseAtomsAdaptor = None


STOP_REQUESTED = False
TIME_LOWER_BOUND = 1e-3
GLOBAL_SEED = 42
VAL_SUBSET_SEED = 42
LOADER_NUM_WORKERS = 1
LOADER_PIN_MEMORY = True

NODE_AVERAGED_METRICS = {
    "loss_v",
    "raw_loss_v",
    "target_v_abs_mean",
    "target_v_norm_mean",
    "pred_v_abs_mean",
    "pred_v_norm_mean",
    "lambda_v_mean",
    "lambda_v_effective",
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


def sample_graph_times(num_graphs: int, device: torch.device) -> torch.Tensor:
    return sample_uniform(
        lb=TIME_LOWER_BOUND,
        size=(num_graphs, 1),
        device=device,
    )


def make_fixed_subset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    # Match facit's current DataModule subset selection exactly.
    rnd = np.random.RandomState(seed=seed)
    indices = rnd.permutation(np.arange(subset_size)).tolist()
    return Subset(dataset, indices)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_epoch_metrics(
    batch_metrics: list[tuple[dict[str, float], float, float]],
    *,
    lambda_v: float,
    lambda_l: float,
) -> dict[str, float]:
    if not batch_metrics:
        raise RuntimeError("Cannot aggregate empty metric list.")

    totals_node: dict[str, float] = {}
    totals_graph: dict[str, float] = {}
    totals_batch: dict[str, float] = {}
    totals_other_graph: dict[str, float] = {}
    min_metrics: dict[str, float] = {}
    max_metrics: dict[str, float] = {}

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
            elif key == "lambda_v_min":
                min_metrics[key] = float(value) if key not in min_metrics else min(min_metrics[key], float(value))
            elif key == "lambda_v_max":
                max_metrics[key] = float(value) if key not in max_metrics else max(max_metrics[key], float(value))
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

    aggregated.update(min_metrics)
    aggregated.update(max_metrics)

    aggregated["loss_v"] = aggregated.get("loss_v", 0.0)
    aggregated["loss_l"] = aggregated.get("loss_l", 0.0)
    aggregated["loss"] = lambda_v * aggregated["loss_v"] + lambda_l * aggregated["loss_l"]
    return aggregated


def _metric_for_ranking(value: float) -> float:
    if value != value:  # NaN
        return float("-inf")
    return float(value)


def _make_atoms_grid(atoms_lst: list[Any]):
    if plt is None or plot_atoms is None or not atoms_lst:
        return None

    bs = len(atoms_lst)
    nrows = ncols = int(np.ceil(np.sqrt(bs)))
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    for i, ax in enumerate(fig.axes):
        if i >= len(atoms_lst):
            ax.axis("off")
            continue
        plot_atoms(atoms_lst[i], ax)
        ax.axis("off")
    return fig


def _structures_to_ase_atoms(structures: list[Any]) -> list[Any]:
    if AseAtomsAdaptor is None:
        return []

    adaptor = AseAtomsAdaptor()
    atoms_lst = []
    for structure in structures:
        if structure is None:
            continue
        try:
            atoms = adaptor.get_atoms(structure)
            atoms.wrap()
            atoms_lst.append(atoms)
        except Exception:
            continue
    return atoms_lst


def log_sampled_structures(
    *,
    epoch: int,
    generated_structures: list[Any],
    gt_structures: list[Any],
    output_dir: Path,
    run,
    num_log_wandb: int = 25,
) -> None:
    generated_atoms = _structures_to_ase_atoms(generated_structures)
    gt_atoms = _structures_to_ase_atoms(gt_structures)

    epoch_dir = output_dir / str(epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    if generated_atoms and ase is not None:
        ase.io.write(filename=str(epoch_dir / "samples.xyz"), images=generated_atoms, format="extxyz")

    if run is None or num_log_wandb <= 0:
        return

    idx = min(len(generated_atoms), num_log_wandb)
    if idx > 0:
        fig = _make_atoms_grid(generated_atoms[-idx:])
        if fig is not None:
            wandb.log({"epoch": epoch, "val/images_generated": wandb.Image(fig)})
            plt.close(fig)

    idx_gt = min(len(gt_atoms), num_log_wandb)
    if idx_gt > 0:
        fig = _make_atoms_grid(gt_atoms[-idx_gt:])
        if fig is not None:
            wandb.log({"epoch": epoch, "val/images_gt": wandb.Image(fig)})
            plt.close(fig)


def update_top_k_checkpoints(
    *,
    checkpoints_dir: Path,
    current_checkpoint_path: Path,
    epoch: int,
    metric_value: float,
    top_k_records: list[dict[str, Any]],
    save_top_k: int,
) -> list[dict[str, Any]]:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoints_dir / "last.pt"
    shutil.copy2(current_checkpoint_path, last_path)

    records = [record for record in top_k_records if int(record["epoch"]) != int(epoch)]
    records.append(
        {
            "epoch": int(epoch),
            "metric": float(metric_value),
            "path": str(current_checkpoint_path),
        }
    )
    records.sort(key=lambda item: (_metric_for_ranking(item["metric"]), item["epoch"]), reverse=True)
    kept = records[: max(0, int(save_top_k))]
    keep_epochs = {int(record["epoch"]) for record in kept}

    for record in records[max(0, int(save_top_k)) :]:
        path = Path(record["path"])
        if path.exists():
            path.unlink(missing_ok=True)

    summary_path = checkpoints_dir / "top_k.json"
    export_csv(
        rows=kept,
        output_path=checkpoints_dir / "top_k.csv",
        fieldnames=["epoch", "metric", "path"],
    )
    summary_path.write_text(
        json.dumps(
            [
                {
                    "epoch": int(record["epoch"]),
                    "metric": float(record["metric"]),
                    "path": str(record["path"]),
                }
                for record in kept
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    for path in checkpoints_dir.glob("epoch_*.pt"):
        match = re.match(r"^epoch_(\d+)\.pt$", path.name)
        if match is None:
            continue
        path_epoch = int(match.group(1))
        if path_epoch not in keep_epochs:
            path.unlink(missing_ok=True)

    return kept


def load_top_k_records(checkpoints_dir: Path) -> list[dict[str, Any]]:
    summary_path = checkpoints_dir / "top_k.json"
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for item in payload:
        try:
            records.append(
                {
                    "epoch": int(item["epoch"]),
                    "metric": float(item["metric"]),
                    "path": str(item["path"]),
                }
            )
        except Exception:
            continue
    return records


class ExponentialMovingAverage:
    def __init__(
        self,
        model: ModelKLDM,
        decay: float = 0.999,
        start_epoch: int = 500,
    ) -> None:
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.num_updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: ModelKLDM, current_epoch: int) -> None:
        if current_epoch <= self.start_epoch:
            return
        self.num_updates += 1
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "start_epoch": self.start_epoch,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.start_epoch = int(state_dict.get("start_epoch", state_dict.get("start_step", self.start_epoch)))
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
    lambda_l: float,
    debug: bool = False,
) -> dict[str, float]:
    model.eval()
    batch = batch.to(device)

    t_graph = sample_graph_times(num_graphs=batch.num_graphs, device=device)

    with torch.no_grad():
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=1.0,
            lambda_l=lambda_l,
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
    lambda_v: float,
    lambda_l: float,
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
        t_graph = sample_graph_times(num_graphs=batch.num_graphs, device=device)

        optimizer.zero_grad()
        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t_graph,
            lambda_v=lambda_v,
            lambda_l=lambda_l,
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
    return aggregate_epoch_metrics(
        batch_metrics,
        lambda_v=lambda_v,
        lambda_l=lambda_l,
    )


def evaluate_loss(
    model: ModelKLDM,
    loader,
    device: torch.device,
    lambda_v: float,
    lambda_l: float,
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
            lambda_l=lambda_l,
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
    return aggregate_epoch_metrics(
        batch_metrics,
        lambda_v=lambda_v,
        lambda_l=lambda_l,
    )


def clean_model_state_dict(model: ModelKLDM) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("_cached_sampling_model")
        and not key.endswith("._lambda_v_table")
    }


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

    axes[0].plot(epochs, [row["train_loss_weighted"] for row in rows], marker="o", label="Train loss_weighted")
    axes[0].plot(epochs, [row["val_loss_weighted"] for row in rows], marker="o", label="Val loss_weighted")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Weighted loss")
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

    axes[2].plot(epochs, [row["val_loss_weighted"] for row in rows], marker="o", label="val/loss_weighted")
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
    reconstruction_results = []
    num_graphs_seen = 0

    model.eval()

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
                n_steps=n_steps,
                batch=batch,
                checkpoint_path=None,
            )

        ptr = batch.ptr.tolist()
        for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            result = evaluate_csp_reconstruction(
                pred_f=pos_t[start:end],
                pred_l=l_t[graph_idx],
                pred_a=h_t[start:end],
                target_f=batch.pos[start:end],
                target_l=batch.l[graph_idx],
                target_a=batch.h[start:end],
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


def run_facit_style_validation_epoch(
    model: ModelKLDM,
    loader,
    device: torch.device,
    *,
    lambda_v: float,
    lambda_l: float,
    n_steps: int,
    ema: ExponentialMovingAverage | None = None,
    current_epoch: int,
    force_ema_sampling: bool,
    max_graphs: int | None = None,
    debug: bool = False,
) -> tuple[dict[str, float], dict[str, float | int], list[Any], list[Any]]:
    """
    Match facitKLDM's validation runner more closely:
    1. one validation pass only
    2. compute val loss and sampling on the same batches
    3. use current model for validation sampling by default
    """
    model.eval()
    batch_metrics: list[tuple[dict[str, float], float, float]] = []
    csp_metrics = CSPMetrics()
    num_graphs_seen = 0
    logged_pred_structures: list[Any] = []
    logged_target_structures: list[Any] = []

    use_ema_sampling = should_use_ema_for_sampling(
        ema=ema,
        current_epoch=current_epoch,
        force_ema=force_ema_sampling,
    )

    for batch in loader:
        batch = batch.to(device)
        num_graphs = float(batch.num_graphs)
        num_nodes = float(batch.pos.shape[0])
        t_graph = sample_graph_times(num_graphs=batch.num_graphs, device=device)

        with torch.no_grad():
            _, metrics = model.algorithm2_loss(
                batch=batch,
                t=t_graph,
                lambda_v=lambda_v,
                lambda_l=lambda_l,
                debug=debug,
            )

            if use_ema_sampling:
                assert ema is not None
                with ema.average_parameters(model):
                    pos_t, _v_t, l_t, h_t = model.sample_CSP_algorithm3(
                        n_steps=n_steps,
                        batch=batch,
                        checkpoint_path=None,
                    )
            else:
                pos_t, _v_t, l_t, h_t = model.sample_CSP_algorithm3(
                    n_steps=n_steps,
                    batch=batch,
                    checkpoint_path=None,
                )

        batch_metrics.append(
            ({key: float(value) for key, value in metrics.items()}, num_nodes, num_graphs)
        )

        pred_structures = structures_from_tensors(
            tensors={"h": h_t, "pos": pos_t, "l": l_t},
            ptr=batch.ptr,
        )
        target_structures = structures_from_batch(batch)

        if max_graphs is not None:
            remaining = max_graphs - num_graphs_seen
            if remaining <= 0:
                break
            pred_structures = pred_structures[:remaining]
            target_structures = target_structures[:remaining]

        csp_metrics.update(pred_structures, target_structures)
        logged_pred_structures.extend(pred_structures)
        logged_target_structures.extend(target_structures)
        num_graphs_seen += len(pred_structures)

        if max_graphs is not None and num_graphs_seen >= max_graphs:
            break

    if not batch_metrics:
        raise RuntimeError("Validation loader is empty; cannot compute epoch metrics.")

    val_metrics = aggregate_epoch_metrics(
        batch_metrics,
        lambda_v=lambda_v,
        lambda_l=lambda_l,
    )
    summary = csp_metrics.summarize()
    sampling_metrics = {
        "num_samples": int(num_graphs_seen),
        "valid": float("nan") if summary["valid"] is None else float(summary["valid"]),
        "match_rate": float("nan") if summary["match_rate"] is None else float(summary["match_rate"]),
        "rmse": float("nan") if summary["rmse"] is None else float(summary["rmse"]),
    }
    return val_metrics, sampling_metrics, logged_pred_structures, logged_target_structures

def should_use_ema_for_sampling(
    ema: ExponentialMovingAverage | None,
    *,
    current_epoch: int,
    force_ema: bool,
) -> bool:
    if ema is None:
        return False
    # Keep sampling on live model until EMA has received at least one update.
    if ema.num_updates <= 0:
        return False
    # Once EMA is active, match facit-style model selection.
    return bool(force_ema or current_epoch > ema.start_epoch)


def prune_old_checkpoints(
    checkpoints_dir: Path,
    keep_last: int = 2,
    current_epoch: int | None = None,
) -> None:
    epoch_pattern = re.compile(r"^checkpoint_epoch_(\d+)$")
    checkpoint_paths = []
    for path in checkpoints_dir.glob("checkpoint_epoch_*.pt"):
        match = epoch_pattern.match(path.stem)
        if match is None:
            continue
        epoch = int(match.group(1))
        if current_epoch is not None and epoch > current_epoch:
            continue
        checkpoint_paths.append((epoch, path))
    checkpoint_paths.sort(key=lambda item: item[0])
    while len(checkpoint_paths) > keep_last:
        checkpoint_paths[0][1].unlink(missing_ok=True)
        checkpoint_paths.pop(0)


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
        if not key.startswith("_cached_sampling_model")
        and not key.endswith("._lambda_v_table")
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
        help="Optional checkpoint path to continue from, e.g. artifacts/HPC/checkpoints/last.pt or epoch_100.pt",
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
            "Defaults to false to match facitKLDM validation semantics. "
            "Enable only for explicit ablations."
        ),
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Number of best validation checkpoints to keep, matching facit's ModelCheckpoint default.",
    )
    parser.add_argument(
        "--log-sampled-atoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save facit-style sampled structures to samples/<epoch>/samples.xyz after validation.",
    )
    parser.add_argument(
        "--num-log-wandb",
        type=int,
        default=25,
        help="Number of sampled/GT structures to show in W&B image grids, matching facit callback default.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable detailed KLDM/TDM convergence diagnostics in logs and W&B.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()
    print(f"starting trainHPC root={root} device={device.type}", flush=True)

    config = {
        "task": "CSP",
        "seed": GLOBAL_SEED,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_v": 1.0,
        "lambda_l": 1.0,
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
        "save_top_k": args.save_top_k,
        "log_sampled_atoms": args.log_sampled_atoms,
        "num_log_wandb": args.num_log_wandb,
        "num_workers": LOADER_NUM_WORKERS,
        "pin_memory": LOADER_PIN_MEMORY,
        "dev": args.dev,
    }

    set_global_seed(config["seed"])

    train_loader = CSPTask().dataloader(
        root=root,
        split="train",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        download=True,
    )
    val_dataset_full = CSPTask().fit_dataset(
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
    )

    print("constructed train/val loaders", flush=True)

    tdm = TrivialisedDiffusionDev(
        eps=1e-6,
        n_lambdas=512 if device.type == "cuda" else 128,
        lambda_num_batches=32 if device.type == "cuda" else 8,
        n_sigmas=2000 if device.type == "cuda" else 512,
    )
    model = ModelKLDM(
        device=device,
        diffusion_v=tdm,
        lattice_eps=1e-3,
        lattice_parameterization="x0",
    ).to(device)
    print("precomputing TDM tables on real train batches", flush=True)
    model.tdm.precompute_lambda_v_table_from_loader(
        train_loader,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        amsgrad=True,
        foreach=True,
        weight_decay=config["weight_decay"],
    )
    ema = ExponentialMovingAverage(
        model=model,
        decay=config["ema_decay"],
        start_epoch=config["ema_start"],
    )
    lambda_table = getattr(model.tdm, "_lambda_v_table", None)
    if args.dev and isinstance(lambda_table, torch.Tensor):
        print(
            "lambda_v_table_stats "
            f"min={float(lambda_table.min()):.6f} "
            f"mean={float(lambda_table.mean()):.6f} "
            f"max={float(lambda_table.max()):.6f} "
            f"p25={float(torch.quantile(lambda_table, 0.25)):.6f} "
            f"p50={float(torch.quantile(lambda_table, 0.50)):.6f} "
            f"p75={float(torch.quantile(lambda_table, 0.75)):.6f}",
            flush=True,
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
    samples_dir = artifacts_dir / "samples"
    history_path = artifacts_dir / "training_history.csv"
    sampling_history_path = artifacts_dir / "sampling_history.csv"
    training_plot_path = artifacts_dir / "training_metrics.png"
    sampling_plot_path = artifacts_dir / "sampling_metrics.png"
    print(f"using artifacts_dir={artifacts_dir}", flush=True)

    history: list[dict[str, Any]] = []
    top_k_records = load_top_k_records(checkpoints_dir)
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
                lambda_v=config["lambda_v"],
                lambda_l=config["lambda_l"],
                debug=args.dev,
            )

            row = {
                "epoch": epoch,
                "train_loss_weighted": train_metrics["loss"],
                "train_loss_v": train_metrics["loss_v"],
                "train_loss_l": train_metrics["loss_l"],
                "val_loss_weighted": float("nan"),
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
                    f"train_loss_weighted={train_metrics['loss']:.6f} "
                    f"(v={train_metrics['loss_v']:.6f}, raw_v={train_metrics['raw_loss_v']:.6f}, l={train_metrics['loss_l']:.6f}) "
                    f"target_v_abs={train_metrics['target_v_abs_mean']:.6f} "
                    f"pred_v_abs={train_metrics['pred_v_abs_mean']:.6f} "
                    f"lambda=[{train_metrics['lambda_v_min']:.3f},{train_metrics['lambda_v_mean']:.3f},{train_metrics['lambda_v_max']:.3f}] "
                    f"grad=[total={train_metrics['grad_norm']:.3f},v={train_metrics['out_v_grad_norm']:.3f},l={train_metrics['out_l_grad_norm']:.3f}]"
                )
            else:
                print(
                    f"epoch={epoch:04d} "
                    f"train_loss_weighted={train_metrics['loss']:.6f} "
                    f"(loss_v={train_metrics['loss_v']:.6f}, loss_l={train_metrics['loss_l']:.6f})"
                )

            should_run_sampling = (epoch % config["validate_every"] == 0)

            train_log_payload = {
                "epoch": epoch,
                "train/loss_weighted": train_metrics["loss"],
                "train/loss_v": train_metrics["loss_v"],
                "train/loss_l": train_metrics["loss_l"],
            }
            if args.dev:
                train_log_payload.update(
                    {
                        "train/raw_loss_v": train_metrics["raw_loss_v"],
                        "train/target_v_abs_mean": train_metrics["target_v_abs_mean"],
                        "train/target_v_norm_mean": train_metrics["target_v_norm_mean"],
                        "train/pred_v_abs_mean": train_metrics["pred_v_abs_mean"],
                        "train/pred_v_norm_mean": train_metrics["pred_v_norm_mean"],
                        "train/lambda_v_mean": train_metrics["lambda_v_mean"],
                        "train/lambda_v_min": train_metrics["lambda_v_min"],
                        "train/lambda_v_max": train_metrics["lambda_v_max"],
                        "train/lambda_v_effective": train_metrics["lambda_v_effective"],
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
                    "train_loss_weighted",
                    "train_loss_v",
                    "train_loss_l",
                    "val_loss_weighted",
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

            checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
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
            use_ema_sampling = should_use_ema_for_sampling(
                ema=ema,
                current_epoch=epoch,
                force_ema=config["sampling_force_ema"],
            )
            if use_ema_sampling:
                print(
                    f"epoch={epoch:04d} running facit-style validation pass with EMA sampling override",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch:04d} running facit-style validation pass on current model",
                    flush=True,
                )

            val_metrics, sampling_metrics, logged_pred_structures, logged_target_structures = run_facit_style_validation_epoch(
                model=model,
                loader=val_loader,
                device=device,
                lambda_v=config["lambda_v"],
                lambda_l=config["lambda_l"],
                n_steps=config["sampling_steps"],
                ema=ema,
                current_epoch=epoch,
                force_ema_sampling=config["sampling_force_ema"],
                max_graphs=config["val_subset_graphs"],
                debug=args.dev,
            )
            print(
                f"epoch={epoch:04d} finished facit-style validation pass",
                flush=True,
            )

            row["val_loss_weighted"] = val_metrics["loss"]
            row["val_loss_v"] = val_metrics["loss_v"]
            row["val_loss_l"] = val_metrics["loss_l"]
            row["valid"] = sampling_metrics["valid"]
            row["match_rate"] = sampling_metrics["match_rate"]
            row["rmse"] = sampling_metrics["rmse"]

            if args.dev:
                print(
                    f"validation_epoch={epoch:04d} "
                    f"val_loss_weighted={val_metrics['loss']:.6f} "
                    f"(loss_v={val_metrics['loss_v']:.6f}, raw_v={val_metrics['raw_loss_v']:.6f}, loss_l={val_metrics['loss_l']:.6f}) "
                    f"valid={sampling_metrics['valid']:.4f} "
                    f"match_rate={sampling_metrics['match_rate']:.4f} "
                    f"rmse={sampling_metrics['rmse']:.6f}"
                )
            else:
                print(
                    f"validation_epoch={epoch:04d} "
                    f"val_loss_weighted={val_metrics['loss']:.6f} "
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

            val_log_payload = {
                "epoch": epoch,
                "val/loss_weighted": val_metrics["loss"],
                "val/loss_v": val_metrics["loss_v"],
                "val/loss_l": val_metrics["loss_l"],
                "val/valid": sampling_metrics["valid"],
                "val/match_rate": sampling_metrics["match_rate"],
                "val/rmse": sampling_metrics["rmse"],
            }
            if args.dev:
                val_log_payload.update(
                    {
                        "val/raw_loss_v": val_metrics["raw_loss_v"],
                        "val/target_v_abs_mean": val_metrics["target_v_abs_mean"],
                        "val/target_v_norm_mean": val_metrics["target_v_norm_mean"],
                        "val/pred_v_abs_mean": val_metrics["pred_v_abs_mean"],
                        "val/pred_v_norm_mean": val_metrics["pred_v_norm_mean"],
                        "val/lambda_v_mean": val_metrics["lambda_v_mean"],
                        "val/lambda_v_min": val_metrics["lambda_v_min"],
                        "val/lambda_v_max": val_metrics["lambda_v_max"],
                        "val/lambda_v_effective": val_metrics["lambda_v_effective"],
                        "val/v_t_abs_mean": val_metrics["v_t_abs_mean"],
                        "val/f_t_abs_mean": val_metrics["f_t_abs_mean"],
                        "val/r_t_abs_mean": val_metrics["r_t_abs_mean"],
                        "val/score_v_abs_mean": val_metrics["score_v_abs_mean"],
                        "val/pred_l_abs_mean": val_metrics["pred_l_abs_mean"],
                        "val/target_l_abs_mean": val_metrics["target_l_abs_mean"],
                    }
                )
            wandb.log(val_log_payload)

            if config["log_sampled_atoms"]:
                try:
                    log_sampled_structures(
                        epoch=epoch,
                        generated_structures=logged_pred_structures,
                        gt_structures=logged_target_structures,
                        output_dir=samples_dir,
                        run=run,
                        num_log_wandb=config["num_log_wandb"],
                    )
                except Exception as exc:
                    print(
                        f"epoch={epoch:04d} warning: sampled structure logging failed: {exc}",
                        flush=True,
                    )

            try_export_csv(
                rows=history,
                output_path=history_path,
                fieldnames=[
                    "epoch",
                    "train_loss_weighted",
                    "train_loss_v",
                    "train_loss_l",
                    "val_loss_weighted",
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
                        "val_loss_weighted": row["val_loss_weighted"],
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
                    "val_loss_weighted",
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

            if checkpoint_written:
                top_k_records = update_top_k_checkpoints(
                    checkpoints_dir=checkpoints_dir,
                    current_checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    metric_value=float(sampling_metrics["match_rate"]),
                    top_k_records=top_k_records,
                    save_top_k=config["save_top_k"],
                )

            if run is not None and checkpoint_written:
                try:
                    print(
                        f"epoch={epoch:04d} logging checkpoint artifact",
                        flush=True,
                    )
                    checkpoint_artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
                    if checkpoint_path.exists():
                        checkpoint_artifact.add_file(str(checkpoint_path))
                    last_checkpoint_path = checkpoints_dir / "last.pt"
                    if last_checkpoint_path.exists():
                        checkpoint_artifact.add_file(str(last_checkpoint_path))
                    run.log_artifact(checkpoint_artifact)
                    print(
                        f"epoch={epoch:04d} finished checkpoint artifact",
                        flush=True,
                    )

                    if sampling_history_path.exists():
                        print(
                            f"epoch={epoch:04d} saving sampling history to wandb",
                            flush=True,
                        )
                        wandb.save(str(sampling_history_path))
                    if sampling_plot_path.exists():
                        print(
                            f"epoch={epoch:04d} saving sampling plot to wandb",
                            flush=True,
                        )
                        wandb.save(str(sampling_plot_path))
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
                f"epoch={epoch:04d} updated facit-style checkpoint set",
                flush=True,
            )

            print(
                f"epoch={epoch:04d} validation block complete",
                flush=True,
            )

    finally:
        final_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}_final.pt"
        final_checkpoint_written = try_export_model_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            output_path=final_checkpoint_path,
            config=config | {"completed_epochs": epoch},
            epoch=epoch,
        )

        final_artifact = wandb.Artifact("csp_hpc_final", type="model")
        if final_checkpoint_written and final_checkpoint_path.exists():
            final_artifact.add_file(str(final_checkpoint_path))
        last_checkpoint_path = checkpoints_dir / "last.pt"
        if last_checkpoint_path.exists():
            final_artifact.add_file(str(last_checkpoint_path))
        top_k_json_path = checkpoints_dir / "top_k.json"
        if top_k_json_path.exists():
            final_artifact.add_file(str(top_k_json_path))
        top_k_csv_path = checkpoints_dir / "top_k.csv"
        if top_k_csv_path.exists():
            final_artifact.add_file(str(top_k_csv_path))
        if history_path.exists():
            final_artifact.add_file(str(history_path))
        if sampling_history_path.exists():
            final_artifact.add_file(str(sampling_history_path))
        if training_plot_path.exists():
            final_artifact.add_file(str(training_plot_path))
        if sampling_plot_path.exists():
            final_artifact.add_file(str(sampling_plot_path))
        if samples_dir.exists():
            for sample_file in samples_dir.glob("*/samples.xyz"):
                final_artifact.add_file(str(sample_file))
        if run is not None:
            run.log_artifact(final_artifact)
            wandb.finish()


if __name__ == "__main__":
    train()
