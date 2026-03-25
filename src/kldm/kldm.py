from __future__ import annotations

from pathlib import Path
import logging
import sys
from typing import Any, Callable, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

from kldm.data import DNGTask, CSPTask
from kldm.diffusionModels.continuous import ContinuousVPDiffusion
from kldm.diffusionModels.trivialized_diffusion import TrivialisedDiffusion as TDM
from kldm.distribution.uniform import sample_uniform
from kldm.scoreNetwork.scoreNetwork import CSPVNet
from kldm.scoreNetwork.utils import scatter_center

logger = logging.getLogger(__name__)


class ModelKLDM(nn.Module):
    r"""
    KLDM (Kinetic Lattice Diffusion Model)

    Implements Algorithms 1-4 from the KLDM paper for crystal structure generation.
    Key innovations:
    1. Trivialized momentum for velocity in tangent space
    2. VP diffusion for lattice parameters and atom types (one-hot-encoded)
    3. Score-based matching with equivariant GNN (CSPVNet)
    4. Separate handling of CSP and DNG tasks

    Components:
    - score_network (CSPVNet): Equivariant graph neural network for score prediction
    - diffusion_v: Trivialized diffusion for atomic velocities
    - diffusion_l: VP diffusion for lattice parameters
    - diffusion_a: VP diffusion for atom types (DNG only)

    Reference: "KLDM: Generative Modeling of Crystal Structure with Continuous Diffusion"
    """

    def __init__(
        self,
        score_network: Optional[CSPVNet] = None,
        diffusion_v: Optional[TDM] = None,
        diffusion_l: Optional[ContinuousVPDiffusion] = None,
        diffusion_a: Optional[ContinuousVPDiffusion] = None,
        device: Optional[torch.device] = None,
        eps = 1e-3,

    ) -> None:
        """
        Initialize KLDM model.

        Args:
            score_network: Equivariant GNN for score prediction. If None, uses CSPVNet()
            diffusion_v: Velocity diffusion model. If None, uses TrivialisedDiffusionMomentum()
            diffusion_l: Lattice diffusion model. If None, uses ContinuousVPDiffusion()
            diffusion_a: Atom-type diffusion model. If None, uses ContinuousVPDiffusion()
            eps: Small epsilon for numerical stability
            device: Device for model. If None, uses CPU
        """
        super().__init__()

        # Components
        self.score_network = score_network or CSPVNet(
            hidden_dim=128,
            num_layers=4,
            h_dim=118,  # Number of elements
            smooth=True,
            pred_v=True,
            pred_l=True,
            pred_h=True,
        )
        self.tdm = diffusion_v or TDM(eps=eps)
        self.diffusion_l = diffusion_l or ContinuousVPDiffusion(eps=eps)
        self.diffusion_a = diffusion_a or ContinuousVPDiffusion(eps=eps)

        self.device = device or torch.device("cpu")
        self.eps = eps

        # Task detection (set after first batch)
        self.task_type: Optional[str] = None  # "csp" or "dng"

    # ============================================================================
    # ALGORITHM 1: Training Targets
    # ============================================================================

    def algorithm1_training_targets(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        device = next(self.parameters()).device
        batch = batch.to(device)

        t_graph = t.to(device)
        if t_graph.ndim == 1:
            t_graph = t_graph[:, None]

        index = batch.batch
        t_node = t_graph[index].squeeze(-1)
        is_dng = hasattr(batch, "task_id") and bool((batch.task_id == 1).all())

        l_t, eps_l = self.diffusion_l.forward_sample(
            t=t_graph.squeeze(-1),
            x0=batch.l,
        )
        target_l = eps_l

        f_t, v_t, epsilon_v, epsilon_r, r_t = self.tdm.forward_sample(
            t=t_node,
            f0=batch.pos,
            index=index,
        )
        f_t = scatter_center(f_t, index=index)
        target_v = self.tdm.score_target(
            t=t_node,
            epsilon_v=epsilon_v,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )

        if is_dng:
            a_t, eps_a = self.diffusion_a.forward_sample(
                t=t_node,
                x0=batch.h,
            )
            target_a = eps_a
            return (v_t, f_t, l_t, a_t), (target_v, target_l, target_a)

        return (v_t, f_t, l_t), (target_v, target_l)



    # ============================================================================
    # ALGORITHM 2: Denoising Score Matching Loss / Training model
    # ============================================================================


    def algorithm2_loss(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        lambda_v: float = 1.0,
        lambda_l: float = 1.0,
        lambda_a: float = 1.0,
        lambda_t_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,

    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Algorithm 2 from KLDM:
        - get noisy samples + targets from Algorithm 1
        - run score network
        - reconstruct full velocity score using Eq. (19)
        - compute weighted DSM losses
        """

        device = next(self.parameters()).device
        batch = batch.to(device)

        t_graph = t.to(device)
        if t_graph.ndim == 1:
            t_graph = t_graph[:, None]

        index = batch.batch
        t_node = t_graph[index].squeeze(-1)
        is_dng = hasattr(batch, "task_id") and bool((batch.task_id == 1).all())

        noisy, targets = self.algorithm1_training_targets(batch=batch, t=t_graph)
        if is_dng:
            v_t, f_t, l_t, a_t = noisy
            target_v, target_l, target_a = targets
        else:
            v_t, f_t, l_t = noisy
            target_v, target_l = targets
            a_t = batch.h

        preds = self.score_network(
            t=t_graph,
            pos=f_t,
            v=v_t,
            h=a_t,
            l=l_t,
            node_index=index,
            edge_node_index=batch.edge_node_index,
        )

        out_v = scatter_center(preds["v"], index=index)
        out_l = preds["l"]

        t_internal = self.tdm.time_scaling_T * t_node
        prefactor = (1.0 - torch.exp(-t_internal)) / (1.0 + torch.exp(-t_internal))
        prefactor = self.tdm._match_dims(prefactor, out_v)
        sigma_v_sq = self.tdm._match_dims(self.tdm.sigma_v(t_internal) ** 2, out_v)

        score_v = prefactor * out_v - v_t / sigma_v_sq
        score_v = scatter_center(score_v, index=index)

        loss_l = F.mse_loss(out_l, target_l, reduction="none")
        loss_l = loss_l.reshape(loss_l.shape[0], -1).mean(dim=1).mean()

        loss_v = F.mse_loss(score_v, target_v, reduction="none")
        loss_v = loss_v.reshape(loss_v.shape[0], -1).mean(dim=1)
        if lambda_t_fn is None:
            lambda_t = torch.ones_like(loss_v)
        elif callable(lambda_t_fn):
            lambda_t = lambda_t_fn(t_node).to(loss_v.device)
        else:
            lambda_t = torch.as_tensor(lambda_t_fn, device=loss_v.device, dtype=loss_v.dtype)
            if lambda_t.ndim > 1:
                lambda_t = lambda_t[index].squeeze(-1) if lambda_t.shape[0] == t_graph.shape[0] else lambda_t.squeeze()
            if lambda_t.ndim == 0:
                lambda_t = torch.full_like(loss_v, float(lambda_t))
        loss_v = (lambda_t * loss_v).mean()

        if is_dng:
            out_a = preds["h"]
            loss_a = F.mse_loss(out_a, target_a, reduction="none")
            loss_a = loss_a.reshape(loss_a.shape[0], -1).mean(dim=1).mean()
            total_loss = lambda_v * loss_v + lambda_l * loss_l + lambda_a * loss_a
            return total_loss, {
                "loss": total_loss.detach(),
                "loss_v": loss_v.detach(),
                "loss_l": loss_l.detach(),
                "loss_a": loss_a.detach(),
            }

        total_loss = lambda_v * loss_v + lambda_l * loss_l
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_v": loss_v.detach(),
            "loss_l": loss_l.detach(),
        }




def train(
    model: ModelKLDM,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    print_every: int = 20,
):
    """Short KLDM training loop using Algorithm 2."""
    model.train()
    running = {"loss": 0.0, "loss_v": 0.0, "loss_l": 0.0, "loss_a": 0.0}
    n_batches = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        num_graphs = batch.num_graphs
        t = sample_uniform(lb=model.eps, size=(num_graphs, 1), device=device)

        optimizer.zero_grad()

        loss, metrics = model.algorithm2_loss(
            batch=batch,
            t=t,
            lambda_v=1.0,
            lambda_l=1.0,
            lambda_a=1.0,
            lambda_t_fn=torch.full_like(t, 4.0) # # since time_scaling_T = 2
        )

        loss.backward()
        optimizer.step()

        n_batches += 1
        for k, v in metrics.items():
            if k in running:
                running[k] += float(v)

        if step % print_every == 0:
            prefix = f"epoch={epoch:02d} " if epoch is not None else ""
            print(
                f"{prefix}step={step:03d} "
                f"loss={float(metrics['loss']):.6f} "
                f"(v={float(metrics['loss_v']):.4f}, "
                f"l={float(metrics['loss_l']):.4f}, "
                f"a={float(metrics.get('loss_a', torch.tensor(0.0))):.4f})"
            )

    for k in running:
        if n_batches > 0:
            running[k] /= n_batches

    return running


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DNGTask().dataloader(
        split="train",
        batch_size=64,
        shuffle=True,
        download=True,
    )

    model = ModelKLDM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5

    for epoch in range(num_epochs):
        stats = train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            print_every=20,
        )
        print(
            f"epoch={epoch:02d} mean_loss={stats['loss']:.6f} "
            f"(v={stats['loss_v']:.4f}, l={stats['loss_l']:.4f}, a={stats['loss_a']:.4f})"
        )


if __name__ == "__main__":
    main()
