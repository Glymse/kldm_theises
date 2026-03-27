from __future__ import annotations

from pathlib import Path
import logging
import sys
from typing import Callable, Optional

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

logger = logging.getLogger(__name__)


class ModelKLDM(nn.Module):
    """
    KLDM model
    """

    def __init__(
        self,
        score_network: Optional[CSPVNet] = None,
        diffusion_v: Optional[TDM] = None,
        diffusion_l: Optional[ContinuousVPDiffusion] = None,
        diffusion_a: Optional[ContinuousVPDiffusion] = None,
        device: Optional[torch.device] = None,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()

        self.score_network = score_network or CSPVNet(
            hidden_dim=128,
            num_layers=4,
            h_dim=118,
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
        self.task_type: Optional[str] = None

    # ============================================================================
    # ALGORITHM 1
    # ============================================================================

    def algorithm1_training_targets(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Algorithm 1 in KLDM:
        sample noisy variables and score targets.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        t_graph = t.to(device)
        if t_graph.ndim == 1:
            t_graph = t_graph[:, None]

        index = batch.batch
        t_node = t_graph[index].squeeze(-1)
        is_dng = hasattr(batch, "task_id") and bool((batch.task_id == 1).all())

        # Diffuse lattice, KLDM Alg. 1
        l_t, eps_l = self.diffusion_l.forward_sample(
            t=t_graph.squeeze(-1),
            x0=batch.l,
        )
        target_l = eps_l

        # Sample (f_t, v_t), KLDM Eqs. (16), (22), (23)
        f_t, v_t, epsilon_v, epsilon_r, r_t = self.tdm.forward_sample(
            t=t_node,
            f0=batch.pos,
        )

        # Velocity target, KLDM Eq. (19)
        target_v = self.tdm.score_target(
            t=t_node,
            epsilon_v=epsilon_v,
            r_t=r_t,
            v_t=v_t,
        )

        if is_dng:
            # Diffuse atom types, KLDM Alg. 1
            a_t, eps_a = self.diffusion_a.forward_sample(
                t=t_node,
                x0=batch.h,
            )
            target_a = eps_a
            return (v_t, f_t, l_t, a_t), (target_v, target_l, target_a)

        return (v_t, f_t, l_t), (target_v, target_l)

    # ============================================================================
    # Loss calculators for algorithm 2
    # ============================================================================

    def mse_loss_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plain MSE, averaged over feature dims.
        """
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.reshape(loss.shape[0], -1).mean(dim=1)


    @staticmethod
    def lambda_t(
        t_node: torch.Tensor,
        lambda_t_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Resolve the KLDM velocity weighting lambda(t).
        """
        if lambda_t_fn is None:
            return torch.ones_like(t_node)
        return lambda_t_fn(t_node).to(t_node.device)

    # ============================================================================
    # ALGORITHM 2
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
        Algorithm 2 in KLDM:
        network prediction + denoising score matching loss.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        t_graph = t.to(device)
        if t_graph.ndim == 1:
            t_graph = t_graph[:, None]

        index = batch.batch
        t_node = t_graph[index].squeeze(-1)
        is_dng = hasattr(batch, "task_id") and bool((batch.task_id == 1).all())

        # Algorithm 1
        noisy, targets = self.algorithm1_training_targets(batch=batch, t=t_graph)

        if is_dng:
            v_t, f_t, l_t, a_t = noisy
            target_v, target_l, target_a = targets
        else:
            v_t, f_t, l_t = noisy
            target_v, target_l = targets
            a_t = batch.h

        # Network prediction, KLDM Alg. 2
        preds = self.score_network(
            t=t_graph,
            pos=f_t,
            v=v_t,
            h=a_t,
            l=l_t,
            node_index=index,
            edge_node_index=batch.edge_node_index,
        )

        ########HERE WE CALCULATE SIMPLIFIED SCORE
        out_v = preds["v"]
        out_l = preds["l"]

        # Full velocity score, KLDM Eq. (19)
        t_internal = self.tdm.time_scaling_T * t_node
        exp_coef = self.tdm._match_dims((1.0 - torch.exp(-t_internal)) / (1.0 + torch.exp(-t_internal)), out_v)
        sigma_v_sq = self.tdm._match_dims(self.tdm.sigma_v(t_internal) ** 2, out_v)

        #The simplified score, assuming initial velocity is 0
        score_v = exp_coef * out_v - v_t / sigma_v_sq

        # KLDM: plain squared error for lattice and atom targets.
        loss_l = self.mse_loss_per_sample(preds["l"], target_l).mean()

        # KLDM: lambda(t) * ||score_v - target_v||^2
        lambda_t = self.lambda_t(t_node=t_node, lambda_t_fn=lambda_t_fn)
        loss_v = (lambda_t * self.mse_loss_per_sample(score_v, target_v)).mean()

        if is_dng:
            loss_a = self.mse_loss_per_sample(preds["h"], target_a).mean()

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
