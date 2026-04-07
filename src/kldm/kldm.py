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

    def _tdm_reverse_exp_step(
        self,
        t_node: torch.Tensor,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        score_v: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse exponential-integrator step for the TDM velocity process.
        """
        del t_node

        dt_t = torch.as_tensor(dt, device=v_t.device, dtype=v_t.dtype)
        noise_v = torch.randn_like(v_t)

        exp_dt = torch.exp(dt_t)
        expm1_dt = torch.expm1(dt_t)
        noise_scale = torch.sqrt(torch.expm1(2.0 * dt_t))

        v_prev = exp_dt * v_t + 2.0 * expm1_dt * score_v + noise_scale * noise_v
        f_prev = self.tdm.wrap_fractional(f_t - dt_t * v_prev)

        return f_prev, v_prev


    # ============================================================================
    # ALGORITHM 2
    # ============================================================================

    def algorithm2_loss(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        lambda_v: float = 1.0,
        lambda_l: float = 1.0,
        lambda_a: float = 20, #Appendix, page 29
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



    # ============================================================================
    # ALGORITHM 3 - Sampling algorithm
    # ============================================================================
    def _load_dng_checkpoint(self, checkpoint_path: str = "artifacts/dng_final_model.pt") -> "ModelKLDM":
        device = next(self.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ModelKLDM(device=device).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _vp_denoise_x0_from_eps(
        self,
        x_t: torch.Tensor,
        pred_eps: torch.Tensor,
        t: torch.Tensor,
        diffusion: ContinuousVPDiffusion,
    ) -> torch.Tensor:
        alpha_t = diffusion._match_dims(diffusion.alpha(t), x_t)
        sigma_t = diffusion._match_dims(diffusion.sigma(t), x_t)
        return (x_t - sigma_t * pred_eps) / alpha_t.clamp_min(1e-8)

    def _vp_reverse_step_from_eps(
        self,
        x_t: torch.Tensor,
        pred_eps: torch.Tensor,
        t_now: torch.Tensor,
        t_next: torch.Tensor,
        diffusion: ContinuousVPDiffusion,
    ) -> torch.Tensor:
        """
        Stable VP reverse step using the model's epsilon prediction.

        We first estimate x0 from the current noisy state, then sample the next
        time level from the closed-form VP marginal at t_next:

            x0_hat = (x_t - sigma(t_now) * eps_hat) / alpha(t_now)
            x_{t_next} = alpha(t_next) * x0_hat + sigma(t_next) * z
        """
        x0_hat = self._vp_denoise_x0_from_eps(
            x_t=x_t,
            pred_eps=pred_eps,
            t=t_now,
            diffusion=diffusion,
        )
        alpha_next = diffusion._match_dims(diffusion.alpha(t_next), x_t)
        sigma_next = diffusion._match_dims(diffusion.sigma(t_next), x_t)
        noise = torch.randn_like(x_t)
        return alpha_next * x0_hat + sigma_next * noise

    def _construct_velocity_score(
        self,
        t_node: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
    ) -> torch.Tensor:
        t_internal = self.tdm.time_scaling_T * t_node
        prefactor = self.tdm._match_dims(
            (1.0 - torch.exp(-t_internal)) / (1.0 + torch.exp(-t_internal)),
            pred_v,
        )
        sigma_v_sq = self.tdm._match_dims(
            self.tdm.sigma_v(t_internal) ** 2,
            pred_v,
        )
        return prefactor * pred_v - v_t / sigma_v_sq.clamp_min(1e-8)

    def sample_DNG_algorithm3(
        self,
        n_steps: int,
        batch: Batch | Data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 3 in KLDM paper for the DNG task.

        Returns:
            f_0, v_t, l_0, a_0
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        node_index = batch.batch
        edge_node_index = batch.edge_node_index
        num_graphs = batch.num_graphs

        # Step 1: sample priors
        f_t = torch.rand_like(batch.pos)
        v_t = torch.randn_like(batch.pos)
        l_t = torch.randn_like(batch.l)
        a_t = torch.randn_like(batch.h)

        # Load trained score model s_theta
        trained_model = self._load_dng_checkpoint("artifacts/dng_final_model.pt")

        # Reverse-time grid: 1 -> eps
        ts = torch.linspace(1.0, self.eps, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                t_now = ts[i]
                t_next = ts[i + 1]
                dt = float((t_now - t_next).item())  # positive step size

                t_graph = torch.full((num_graphs, 1), float(t_now.item()), device=device)
                t_node = t_graph[node_index].squeeze(-1)

                # Step 2: network prediction
                preds = trained_model.score_network(
                    t=t_graph,
                    pos=f_t,
                    v=v_t,
                    h=a_t,
                    l=l_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                )

                out_v = preds["v"]
                out_l = preds["l"]
                out_a = preds["h"]

                # Step 3: construct velocity score, KLDM Eq. (19)
                score_v = self._construct_velocity_score(
                    t_node=t_node,
                    v_t=v_t,
                    pred_v=out_v,
                )

                # Step 5: reverse updates
                f_t, v_t = self._tdm_reverse_exp_step(
                    t_node=t_node,
                    f_t=f_t,
                    v_t=v_t,
                    score_v=score_v,
                    dt=dt,
                )

                t_graph_next = torch.full((num_graphs, 1), float(t_next.item()), device=device)
                t_node_next = t_graph_next[node_index].squeeze(-1)

                l_t = self._vp_reverse_step_from_eps(
                    x_t=l_t,
                    pred_eps=out_l,
                    t_now=t_graph.squeeze(-1),
                    t_next=t_graph_next.squeeze(-1),
                    diffusion=self.diffusion_l,
                )

                a_t = self._vp_reverse_step_from_eps(
                    x_t=a_t,
                    pred_eps=out_a,
                    t_now=t_node,
                    t_next=t_node_next,
                    diffusion=self.diffusion_a,
                )

            # Final denoising step at t = eps
            t_graph = torch.full((num_graphs, 1), self.eps, device=device)
            t_node = t_graph[node_index].squeeze(-1)

            final_preds = trained_model.score_network(
                t=t_graph,
                pos=f_t,
                v=v_t,
                h=a_t,
                l=l_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )

            l_0 = self._vp_denoise_x0_from_eps(
                x_t=l_t,
                pred_eps=final_preds["l"],
                t=t_graph.squeeze(-1),
                diffusion=self.diffusion_l,
            )

            a_0 = self._vp_denoise_x0_from_eps(
                x_t=a_t,
                pred_eps=final_preds["h"],
                t=t_node,
                diffusion=self.diffusion_a,
            )

            f_0 = self.tdm.wrap_fractional(f_t)

        return f_0, v_t, l_0, a_0



def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from kldm.data import resolve_data_root
    root = resolve_data_root()

    loader = DNGTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    model = ModelKLDM(device=device).to(device)

    pos_t, v_t, l_t, h_t = model.sample_DNG_algorithm3(
        N=1000,
        batch=batch,
    )

    print("Sampled one DNG crystal")
    print("pos shape:", tuple(pos_t.shape))
    print("v shape:", tuple(v_t.shape))
    print("l shape:", tuple(l_t.shape))
    print("h shape:", tuple(h_t.shape))

    print("\nFirst 3 sampled fractional coordinates:")
    print(pos_t[:3])

    print("\nSampled lattice:")
    print(l_t)

if __name__ == "__main__":
    main()
