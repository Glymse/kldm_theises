from __future__ import annotations

from pathlib import Path
import logging
import sys
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

from kldm.data import CSPTask
from kldm.diffusionModels.continuous import ContinuousVPDiffusion
from kldm.diffusionModels.trivialized_diffusion import TrivialisedDiffusion as TDM
from kldm.distribution.uniform import sample_uniform
from kldm.scoreNetwork.scoreNetwork import CSPVNet
from kldm.scoreNetwork.utils import scatter_center


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
        #diffusion_h: Optional[ContinuousVPDiffusion] = None, Deactive when DNG ready
        device: Optional[torch.device] = None,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        self.score_network = score_network or CSPVNet(
            hidden_dim=128,
            time_dim=128,
            num_layers=4,
            num_freqs=128,
            ln=True,
            h_dim=118,
            smooth=False,
            pred_v=True,
            pred_l=True,
            pred_h=False,
            zero_cog=True # center-of-mass / zero mean per graph.
        )

        self.tdm = diffusion_v or TDM(
            eps=eps,
            sigma_norm_num_samples=20000 if self.device.type == "cuda" else 4096,
            sigma_norm_chunk_size=128 if self.device.type == "cuda" else 32,
        )
        self.diffusion_l = diffusion_l or ContinuousVPDiffusion(eps=eps)
        self.eps = eps
        self.task_type: Optional[str] = None
        self._cached_sampling_checkpoint_path: Optional[str] = None
        self.__dict__["_cached_sampling_score_network_obj"] = None

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
        # Diffuse lattice, KLDM Alg. 1
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

        target_v = self.tdm.score_target(
            t=t_node,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )


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

    # ============================================================================
    # ALGORITHM 2
    # ============================================================================

    def algorithm2_loss(
        self,
        batch: Data | Batch,
        t: torch.Tensor,
        lambda_v: float = 1.0,
        lambda_l: float = 1.0,
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
        # Algorithm 1
        noisy, targets = self.algorithm1_training_targets(batch=batch, t=t_graph)

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

        # KLDM: plain squared error for lattice targets.
        loss_l = self.mse_loss_per_sample(preds["l"], target_l).mean()

        # Facit-style simplified target training uses plain MSE after target normalization.
        loss_v = self.mse_loss_per_sample(out_v, target_v).mean()

        total_loss = lambda_v * loss_v + lambda_l * loss_l
        return total_loss, {
            "loss": total_loss.detach(),
            "loss_v": loss_v.detach(),
            "loss_l": loss_l.detach(),
        }



    # ============================================================================
    # ALGORITHM 3 - Sampling algorithm
    # ============================================================================
    def _load_csp_score_network(self, checkpoint_path: str = "artifacts/csp_final_model.pt") -> CSPVNet:
        device = next(self.parameters()).device
        cached_score_network = self.__dict__.get("_cached_sampling_score_network_obj")
        if (
            cached_score_network is not None
            and self._cached_sampling_checkpoint_path == checkpoint_path
        ):
            return cached_score_network

        checkpoint = torch.load(checkpoint_path, map_location=device)
        cleaned_state_dict = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if not key.startswith("_cached_sampling_score_network")
        }
        model = ModelKLDM(device=device).to(device)
        model.load_state_dict(cleaned_state_dict, strict=False)
        score_network = model.score_network.to(device)
        score_network.eval()

        self._cached_sampling_checkpoint_path = checkpoint_path
        self.__dict__["_cached_sampling_score_network_obj"] = score_network
        return score_network

    def sample_CSP_algorithm3(
        self,
        n_steps: int,
        batch: Batch | Data,
        checkpoint_path: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithm 3 sampling for CSP, KLDM-epsilon version.

        Returns:
            f_N, v_N, l_N, a
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        node_index = batch.batch
        edge_node_index = batch.edge_node_index
        num_graphs = batch.num_graphs

        # Algorithm 3 priors:
        # v_T ~ N_v(0, I) with zero-net translation
        # f_T is kept in the centered unit-period chart internally.
        # l_T ~ N(0, I)
        v_t = scatter_center(torch.randn_like(batch.pos), index=node_index)
        f_t = self.tdm.wrap_displacements(torch.rand_like(batch.pos))
        l_t = torch.randn_like(batch.l)
        a_t = batch.h  # CSP conditioning

        if checkpoint_path is None:
            score_network = self.score_network
            restore_training = score_network.training
            score_network.eval()
        else:
            score_network = self._load_csp_score_network(checkpoint_path)
            restore_training = False

        dt = 1.0 / n_steps

        try:
            with torch.no_grad():
                for n in range(1, n_steps + 1):
                    t_scalar = 1.0 - (n - 1) * dt
                    t_graph = torch.full((num_graphs, 1), t_scalar, device=device)
                    t_node = t_graph[node_index].squeeze(-1)

                    preds = score_network(
                        t=t_graph,
                        pos=f_t,
                        v=v_t,
                        h=a_t,
                        l=l_t,
                        node_index=node_index,
                        edge_node_index=edge_node_index,
                    )

                    pred_v = preds["v"]
                    pred_l = preds["l"]

                    # Eq. (19): construct KLDM simplified velocity score
                    # Full velocity score:
                    # s_v(x_t, t) = ((1 - exp(-t)) / (1 + exp(-t))) * s_f_theta(x_t, t) - v_t / sigma_v(t)^2

                    score_v = self.tdm.construct_velocity_score(
                        t=t_node,
                        v_t=v_t,
                        pred_v=pred_v,
                    )

                    # Update v and f with exponential integrator
                    f_t, v_t = self.tdm.reverse_exp_step(
                        f_t=f_t,
                        v_t=v_t,
                        score_v=score_v,
                        index=node_index,
                        dt=dt, #This is scaled internally.
                    )

                    # KLDM-epsilon lattice branch:
                    # Algorithm 3 does EM on l using the score corresponding to out_l
                    l_t = self.diffusion_l.reverse_em_step_from_eps(
                        t=t_graph.squeeze(-1),
                        x_t=l_t,
                        pred_eps=pred_l,
                        dt=dt,
                    )

                # For KLDM-epsilon Algorithm 3, return the final sampled l_N directly
                return self.tdm.wrap_positions(f_t), v_t, l_t, a_t
        finally:
            if checkpoint_path is None and restore_training:
                score_network.train()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from kldm.data import resolve_data_root
    root = resolve_data_root()

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=False,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    model = ModelKLDM(device=device).to(device)

    pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
        n_steps=1000,
        batch=batch,
    )

    print("Sampled one CSP crystal")
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
