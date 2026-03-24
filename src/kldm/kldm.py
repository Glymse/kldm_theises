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
from kldm.diffusionModels.trivialized_diffusion import TrivialisedDiffusionMomentum
from kldm.scoreNetwork.scoreNetwork import CSPVNet
from kldm.distribution.uniform import sample_uniform

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
        diffusion_v: Optional[TrivialisedDiffusionMomentum] = None,
        diffusion_l: Optional[ContinuousVPDiffusion] = None,
        diffusion_a: Optional[ContinuousVPDiffusion] = None,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
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
        self.diffusion_v = diffusion_v or TrivialisedDiffusionMomentum(eps=eps)
        self.diffusion_l = diffusion_l or ContinuousVPDiffusion(eps=eps)
        self.diffusion_a = diffusion_a or ContinuousVPDiffusion(eps=eps)

        self.eps = eps
        self.device = device or torch.device("cpu")

        # Task detection (set after first batch)
        self.task_type: Optional[str] = None  # "csp" or "dng"

    def detect_task(self, batch: Data | Batch) -> str:
        """
        Detect task type from batch (CSP vs DNG).

        Args:
            batch: Input batch

        Returns:
            "csp" if atom types are fixed, "dng" if diffused
        """
        if hasattr(batch, "diffuse_h"):
            diffuse_h = batch.diffuse_h
            if isinstance(diffuse_h, torch.Tensor):
                diffuse_h = diffuse_h.item() if diffuse_h.numel() == 1 else diffuse_h[0].item()
            return "dng" if diffuse_h else "csp"

        # Default heuristic: check if atom types would be diffused
        return "csp"

    # ============================================================================
    # ALGORITHM 1: Training Targets
    # ============================================================================

    def training_targets(
        self,
        batch: Data | Batch,
        t: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 from KLDM paper: Compute training targets for score matching.

        For each component (velocity, lattice, atom types), sample from transition
        kernel and compute noise-prediction targets.

        Args:
            batch: Input crystal structure batch with fields:
                - pos: Atomic positions [n_atoms, 3]
                - h: Atom types [n_atoms]
                - lengths: Lattice lengths [n_graphs, 3]
                - angles: Lattice angles [n_graphs, 3]
                - edge_index: Graph edges
                - batch: Graph batch indices
            t: Time steps [n_graphs, 1]. If None, sampled uniformly from [eps, 1]

        Returns:
            Dictionary containing:
                - perturbed states: v_t, l_t, a_t (if applicable)
                - noise targets: target_v, target_l, target_a (if applicable)
                - intermediate values: eps_v, eps_l, eps_a (for loss weighting)
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        # Sample timesteps if not provided
        if t is None:
            n_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
            t = sample_uniform(lb=self.eps, size=(n_graphs, 1), device=device)

        # Detect task
        self.task_type = self.detect_task(batch)

        # Extract clean data
        v0 = self._extract_velocities(batch)  # [n_atoms, 3]
        l0 = self._extract_lattice(batch)      # [n_graphs, 6]
        h0 = batch.h if hasattr(batch, "h") else None  # [n_atoms]

        # Ensure h0 is in valid range [0, 118]
        if h0 is not None and h0.ndim == 1:
            h0 = h0.long()
            h0 = torch.clamp(h0, min=0, max=117)

        # Align time with graph structure
        graph_t = self._align_time_to_graph(t, batch)  # [n_atoms, 1] for velocity
        node_t = graph_t  # Per-atom timestep

        targets = {}

        # === VELOCITY DIFFUSION (Trivialized momentum) ===
        # v_t = alpha_v(t) * v0 + sigma_v(t) * eps_v, where v0 = 0 by design
        _, vt, eps_v = self.diffusion_v.perturb(t=node_t, f0=batch.pos, v0=v0)
        sigma_v = self.diffusion_v.sigma_v(node_t)

        # Target: score = -eps / sigma (epsilon-prediction)
        target_v = self.diffusion_v.target_score(t=node_t, vt=vt, eps_v=eps_v)

        # For positions, compute fractional coordinate sensitivity (Eq. 26 in paper)
        # This accounts for lattice structure during position perturbation
        target_s = self._compute_fractional_target(batch, node_t)
        target_v = target_v + target_s

        targets.update({
            "v_t": vt,
            "target_v": target_v,
            "eps_v": eps_v,
            "sigma_v": sigma_v,
        })

        # === LATTICE DIFFUSION (VP SDE) ===
        # l_t = alpha(t) * l0 + sigma(t) * eps_l
        lt, eps_l = self.diffusion_l.transition_kernel_sample(t=t, x0=l0)
        target_l_eps = eps_l  # Epsilon-prediction target

        targets.update({
            "l_t": lt,
            "target_l": target_l_eps,
            "eps_l": eps_l,
        })

        # === ATOM TYPE DIFFUSION (VP SDE) - DNG only ===
        if self.task_type == "dng" and h0 is not None:
            # Continuous representation of categorical distribution
            h_continuous = self._h_to_continuous(h0)  # [n_atoms, n_types]
            at, eps_a = self.diffusion_a.transition_kernel_sample(
                t=node_t, x0=h_continuous
            )
            target_a_eps = eps_a

            targets.update({
                "a_t": at,
                "target_a": target_a_eps,
                "eps_a": eps_a,
            })

        return targets


    # ============================================================================
    # ALGORITHM 2: Denoising Score Matching Loss
    # ============================================================================

    def denoise_score_matching(
        self,
        batch: Data | Batch,
        targets: dict[str, torch.Tensor],
        weights: Optional[dict[str, float]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 2 from KLDM paper: Compute denoising score matching loss.

        Loss = E_t [ sum_component w_i * || score_network(x_t, t) - target_i ||^2 ]

        Args:
            batch: Original crystal structures
            targets: Output from training_targets()
            weights: Loss weights for each component {v, l, a}. Default: uniform weights

        Returns:
            Dictionary with component losses and total loss:
                - loss_v, loss_l, loss_a (if applicable)
                - loss_total (weighted sum)
        """
        device = next(self.parameters()).device

        # Default weights
        if weights is None:
            weights = {"v": 1.0, "l": 1.0, "a": 1.0}

        # Construct perturbed batch for network input
        batch_t = self._construct_perturbed_batch(batch, targets)
        batch_t = batch_t.to(device)

        # Extract time for this batch
        if hasattr(batch_t, "batch"):
            t = torch.linspace(0.1, 0.9, batch_t.num_graphs, device=device).unsqueeze(-1)
        else:
            t = torch.tensor([[0.5]], device=device)

        # Score prediction from network
        scores_dict = self._predict_scores(batch_t, t)

        losses = {}

        # Velocity score loss
        if "v" in scores_dict and "target_v" in targets:
            loss_v = F.mse_loss(scores_dict["v"], targets["target_v"])
            losses["loss_v"] = loss_v * weights["v"]

        # Lattice score loss
        if "l" in scores_dict and "target_l" in targets:
            loss_l = F.mse_loss(scores_dict["l"], targets["target_l"])
            losses["loss_l"] = loss_l * weights["l"]

        # Atom-type score loss (DNG)
        if "h" in scores_dict and "target_a" in targets:
            loss_a = F.mse_loss(scores_dict["h"], targets["target_a"])
            losses["loss_a"] = loss_a * weights["a"]

        # Total loss
        losses["loss_total"] = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        return losses

    def training_step(
        self,
        batch: Data | Batch,
        batch_idx: int = 0,
        loss_weights: Optional[dict[str, float]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Single training step combining Algorithm 1 and 2.

        Args:
            batch: Crystal structure batch
            batch_idx: Batch index (for logging)
            loss_weights: Loss component weights

        Returns:
            Loss dictionary with breakdown
        """
        # Compute training targets (Algorithm 1)
        targets = self.training_targets(batch)

        # Compute score matching loss (Algorithm 2)
        losses = self.denoise_score_matching(batch, targets, weights=loss_weights)

        return losses

    # ============================================================================
    # ALGORITHM 3 & 4: Sampling / Generation
    # ============================================================================

    @torch.no_grad()
    def sample(
        self,
        batch: Data | Batch,
        steps: int = 100,
        solver: str = "euler",
        return_trajectory: bool = False,
    ) -> Data | Batch | list[Data | Batch]:
        """
        Algorithm 3-4: Generate crystal structures via reverse SDE/ODE.

        Integrates reverse process from noise distribution to data distribution.

        Args:
            batch: Initial/prior batch (noise or partial information)
            steps: Number of integration steps
            solver: ODE solver ("euler", "rk45", "heun")
            return_trajectory: If True, return full trajectory; else just final sample

        Returns:
            Final crystal structure batch, or list of intermediate states if return_trajectory=True
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        # Initialize states from noise
        batch_x = self._initialize_sampling_states(batch)

        # Time discretization (reverse: T -> 0)
        times = torch.linspace(1.0, self.eps, steps + 1, device=device)
        dt = times[0] - times[1]

        trajectory = [batch_x] if return_trajectory else []

        # Reverse integration
        for i in range(steps):
            t = times[i].view(1, -1) if times[i].dim() == 0 else times[i].unsqueeze(-1)
            dt_step = (times[i] - times[i + 1]).item()

            # Predict scores at current state
            scores_dict = self._predict_scores(batch_x, t)

            # Update states using reverse SDE/ODE
            batch_x = self._reverse_integration_step(
                batch_x, scores_dict, dt_step, solver=solver
            )

            if return_trajectory:
                trajectory.append(batch_x)

        # Post-process: ensure valid structures
        batch_x = self._postprocess_structures(batch_x)

        return trajectory if return_trajectory else batch_x

    @torch.no_grad()
    def sample_prior(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
    ) -> Data | Batch:
        """
        Sample from prior distribution (noise).

        Args:
            n_samples: Number of samples
            device: Device for sampling

        Returns:
            Batch with sampled noise states
        """
        device = device or self.device

        # Create noise structures
        batch = self._sample_noise_prior(n_samples, device=device)

        return batch

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _extract_velocities(self, batch: Data | Batch) -> torch.Tensor:
        """Extract or initialize velocities from batch."""
        if hasattr(batch, "v") and batch.v is not None:
            return batch.v
        # Default: zero velocities (as per paper)
        return torch.zeros_like(batch.pos)

    def _extract_lattice(self, batch: Data | Batch) -> torch.Tensor:
        """Extract lattice parameters as [lengths, angles]."""
        if not hasattr(batch, "lengths") or not hasattr(batch, "angles"):
            raise ValueError("Batch must contain 'lengths' and 'angles'")

        # Concatenate [lengths (3,), angles (3,)] -> [6,]
        return torch.cat([batch.lengths, batch.angles], dim=-1)

    def _align_time_to_graph(
        self, t: torch.Tensor, batch: Data | Batch
    ) -> torch.Tensor:
        """
        Align per-graph time steps to per-atom time steps.

        Args:
            t: [n_graphs, 1]
            batch: Graph batch

        Returns:
            [n_atoms, 1] with repeated time values
        """
        if hasattr(batch, "batch"):
            # batch.batch contains graph indices for each atom
            return t[batch.batch]
        return t.expand(batch.pos.shape[0], -1)

    def _compute_fractional_target(
        self, batch: Data | Batch, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute target for fractional coordinate sensitivity (Eq. 26 in paper).

        This accounts for how lattice deformation affects position coordinates.
        """
        # Simplified version: assume small sensitivity for now
        # Full version would include lattice_matrix jacobian
        return torch.zeros_like(batch.pos)

    def _h_to_continuous(self, h: torch.Tensor, n_types: int = 118) -> torch.Tensor:
        """Convert categorical atom types to continuous representation."""
        if h.ndim == 2:
            return h.to(dtype=torch.float32)

        h = h.to(dtype=torch.long)
        if h.numel() == 0:
            return torch.empty((0, n_types), dtype=torch.float32, device=h.device)

        if int(h.min().item()) >= 1:
            h = h - 1

        return F.one_hot(h, num_classes=n_types).float()

    def _construct_perturbed_batch(
        self, batch: Data | Batch, targets: dict[str, torch.Tensor]
    ) -> Data | Batch:
        """
        Create batch with perturbed states (x_t) for network input.

        CSPVNet expects: x_t = {pos, v, h, l, t, edge_index, node_index}
        """
        batch_t = batch.clone() if isinstance(batch, Data) else batch

        # Replace perturbed versions (already in targets as v_t, l_t, etc.)
        if "v_t" in targets:
            batch_t.v = targets["v_t"]

        if "l_t" in targets:
            l_t = targets["l_t"]
            batch_t.l = l_t
            batch_t.lengths = l_t[..., :3]
            batch_t.angles = l_t[..., 3:]

        if "a_t" in targets:
            batch_t.h = targets["a_t"]

        # Keep pos as-is or compute from v_t if needed
        # For now, pos stays the same (score network uses v separately)

        return batch_t

    def _parse_score_output(
        self, scores: dict[str, torch.Tensor], batch: Data | Batch
    ) -> dict[str, torch.Tensor]:
        """
        Parse score network output (which is already a dict from CSPVNet).

        CSPVNet returns:
            {
                "v": [n_atoms, 3] velocity scores (if pred_v=True),
                "l": [n_graphs, 6] lattice scores (if pred_l=True),
                "h": [n_atoms, n_types] atom type scores (if pred_h=True),
            }
        """
        # CSPVNet already outputs a dict, just return it
        return scores

    def _initialize_sampling_states(self, batch: Data | Batch) -> Data | Batch:
        """Initialize states for sampling (from noise)."""
        # Start from Gaussian noise
        batch_x = batch.clone() if isinstance(batch, Data) else batch

        batch_x.pos = torch.randn_like(batch.pos)
        batch_x.v = torch.randn_like(batch.pos) if hasattr(batch, "pos") else None

        if hasattr(batch_x, "lengths"):
            batch_x.lengths = torch.abs(torch.randn_like(batch_x.lengths)) + 3.0
        if hasattr(batch_x, "angles"):
            batch_x.angles = torch.abs(torch.randn_like(batch_x.angles)) * 60 + 60

        return batch_x

    def _predict_scores(
        self, batch_x: Data | Batch, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Predict scores at current state and time using CSPVNet.

        CSPVNet signature:
            forward(t, pos, v, h, l, node_index, edge_node_index) -> dict[str, Tensor]

        Args:
            batch_x: Current batch with pos, v, h, l
            t: Time [B, 1] or [1, 1]

        Returns:
            Scores dict with keys: v, l, h (depending on pred_* flags)
        """
        # Extract required fields
        pos = batch_x.pos  # [N, 3]
        v = batch_x.v if hasattr(batch_x, "v") else torch.zeros_like(batch_x.pos)  # [N, 3]
        h = batch_x.h if hasattr(batch_x, "h") else torch.zeros(batch_x.pos.shape[0], dtype=torch.long, device=batch_x.pos.device)  # [N]
        if getattr(self.score_network, "smooth", False):
            h = self._h_to_continuous(h)

        # Keep one-hot features as float for the smooth network path.
        # Only coerce to integer atom ids when the network uses embeddings.
        if h is not None and not getattr(self.score_network, "smooth", False):
            h = h.long()
            h = torch.clamp(h, min=0, max=117)

        # Lattice parameters
        if hasattr(batch_x, "l") and batch_x.l is not None:
            l = batch_x.l
        elif hasattr(batch_x, "lengths") and hasattr(batch_x, "angles"):
            lengths = batch_x.lengths.squeeze() if batch_x.lengths.dim() > 2 else batch_x.lengths
            angles = batch_x.angles.squeeze() if batch_x.angles.dim() > 2 else batch_x.angles
            # Ensure they're 2D [num_graphs, 3]
            if lengths.dim() == 1:
                lengths = lengths.unsqueeze(0)
            if angles.dim() == 1:
                angles = angles.unsqueeze(0)
            l = torch.cat([lengths, angles], dim=-1)  # [B, 6]
        else:
            l = torch.eye(3, device=batch_x.pos.device).unsqueeze(0).repeat(batch_x.num_graphs if hasattr(batch_x, "num_graphs") else 1, 1).view(-1, 6)

        # Graph batch indices
        if hasattr(batch_x, "batch"):
            node_index = batch_x.batch.long()  # [N]
        else:
            node_index = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        # Edge indices
        if hasattr(batch_x, "edge_node_index") and batch_x.edge_node_index is not None:
            edge_node_index = batch_x.edge_node_index  # [2, E]
        elif hasattr(batch_x, "edge_index") and batch_x.edge_index is not None:
            edge_node_index = batch_x.edge_index  # [2, E]
        else:
            # Construct full edge index if not present (fallback)
            edge_node_index = torch.tensor([[i, j] for i in range(pos.shape[0]) for j in range(pos.shape[0]) if i != j],
                                          dtype=torch.long, device=pos.device).T
            if edge_node_index.numel() == 0:
                # Single atom case
                edge_node_index = torch.zeros((2, 1), dtype=torch.long, device=pos.device)

        # Align time to per-graph
        # Ensure t is 2D: [num_graphs, 1] or [num_graphs, time_dim]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == 1:
            t = t.expand(batch_x.num_graphs if hasattr(batch_x, "num_graphs") else 1, -1)

        # Call CSPVNet
        scores = self.score_network(
            t=t,
            pos=pos,
            v=v,
            h=h,
            l=l,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )

        return scores

    def predict_scores(
        self,
        batch_x: Data | Batch,
        t: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        """Public wrapper used by the sampling utilities."""
        device = batch_x.pos.device

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        else:
            t = t.to(device=device, dtype=torch.float32)

        if t.dim() == 0:
            num_graphs = batch_x.num_graphs if hasattr(batch_x, "num_graphs") else 1
            t = t.repeat(num_graphs).unsqueeze(-1)

        return self._predict_scores(batch_x, t)

    def _reverse_integration_step(
        self,
        batch_x: Data | Batch,
        scores_dict: dict[str, torch.Tensor],
        dt: float,
        solver: str = "euler",
    ) -> Data | Batch:
        """
        Single reverse integration step.

        dx = 0.5 * beta(t) * score(x, t) * dt  (for VP)
        dv = (alpha_v(t)/sigma_v(t)) * score_v(x, t) * dt  (for trivialized)
        """
        batch_x_next = batch_x.clone() if isinstance(batch_x, Data) else batch_x

        if solver == "euler":
            # Simple Euler method
            batch_x_next.pos = batch_x.pos + 0.5 * scores_dict["v"] * dt
            batch_x_next.lengths = batch_x.lengths + 0.5 * scores_dict["l"] * dt
            batch_x_next.angles = batch_x.angles + 0.5 * scores_dict["l"] * dt

        elif solver == "heun":
            # Heun's method (2nd order)
            k1_v = scores_dict["v"]
            batch_x_next.pos = batch_x.pos + 0.5 * k1_v * dt
            # Would need second evaluation here

        # Add stochastic term for SDE (optional)
        # batch_x_next += sqrt(dt) * noise

        return batch_x_next

    def _postprocess_structures(self, batch_x: Data | Batch) -> Data | Batch:
        """Post-process sampled structures to ensure validity."""
        # Clamp lattice parameters to valid range
        if hasattr(batch_x, "lengths"):
            batch_x.lengths = torch.clamp(batch_x.lengths, min=0.1)
        if hasattr(batch_x, "angles"):
            batch_x.angles = torch.clamp(batch_x.angles, min=1.0, max=179.0)

        # Normalize positions to unit cell or wrap
        # batch_x.pos = torch.remainder(batch_x.pos, 1.0)

        return batch_x

    def _sample_noise_prior(
        self, n_samples: int, device: torch.device
    ) -> Data | Batch:
        """Sample initial noise structures from prior."""
        # Create dummy batch structure
        from torch_geometric.data import Data, Batch

        samples = []
        for _ in range(n_samples):
            # Typical crystal: 8-32 atoms
            n_atoms = torch.randint(8, 32, (1,)).item()
            data = Data(
                pos=torch.randn(n_atoms, 3, device=device),
                h=torch.randint(0, 118, (n_atoms,), device=device),
                lengths=torch.abs(torch.randn(3, device=device)) + 3.0,
                angles=torch.abs(torch.randn(3, device=device)) * 60 + 60,
            )
            samples.append(data)

        return Batch.from_data_list(samples)

    def _match_dims_for_broadcast(
        self, param: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Match dimensions for broadcasting (e.g., time to positions)."""
        while param.dim() < target.dim():
            param = param.unsqueeze(-1)
        return param






# ============================================================================
# Training Loop Template
# ============================================================================

def train_kldm(
    model: ModelKLDM,
    datamodule,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    loss_weights: Optional[dict[str, float]] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None,
) -> dict[str, list[float]]:
    """
    Example training loop for KLDM.

    Args:
        model: KLDM model instance
        datamodule: DataModule with train/val dataloaders
        num_epochs: Number of training epochs
        learning_rate: Optimizer learning rate
        loss_weights: Loss component weights
        device: Device for training
        checkpoint_dir: Directory for saving checkpoints

    Returns:
        Training history with loss curves
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(datamodule.train_dataloader()):
            optimizer.zero_grad(set_to_none=True)

            # Single training step (Algorithms 1 & 2)
            losses = model.training_step(batch, batch_idx, loss_weights=loss_weights)
            loss_total = losses["loss_total"]

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss_total.item())

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx}: "
                    + " ".join(f"{k}={v:.4f}" for k, v in losses.items() if k != "loss_total")
                )

        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        if datamodule.val_dataloader() is not None:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in datamodule.val_dataloader():
                    losses = model.training_step(batch, loss_weights=loss_weights)
                    val_losses.append(losses["loss_total"].item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            history["val_loss"].append(avg_val_loss)
            logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        scheduler.step()

        # Checkpoint saving
        if checkpoint_dir and epoch % 10 == 0:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )

    return history


# ============================================================================
# Sampling / Generation Template
# ============================================================================

@torch.no_grad()
def generate_crystals(
    model: ModelKLDM,
    n_samples: int = 10,
    steps: int = 100,
    device: Optional[torch.device] = None,
) -> list[Data]:
    """
    Generate crystal structures using trained KLDM.

    Args:
        model: Trained KLDM model
        n_samples: Number of structures to generate
        steps: Number of reverse integration steps
        device: Device for generation

    Returns:
        List of generated crystal structure Data objects
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Sample from prior (noise)
    prior_batch = model.sample_prior(n_samples=n_samples, device=device)

    # Reverse process (Algorithms 3-4)
    generated_batch = model.sample(
        prior_batch,
        steps=steps,
        solver="euler",
        return_trajectory=False,
    )

    # Convert batch to list if needed
    if isinstance(generated_batch, Batch):
        structures = generated_batch.to_data_list()
    else:
        structures = [generated_batch]

    return structures


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ModelKLDM(device=device)
    data_root = Path("src/kldm/data/data/mp_20")


    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        print("Model construction succeeded; skipping data smoke test.")
        return

    task = DNGTask()
    train_dataset = task.fit_dataset(data_root, split="train")
    batch = Batch.from_data_list([train_dataset[0], train_dataset[1]]).to(device)
    print("batchxx", batch)

    targets = model.training_targets(batch)

    print("\nKLDM smoke test")
    print("=" * 80)
    print(f"Loaded dataset from: {data_root}")
    print(f"Batch pos shape: {tuple(batch.pos.shape)}")
    print(f"Batch h shape: {tuple(batch.h.shape)}")
    print(f"Target keys: {sorted(targets.keys())}")

    generated = generate_crystals(
        model,
        n_samples=2,
        steps=5,
        device=device,
    )
    print(f"Generated {len(generated)} prior samples")


if __name__ == "__main__":
    main()
