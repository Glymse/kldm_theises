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
        self.diffusion_v = TrivialisedDiffusionMomentum(eps=eps)
        self.diffusion_l = ContinuousVPDiffusion(eps=eps)
        self.diffusion_a = ContinuousVPDiffusion(eps=eps)

        self.eps = eps
        self.device = device or torch.device("cpu")

        # Task detection (set after first batch)
        self.task_type: Optional[str] = None  # "csp" or "dng"



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
            t: Time steps [n_graphs, 1]

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
