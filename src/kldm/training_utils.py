

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)



class DenoiseScoreMatchingLoss(nn.Module):
    """
    Denoising Score Matching (DSM) loss.

    L = E_t,x [ w(t) || s_θ(x_t, t) - ∇_{x_t} log p(x_t | x_0) ||^2 ]

    where w(t) is a time-dependent weight function.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        scores_pred: dict[str, torch.Tensor],
        scores_target: dict[str, torch.Tensor],
        t: torch.Tensor,
        sigmas: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute DSM loss.

        Args:
            scores_pred: Predicted scores {v, l, a}
            scores_target: Target scores {v, l, a}
            t: Time steps [B, 1]
            sigmas: Noise std devs {v, l, a} for SNR weighting

        Returns:
            Loss dictionary with components and total
        """
        losses = {}

        # Time weighting: w(t) = (1 - exp(-2t))  (for VP process)
        weight_t = (1.0 - torch.exp(-2.0 * t)) if self.config.use_time_weighting else 1.0

        # Velocity loss
        if "v" in scores_pred and "v" in scores_target:
            loss_v = F.mse_loss(scores_pred["v"], scores_target["v"], reduction="mean")
            loss_v = loss_v * weight_t.mean()
            losses["loss_v"] = loss_v * self.config.weight_v

        # Lattice loss
        if "l" in scores_pred and "l" in scores_target:
            loss_l = F.mse_loss(scores_pred["l"], scores_target["l"], reduction="mean")
            loss_l = loss_l * weight_t.mean()
            losses["loss_l"] = loss_l * self.config.weight_l

        # Atom type loss
        if "a" in scores_pred and "a" in scores_target:
            loss_a = F.mse_loss(scores_pred["a"], scores_target["a"], reduction="mean")
            loss_a = loss_a * weight_t.mean()
            losses["loss_a"] = loss_a * self.config.weight_a

        # Total loss
        losses["loss_total"] = sum(v for k, v in losses.items() if k.startswith("loss_"))

        return losses


class SNRWeightedScoreMatchingLoss(DenoiseScoreMatchingLoss):
    """
    Score Matching loss with SNR (Signal-to-Noise Ratio) weighting.

    Weights loss by SNR(t) to balance learning across all noise levels.
    Important for generative modeling with continuous diffusion.
    """

    def forward(
        self,
        scores_pred: dict[str, torch.Tensor],
        scores_target: dict[str, torch.Tensor],
        t: torch.Tensor,
        alphas: Optional[dict[str, torch.Tensor]] = None,
        sigmas: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SNR-weighted loss.

        w(t) = SNR(t) = (alpha(t) / sigma(t))^2
        """
        losses = {}

        if self.config.use_snr_weighting and alphas and sigmas:
            # SNR weighting for each component
            snr_v = (alphas["v"] / torch.clamp_min(sigmas["v"], self.config.eps)) ** 2
            snr_l = (alphas["l"] / torch.clamp_min(sigmas["l"], self.config.eps)) ** 2
            snr_a = (alphas["a"] / torch.clamp_min(sigmas["a"], self.config.eps)) ** 2

            # Clamp SNR
            snr_v = torch.clamp_min(snr_v, self.config.min_snr)
            snr_l = torch.clamp_min(snr_l, self.config.min_snr)
            snr_a = torch.clamp_min(snr_a, self.config.min_snr)

            # Velocity loss
            if "v" in scores_pred and "v" in scores_target:
                loss_v = F.mse_loss(scores_pred["v"], scores_target["v"], reduction="mean")
                loss_v = (loss_v * snr_v.mean()) / (1.0 + snr_v.mean())
                losses["loss_v"] = loss_v * self.config.weight_v

            # Lattice loss
            if "l" in scores_pred and "l" in scores_target:
                loss_l = F.mse_loss(scores_pred["l"], scores_target["l"], reduction="mean")
                loss_l = (loss_l * snr_l.mean()) / (1.0 + snr_l.mean())
                losses["loss_l"] = loss_l * self.config.weight_l

            # Atom type loss
            if "a" in scores_pred and "a" in scores_target:
                loss_a = F.mse_loss(scores_pred["a"], scores_target["a"], reduction="mean")
                loss_a = (loss_a * snr_a.mean()) / (1.0 + snr_a.mean())
                losses["loss_a"] = loss_a * self.config.weight_a
        else:
            # Fall back to parent class
            return super().forward(scores_pred, scores_target, t, sigmas)

        losses["loss_total"] = sum(v for k, v in losses.items() if k.startswith("loss_"))

        return losses


class CheckpointManager:
    """Manage model checkpoints and best model tracking."""

    def __init__(self, checkpoint_dir: Path, keep_best: int = 3, keep_last: int = 1):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            keep_best: Number of best checkpoints to keep
            keep_last: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best = keep_best
        self.keep_last = keep_last
        self.best_metrics = []  # List of (loss, path) tuples

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metric dictionary
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved best checkpoint: {path}")
            return path
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(checkpoint, path)

            # Clean up old checkpoints
            self._cleanup_checkpoints()

            return path

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))

        if len(checkpoint_files) > self.keep_last:
            for path in checkpoint_files[:-self.keep_last]:
                path.unlink()
                logger.info(f"Removed old checkpoint: {path}")

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load state

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state"])

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}: {checkpoint_path}")

        return epoch
