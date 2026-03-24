#!/usr/bin/env python
"""
KLDM Full Workflow Example - CSPVNet Integrated Version

This example shows how to:
1. Load data with proper batch format
2. Initialize KLDM with CSPVNet
3. Run training with Algorithms 1 & 2
4. Generate structures with Algorithms 3-4
5. Validate and post-process results
"""

import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

# KLDM imports
from kldm import (
    ModelKLDM,
    CheckpointManager,
    EarlyStoppingCallback,
    MetricsTracker,
    setup_logging,
    get_device,
)
from kldm import SamplingConfig, CrystalSampler, StructureValidator
from kldm.data import DNGTask

# Setup
logger = setup_logging("logs/", level=logging.INFO)
DEVICE = get_device("cuda")


def filter_valid_structures(batch, skip_filtering=False):
    """Filter out structures with invalid (NaN/Inf) lattice parameters."""
    if skip_filtering or not hasattr(batch, 'lengths') or not hasattr(batch, 'angles'):
        return batch

    # Just check for NaN/Inf, don't filter based on range
    # (data may be normalized)
    valid_mask = torch.isfinite(batch.lengths).all(dim=1) & torch.isfinite(batch.angles).all(dim=1)

    if not valid_mask.all():
        n_invalid = (~valid_mask).sum().item()
        logger.warning(f"Filtering out {n_invalid} structures with NaN/Inf values")

        # Get valid graph indices
        valid_graphs = torch.where(valid_mask)[0]
        if len(valid_graphs) == 0:
            logger.warning(f"ALL structures have NaN/Inf values!")
            return None

        # Filter batch attributes
        filtered_data = []

        for i, data in enumerate(batch.to_data_list()):
            if valid_mask[i]:
                filtered_data.append(data)

        if len(filtered_data) == 0:
            return None

        new_batch = Batch.from_data_list(filtered_data)
        return new_batch

    return batch


def inspect_batch(batch):
    """Validate batch format for CSPVNet."""
    print("\nBatch Format Validation:")
    print(f"  pos: {batch.pos.shape} [n_atoms, 3]")
    print(f"  h: {batch.h.shape} dtype={batch.h.dtype}")

    if hasattr(batch, "v") and batch.v is not None:
        print(f"  v: {batch.v.shape} [n_atoms, 3]")
    else:
        print(f"  v: Not present (will use zeros)")

    print(f"  lengths: {batch.lengths.shape if hasattr(batch, 'lengths') else 'missing'}")
    if hasattr(batch, 'lengths'):
        print(f"    min={batch.lengths.min().item():.4f}, max={batch.lengths.max().item():.4f}")

    print(f"  angles: {batch.angles.shape if hasattr(batch, 'angles') else 'missing'}")
    if hasattr(batch, 'angles'):
        print(f"    min={batch.angles.min().item():.4f}, max={batch.angles.max().item():.4f}")

    edge_shape = "missing"
    if hasattr(batch, "edge_node_index") and batch.edge_node_index is not None:
        edge_shape = batch.edge_node_index.shape
    elif hasattr(batch, "edge_index") and batch.edge_index is not None:
        edge_shape = batch.edge_index.shape
    print(f"  edge_node_index: {edge_shape}")
    print(f"  batch (node indices): {batch.batch.shape if hasattr(batch, 'batch') else 'missing'}")
    print(f"  num_graphs: {batch.num_graphs}")

    # Validate constraints
    if batch.h.ndim == 1:
        assert batch.h.dtype in [torch.long, torch.int64], "1D h must be integer type"
        assert (batch.h >= 0).all() and (batch.h < 118).all(), "h must be in [0, 117]"
    elif batch.h.ndim == 2:
        assert batch.h.dtype.is_floating_point, "2D h must be floating-point one-hot"
        assert batch.h.shape[1] == 118, "One-hot h must have width 118"
        assert torch.isfinite(batch.h).all(), "One-hot h must be finite"
    else:
        raise AssertionError("h must be either [n_atoms] integers or [n_atoms, 118] one-hot")

    print("✓ Batch format valid for CSPVNet (lengths/angles will be filtered if invalid)")


def train_kldm_full(
    model: ModelKLDM,
    datamodule,
    config,
    checkpoint_dir: Path = Path("models/kldm_checkpoints/"),
):
    """
    Complete training loop for KLDM with CSPVNet integration.

    Args:
        model: KLDM model
        datamodule: Data module with train/val dataloaders
        config: Training configuration
        checkpoint_dir: Where to save checkpoints
    """
    model = model.to(DEVICE)

    # Optimizer & scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Checkpoint management
    ckpt_mgr = CheckpointManager(checkpoint_dir)
    early_stopping = EarlyStoppingCallback(patience=10)

    # Metrics
    tracker_train = MetricsTracker()
    tracker_val = MetricsTracker()

    loss_weights = config["loss_weights"]
    best_val_loss = float("inf")
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")

    logger.info("=" * 80)
    logger.info("KLDM Training with CSPVNet Integration")
    logger.info("=" * 80)

    # First batch inspection
    first_batch = next(iter(datamodule.train_dataloader()))
    inspect_batch(first_batch)

    # Track batches and filtering
    filtered_count = 0
    total_batches_seen = 0
    disable_filtering = False

    # Training loop
    for epoch in range(config["num_epochs"]):
        # ====== Training Phase ======
        model.train()

        for batch_idx, batch in enumerate(datamodule.train_dataloader()):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            total_batches_seen += 1

            # Filter out invalid structures BEFORE moving to device
            # But disable filtering if more than 50% of batches are empty after filtering
            batch_filtered = filter_valid_structures(batch, skip_filtering=disable_filtering)
            if batch_filtered is None:
                filtered_count += 1
                if total_batches_seen > 5 and filtered_count / total_batches_seen > 0.5:
                    logger.warning("Disabling filtering: >50% of batches are completely filtered out")
                    disable_filtering = True
                    batch_filtered = batch  # Use original batch
                else:
                    continue

            batch = batch_filtered

            # Then move to device
            batch = batch.to(DEVICE)

            # Forward pass (Algorithms 1 & 2)
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_step(batch, loss_weights=loss_weights)
            loss_total = losses["loss_total"]

            # Backward pass
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track
            tracker_train.update(**{k: v.detach() for k, v in losses.items()})

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx}: "
                    + " ".join(f"{k}={v:.4f}" for k, v in losses.items())
                )

        # Log epoch training summary
        avg_losses = tracker_train.get_all_averages()
        logger.info(f"Epoch {epoch} Train: " + " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))
        tracker_train.reset()

        # ====== Validation Phase ======
        if datamodule.val_dataloader() is not None:
            model.eval()

            with torch.no_grad():
                for batch_idx, batch in enumerate(datamodule.val_dataloader()):
                    if max_val_batches is not None and batch_idx >= max_val_batches:
                        break
                    # Filter out invalid structures BEFORE moving to device
                    batch_filtered = filter_valid_structures(batch, skip_filtering=disable_filtering)
                    if batch_filtered is None:
                        continue

                    batch = batch_filtered

                    # Then move to device
                    batch = batch.to(DEVICE)

                    losses = model.training_step(batch, loss_weights=loss_weights)
                    tracker_val.update(**{k: v.detach() for k, v in losses.items()})

            val_losses = tracker_val.get_all_averages()
            val_loss = val_losses.get("loss_total", 0.0)
            logger.info(f"Epoch {epoch} Valid: " + " ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))

            # Checkpoint saving
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)

            if (epoch + 1) % 10 == 0 or is_best:
                ckpt_mgr.save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_losses,
                    is_best=is_best,
                )

            # Early stopping
            early_stopping.on_epoch_end(epoch, {"val_loss": val_loss})
            if early_stopping.should_stop:
                logger.info("Early stopping triggered")
                break

            tracker_val.reset()

        scheduler.step()

    logger.info("Training complete!")
    return best_val_loss


def generate_and_validate(
    model: ModelKLDM,
    n_samples: int = 10,
    steps: int = 100,
):
    """
    Generate crystal structures and validate them.

    Uses Algorithm 3-4 (reverse diffusion).

    Args:
        model: Trained KLDM model
        n_samples: Number to generate
        steps: ODE integration steps
    """
    model = model.to(DEVICE)
    model.eval()

    logger.info("=" * 80)
    logger.info(f"Generating {n_samples} Crystal Structures")
    logger.info("=" * 80)

    # Sample configuration
    config = SamplingConfig(
        num_steps=steps,
        solver="euler",
        t_start=1.0,
        t_end=1e-3,
        add_noise=True,
        save_trajectory=False,
    )

    # Generate
    sampler = CrystalSampler(model, config, device=DEVICE)
    generated_batch = sampler.sample_from_prior(n_samples=n_samples)

    # Convert to list
    structures = generated_batch.to_data_list()

    # Validate
    valid_structures = []
    for i, struct in enumerate(structures):
        is_valid = StructureValidator.is_valid(struct)
        validity_str = "✓ Valid" if is_valid else "✗ Invalid"

        if is_valid:
            valid_structures.append(struct)
            stability = StructureValidator.compute_stability_score(struct)
            logger.info(f"Structure {i}: {validity_str} (stability={stability:.2f})")
        else:
            logger.warning(f"Structure {i}: {validity_str}")

    logger.info(f"Generated {len(valid_structures)}/{n_samples} valid structures")

    return valid_structures


def main():
    """Run full KLDM workflow."""

    # Configuration
    config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "num_epochs": 2,
        "loss_weights": {"v": 1.0, "l": 1.0, "a": 1.0},
        "max_train_batches": 2,
        "max_val_batches": 1,
        "train_subset_size": 8,
        "val_subset_size": 4,
    }

    # Data loading
    logger.info("Loading data...")
    task = DNGTask(species_mode="one_hot")
    datamodule = task.datamodule(
        train_path=Path("data/mp_20/train.pt"),
        val_path=Path("data/mp_20/val.pt"),
        batch_size=config["batch_size"],
    )

    train_dataset = datamodule.datasets["train"]
    train_size = min(config["train_subset_size"], len(train_dataset))
    train_dataset.samples = train_dataset.samples[:train_size]

    val_dataset = datamodule.datasets["val"]
    val_size = 0 if val_dataset is None else min(config["val_subset_size"], len(val_dataset))
    if val_dataset is not None:
        val_dataset.samples = val_dataset.samples[:val_size]
    logger.info(
        "Using quick demo subset: train=%s samples, val=%s samples, epochs=%s, max_train_batches=%s, max_val_batches=%s",
        train_size,
        val_size,
        config["num_epochs"],
        config["max_train_batches"],
        config["max_val_batches"],
    )

    # Model initialization
    logger.info("Initializing KLDM with CSPVNet...")
    model = ModelKLDM(device=DEVICE)

    # Training
    best_val_loss = train_kldm_full(
        model,
        datamodule,
        config,
        checkpoint_dir=Path("models/kldm_checkpoints/"),
    )

    # Generation
    structures = generate_and_validate(model, n_samples=2, steps=10)

    # Save results
    output_dir = Path("generated_structures/")
    output_dir.mkdir(exist_ok=True)
    for i, struct in enumerate(structures):
        torch.save(struct, output_dir / f"structure_{i:04d}.pt")

    logger.info(f"Saved {len(structures)} structures to {output_dir}")


if __name__ == "__main__":
    main()
