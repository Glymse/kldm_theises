from __future__ import annotations

import torch


def interpolate_lambda_table(
    lambda_table: torch.Tensor,
    t01: torch.Tensor,
) -> torch.Tensor:
    """
    Linearly interpolate a precomputed λ(t) table on normalized time t01 in [0, 1].
    """
    num_bins = lambda_table.shape[0]
    scaled = torch.clamp(t01, 0.0, 1.0) * (num_bins - 1)
    idx_lo = torch.floor(scaled).long()
    idx_hi = torch.clamp(idx_lo + 1, max=num_bins - 1)
    frac = (scaled - idx_lo.to(scaled.dtype)).to(lambda_table.dtype)

    lambda_lo = lambda_table[idx_lo]
    lambda_hi = lambda_table[idx_hi]
    return lambda_lo + (lambda_hi - lambda_lo) * frac


@torch.no_grad()
def precompute_lambda_time_grid_from_loader(
    diffusion,
    loader,
    t01_grid: torch.Tensor,
    num_batches: int = 32,
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of

        λ(t) = 1 / E ||∇_τ log p_t|0(τ | 0)||^2

    for the velocity states in the unit-period TDM convention.

    We estimate the score norm from the same simplified target used in training
    on real train batches with their real `batch.batch` graph structure. This is
    the stripped-down wrapped-normal target without the KLDM prefactor, because
    the prefactor is reinserted only during score reconstruction. Because the
    tangent space is Euclidean here, the loss computation uses the usual squared
    norm in R^m.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive for lambda precomputation.")

    table_device = t01_grid.device if device is None else torch.device(device)
    lambda_values = []

    for t01 in t01_grid:
        batch_estimates = []

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break

            batch = batch.to(table_device)
            index = batch.batch
            f0 = batch.pos.to(device=table_device)
            num_nodes = int(index.shape[0])
            t_node = torch.full(
                (num_nodes,),
                float(t01.item()),
                device=table_device,
                dtype=f0.dtype,
            )

            _, v_t, _, _, r_t = diffusion.forward_sample(
                t=t_node,
                f0=f0,
                index=index,
            )
            target_v = diffusion.score_target(
                t=t_node,
                r_t=r_t,
                v_t=v_t,
                index=index,
            )

            sq_norm = target_v.reshape(num_nodes, -1).pow(2).sum(dim=1).mean()
            batch_estimates.append(sq_norm)

        if not batch_estimates:
            raise RuntimeError("Train loader is empty; cannot precompute lambda(t).")

        expected_sq_norm = torch.stack(batch_estimates).mean()
        lambda_values.append(1.0 / expected_sq_norm.clamp_min(diffusion.eps))

    lambda_table = torch.stack(lambda_values)
    lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    return lambda_table
