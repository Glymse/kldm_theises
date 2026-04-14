from __future__ import annotations

import torch
import torch.nn.functional as F


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
    clamp_min: float = 0.2,
    clamp_max: float = 5.0,
    smooth: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of λ(t) for the exact wrapped-normal target used in training:

        λ(t) = 1 / E[ mean( target(t)^2 ) + eps ]

    where `target(t)` is the return value of `diffusion.score_target(...)`.

    This is intentionally based on the stripped wrapped-normal training target,
    because that is the quantity the network is regressed against.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive for lambda precomputation.")

    table_device = t01_grid.device if device is None else torch.device(device)
    num_bins = int(t01_grid.shape[0])

    sq_norm_sums = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)
    counts = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        batch = batch.to(table_device)
        index = batch.batch
        f0 = batch.pos.to(table_device)

        num_nodes = int(index.shape[0])
        num_graphs = int(batch.num_graphs)

        # One random time bin per graph, then broadcast it to that graph's nodes.
        graph_bin_idx = torch.randint(0, num_bins, (num_graphs,), device=table_device)
        node_bin_idx = graph_bin_idx[index]
        t_graph = t01_grid[graph_bin_idx]
        t_node = t_graph[index]

        _, v_t, _, _, r_t = diffusion.forward_sample(
            t=t_node,
            f0=f0,
            index=index,
        )

        target = diffusion.score_target(
            t=t_node,
            r_t=r_t,
            v_t=v_t,
            index=index,
        )

        # same reduction style as training loss
        node_sq = target.reshape(num_nodes, -1).pow(2).mean(dim=1)

        # Match the node-weighted averaging induced by the actual training loss.
        sq_norm_sums.scatter_add_(0, node_bin_idx, node_sq)
        counts.scatter_add_(0, node_bin_idx, torch.ones_like(node_sq))

    if not torch.any(counts > 0):
        raise RuntimeError("Train loader is empty; cannot precompute lambda(t).")

    expected_sq = sq_norm_sums / counts.clamp_min(1.0)

    # fill empty bins by nearest valid bin
    missing = counts <= 0
    if torch.any(missing):
        valid_idx = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
        all_idx = torch.arange(num_bins, device=table_device)
        nearest_valid = valid_idx[
            torch.argmin(torch.abs(all_idx[:, None] - valid_idx[None, :]), dim=1)
        ]
        expected_sq = expected_sq.clone()
        expected_sq[missing] = expected_sq[nearest_valid[missing]]

    # DSM-style inverse expected squared target norm
    lambda_table = 1.0 / expected_sq.clamp_min(diffusion.eps)

    # smooth in log-space for stability
    if smooth and num_bins >= 5:
        log_lambda = torch.log(lambda_table.clamp_min(diffusion.eps))
        kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=table_device, dtype=log_lambda.dtype)
        kernel = kernel / kernel.sum()
        x = F.pad(log_lambda[None, None, :], (2, 2), mode="replicate")
        log_lambda = F.conv1d(x, kernel[None, None, :]).squeeze(0).squeeze(0)
        lambda_table = torch.exp(log_lambda)

    # normalize around mean 1 for optimizer stability
    lambda_table = lambda_table / lambda_table.mean().clamp_min(diffusion.eps)

    # mild clipping only
    lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    return lambda_table
