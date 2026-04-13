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
    clamp_min: float = 0.05,
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
    num_bins = int(t01_grid.shape[0])
    sq_norm_sums = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)
    counts = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        batch = batch.to(table_device)
        index = batch.batch
        f0 = batch.pos.to(device=table_device)
        num_nodes = int(index.shape[0])
        num_graphs = int(batch.num_graphs)

        # Sample grid times per graph so one real batch contributes to many bins.
        graph_bin_idx = torch.randint(0, num_bins, (num_graphs,), device=table_device)
        t_graph = t01_grid[graph_bin_idx]
        t_node = t_graph[index]

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

        # Match the per-node reduction used by mse_loss_per_sample(...).
        node_sq_norm = target_v.reshape(num_nodes, -1).pow(2).mean(dim=1)
        graph_sq_sums = torch.zeros(num_graphs, device=table_device, dtype=node_sq_norm.dtype)
        graph_counts = torch.zeros(num_graphs, device=table_device, dtype=node_sq_norm.dtype)
        graph_sq_sums.scatter_add_(0, index, node_sq_norm)
        graph_counts.scatter_add_(0, index, torch.ones_like(node_sq_norm))
        graph_mean_sq_norm = graph_sq_sums / graph_counts.clamp_min(1.0)

        sq_norm_sums.scatter_add_(0, graph_bin_idx, graph_mean_sq_norm)
        counts.scatter_add_(0, graph_bin_idx, torch.ones_like(graph_mean_sq_norm))

    if not torch.any(counts > 0):
        raise RuntimeError("Train loader is empty; cannot precompute lambda(t).")

    expected_sq_norm = sq_norm_sums / counts.clamp_min(1.0)
    missing = counts <= 0
    if torch.any(missing):
        valid_idx = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
        all_idx = torch.arange(num_bins, device=table_device)
        nearest_valid = valid_idx[
            torch.argmin(
                torch.abs(all_idx[:, None] - valid_idx[None, :]),
                dim=1,
            )
        ]
        expected_sq_norm = expected_sq_norm.clone()
        expected_sq_norm[missing] = expected_sq_norm[nearest_valid[missing]]

    lambda_table = 1.0 / expected_sq_norm.clamp_min(diffusion.eps)
    lambda_table = lambda_table / lambda_table.mean().clamp_min(diffusion.eps)
    lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    return lambda_table
