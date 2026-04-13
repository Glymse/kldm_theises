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
def precompute_lambda_time_grid(
    diffusion,
    t01_grid: torch.Tensor,
    num_batches: int = 32,
    graphs_per_batch: int = 16,
    nodes_per_graph: int = 16,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """
    Monte Carlo estimate of

        λ(t) = 1 / E ||∇_τ log p_t|0(τ | 0)||^2

    for the velocity states in the unit-period TDM convention.

    We estimate the score norm from the same simplified target used in training:
    sample from the forward kernel with zero initial positions / velocities, then
    call diffusion.score_target(...). Because the tangent space is Euclidean here,
    the loss computation uses the usual squared norm in R^m.
    """
    device = t01_grid.device
    dtype = t01_grid.dtype
    lambda_values = []

    for t01 in t01_grid:
        batch_estimates = []

        for _ in range(num_batches):
            index = torch.arange(graphs_per_batch, device=device).repeat_interleave(nodes_per_graph)
            num_nodes = int(index.shape[0])

            # Zero initial state for the Monte Carlo approximation in p_t|0(. | 0).
            f0 = torch.zeros((num_nodes, 3), device=device, dtype=dtype)
            t_node = torch.full((num_nodes,), float(t01.item()), device=device, dtype=dtype)

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

        expected_sq_norm = torch.stack(batch_estimates).mean()
        lambda_values.append(1.0 / expected_sq_norm.clamp_min(diffusion.eps))

    lambda_table = torch.stack(lambda_values)
    lambda_table = lambda_table / lambda_table.mean().clamp_min(diffusion.eps)
    lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    return lambda_table
