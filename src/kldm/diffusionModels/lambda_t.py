from __future__ import annotations

import torch
import torch.nn.functional as F

from kldm.distribution import d_log_wrapped_normal


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
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    smooth: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of the Torsional Diffusion Eq. (4) weight:

        λ(t) = 1 / E_{τ ~ p_{t|0}(·|0)} [ ||∇_τ log p_{t|0}(τ | 0)||_2^2 ]

    Implementation details:
    - sample directly from the zero-centered wrapped-normal perturbation kernel
    - evaluate the raw wrapped-normal score directly
    - use squared Euclidean norm over the full 3D fractional coordinate vector

    The loader argument is kept only for API compatibility with the training code.
    No graph structure, centering, or real-batch statistics are used here.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive for lambda precomputation.")

    del loader

    table_device = t01_grid.device if device is None else torch.device(device)
    t01_grid = t01_grid.to(table_device)
    num_bins = int(t01_grid.shape[0])
    spatial_dim = 3
    samples_per_batch = 2048

    sq_norm_sums = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)
    counts = torch.zeros(num_bins, device=table_device, dtype=t01_grid.dtype)

    for _ in range(num_batches):
        bin_idx = torch.randint(0, num_bins, (samples_per_batch,), device=table_device)
        t_node = t01_grid[bin_idx]
        t_internal = diffusion.time_scaling_T * t_node
        sigma_scalar = diffusion.wrapped_gaussian_sigma_r_t(t_internal).to(dtype=t01_grid.dtype)
        sigma = sigma_scalar[:, None].expand(-1, spatial_dim)
        eps = torch.randn((samples_per_batch, spatial_dim), device=table_device, dtype=t01_grid.dtype)
        r_t = diffusion.wrap_displacements(sigma * eps)
        mu = torch.zeros_like(r_t)

        raw_score = d_log_wrapped_normal(
            r=r_t,
            mu=mu,
            sigma=sigma,
            K=int(getattr(diffusion, "k_wn_score", 9)),
            T=float(getattr(diffusion, "scale_pos", 1.0)),
            eps=diffusion.eps,
        )

        node_sq = raw_score.pow(2).sum(dim=1)
        sq_norm_sums.scatter_add_(0, bin_idx, node_sq)
        counts.scatter_add_(0, bin_idx, torch.ones_like(node_sq))

    expected_sq = sq_norm_sums / counts.clamp_min(1.0)

    missing = counts <= 0
    if torch.any(missing):
        valid_idx = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
        all_idx = torch.arange(num_bins, device=table_device)
        nearest_valid = valid_idx[
            torch.argmin(torch.abs(all_idx[:, None] - valid_idx[None, :]), dim=1)
        ]
        expected_sq = expected_sq.clone()
        expected_sq[missing] = expected_sq[nearest_valid[missing]]

    lambda_table = 1.0 / expected_sq.clamp_min(diffusion.eps)

    if smooth and num_bins >= 5:
        log_lambda = torch.log(lambda_table.clamp_min(diffusion.eps))
        kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=table_device, dtype=log_lambda.dtype)
        kernel = kernel / kernel.sum()
        x = F.pad(log_lambda[None, None, :], (2, 2), mode="replicate")
        log_lambda = F.conv1d(x, kernel[None, None, :]).squeeze(0).squeeze(0)
        lambda_table = torch.exp(log_lambda)

    if clamp_min is not None and clamp_max is not None:
        lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    elif clamp_min is not None:
        lambda_table = lambda_table.clamp_min(clamp_min)
    elif clamp_max is not None:
        lambda_table = lambda_table.clamp_max(clamp_max)

    return lambda_table
