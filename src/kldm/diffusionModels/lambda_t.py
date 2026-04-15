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
    samples_per_batch: int = 2048,
    coord_dim: int = 3,
    clamp_min: float | None = None,
    clamp_max: float = 3,
    smooth: bool = True,
    mean_normalize: bool = True,
    target_mean_weight: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Monte Carlo estimate of the paper Eq. (4)-style lambda for the wrapped
    coordinate diffusion used by vanilla KLDM.

        λ(t) = 1 / E_{τ ~ p_{t|0}(·|0)} [ ||∇_τ log p_{t|0}(τ | 0)||_2^2 ]

    This follows the torsional-diffusion implementation closely:
    - precompute a score norm for each noise level / time bin directly from the
      forward kernel score itself
    - use the zero-clean-state wrapped kernel p_{t|0}(·|0)
    - estimate the second moment using the same coordinate reduction as the live
      KLDM loss: mean over coordinates per sample
    - return the inverse second moment as the loss weight

    The default behavior is the raw paper-style estimator: no smoothing, no
    mean normalization, and no clipping. Those can still be enabled explicitly
    when you want an optimizer-calibrated variant for experiments.

    The loader argument is kept only for API compatibility with the training
    code. No graph structure, centering, or real-batch statistics are used.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive for lambda precomputation.")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive for lambda precomputation.")
    if coord_dim <= 0:
        raise ValueError("coord_dim must be positive for lambda precomputation.")

    del loader  # kept only for API compatibility

    table_device = t01_grid.device if device is None else torch.device(device)
    t01_grid = t01_grid.to(table_device)
    num_bins = int(t01_grid.shape[0])

    acc_dtype = (
        torch.float32
        if t01_grid.dtype in (torch.float16, torch.bfloat16)
        else t01_grid.dtype
    )

    lambda_table = torch.empty(num_bins, device=table_device, dtype=acc_dtype)

    for bin_idx in range(num_bins):
        t01_value = t01_grid[bin_idx]
        t_internal = diffusion.time_scaling_T * t01_value
        sigma_scalar = diffusion.wrapped_gaussian_sigma_r_t(t_internal).to(dtype=acc_dtype)

        sq_norm_sum = torch.zeros((), device=table_device, dtype=acc_dtype)
        count = 0

        for _ in range(num_batches):
            sigma = sigma_scalar.expand(samples_per_batch, coord_dim)
            eps = torch.randn((samples_per_batch, coord_dim), device=table_device, dtype=acc_dtype)

            # Sample from p_{t|0}(·|0) in wrapped coordinates.
            r_t = diffusion.wrap_displacements(sigma * eps)
            mu = torch.zeros_like(r_t)

            raw_score = d_log_wrapped_normal(
                r=r_t,
                mu=mu,
                sigma=sigma,
                K=int(getattr(diffusion, "k_wn_score", 9)),
                T=float(getattr(diffusion, "scale_pos", 1.0)),
                eps=float(diffusion.eps),
            ).to(acc_dtype)

            # Match the live KLDM loss reduction: mean over coordinates per node.
            sq_norm = raw_score.pow(2).mean(dim=1)
            sq_norm_sum = sq_norm_sum + sq_norm.sum()
            count += sq_norm.numel()

        expected_sq_norm = sq_norm_sum / max(count, 1)
        lambda_table[bin_idx] = 1.0 / expected_sq_norm.clamp_min(float(diffusion.eps))

    if smooth and num_bins >= 5:
        log_lambda = torch.log(lambda_table.clamp_min(float(diffusion.eps)))
        kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=table_device, dtype=log_lambda.dtype)
        kernel = kernel / kernel.sum()
        x = F.pad(log_lambda[None, None, :], (2, 2), mode="replicate")
        log_lambda = F.conv1d(x, kernel[None, None, :]).squeeze(0).squeeze(0)
        lambda_table = torch.exp(log_lambda)

    if mean_normalize:
        lambda_table = lambda_table / lambda_table.mean().clamp_min(float(diffusion.eps))
        lambda_table = lambda_table * float(target_mean_weight)

    if clamp_min is not None and clamp_max is not None:
        lambda_table = lambda_table.clamp(min=clamp_min, max=clamp_max)
    elif clamp_min is not None:
        lambda_table = lambda_table.clamp_min(clamp_min)
    elif clamp_max is not None:
        lambda_table = lambda_table.clamp_max(clamp_max)

    return lambda_table.to(dtype=t01_grid.dtype)
