from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from .wrapped_normal import d_log_wrapped_normal


### RIGHT NOW IT IS USING MONTE CARLO ESTIMATION
### HOWEVER WE WILL LATEr BE IMPLEMENTING THE INTEGRAL VERSIONS OF THIS
### AS WE CAN FIND THE ANALYTIC SOLUTION.

def chunk_ranges(total_size: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield [start, end) chunks."""
    for start in range(0, total_size, chunk_size):
        yield start, min(start + chunk_size, total_size)


def wrap_signed_unit(x: torch.Tensor) -> torch.Tensor:
    """Wrap to [-0.5, 0.5), used to create variables living in WN(0,1)"""
    return torch.remainder(x + 0.5, 1.0) - 0.5


def sample_wrapped_zero_mean(
    sigma: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Sample r = wrap(sigma * eps)."""
    sigma = sigma.reshape(-1)

    # Shape: [num_samples, num_sigmas]
    noise = torch.randn(
        (num_samples, sigma.numel()),
        device=sigma.device,
        dtype=sigma.dtype,
    )

    # Each column uses its own sigma.
    return wrap_signed_unit(noise * sigma.unsqueeze(0))


def wrapped_normal_zero_mean_score(
    r: torch.Tensor,
    sigma: torch.Tensor,
    *,
    K: int,
    eps: float,
) -> torch.Tensor:
    """Compute score at mu = 0."""
    sigma = sigma.reshape(-1).clamp_min(eps)

    # Broadcast sigma over Monte Carlo samples without copying.
    sigma_batch = sigma.unsqueeze(0).expand_as(r)

    return d_log_wrapped_normal(
        r_t=r,
        mu_r_t=torch.zeros_like(r),
        sigma_r_t=sigma_batch,
        K=K,
        eps=eps,
    )


def estimate_score_square_for_sigma_batch(
    sigma: torch.Tensor,
    *,
    num_samples: int,
    K: int,
    eps: float,
) -> torch.Tensor:
    """Estimate E[s_sigma(r)^2] for one sigma batch."""
    sigma = sigma.reshape(-1).clamp_min(eps)

    # Draw r from WN(0, sigma^2) using wrapped Gaussian samples.
    r = sample_wrapped_zero_mean(
        sigma=sigma,
        num_samples=num_samples,
    )

    # Compute wrapped-normal score with respect to the mean.
    score = wrapped_normal_zero_mean_score(
        r=r,
        sigma=sigma,
        K=K,
        eps=eps,
    )

    # Average over Monte Carlo samples.
    return score.square().mean(dim=0)


@dataclass
class WrappedNormalSigmaNorm:
    """Monte Carlo table for E[s_sigma(r)^2]."""

    K: int = 3
    num_monte_carlo_samples: int = 20000
    sigma_batch_size: int = 128
    sample_batch_size: int = 2048
    eps: float = 1e-8

    def __call__(
        self,
        sigma_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate score norm over a sigma grid."""
        sigma_grid = sigma_grid.reshape(-1)
        score_norm_grid = torch.empty_like(sigma_grid)

        with torch.no_grad():
            # Outer chunks avoid too many sigma values at once.
            for sigma_start, sigma_end in chunk_ranges(
                sigma_grid.numel(),
                self.sigma_batch_size,
            ):
                sigma = sigma_grid[sigma_start:sigma_end].clamp_min(self.eps)

                score_square_sum = torch.zeros_like(sigma)
                num_used_samples = 0

                # Inner chunks avoid storing all MC samples at once.
                for sample_start, sample_end in chunk_ranges(
                    self.num_monte_carlo_samples,
                    self.sample_batch_size,
                ):
                    num_samples = sample_end - sample_start

                    # Mean over this MC chunk only.
                    score_square_mean = estimate_score_square_for_sigma_batch(
                        sigma=sigma,
                        num_samples=num_samples,
                        K=self.K,
                        eps=self.eps,
                    )

                    # Convert chunk mean back to weighted sum.
                    score_square_sum += score_square_mean * float(num_samples)
                    num_used_samples += num_samples

                score_norm_grid[sigma_start:sigma_end] = (
                    score_square_sum / float(num_used_samples)
                )

        return score_norm_grid
