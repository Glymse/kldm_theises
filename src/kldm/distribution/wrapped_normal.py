from __future__ import annotations

import torch


def d_log_wrapped_normal(
    r_t: torch.Tensor,
    mu_r_t: torch.Tensor,
    sigma_r_t: torch.Tensor,
    K: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Wrapped-normal score with respect to the mean on the unit interval.

    We work in the unit-period chart, so the wrapped normal is written as

        WN_K(r_t | mu_r_t, sigma_r_t^2)
            = Σ_{k=-K}^{K}
              exp(-(r_t + k - mu_r_t)^2 / (2 sigma_r_t^2)).

    The score used by TDM is the derivative with respect to the mean:

        s_K(r_t; mu_r_t, sigma_r_t)
            = ∇_{mu_r_t} log WN_K(r_t | mu_r_t, sigma_r_t^2)

            = [Σ_k ((r_t + k - mu_r_t) / sigma_r_t^2) exp(...)]
              / [Σ_k exp(...)].


    """
    k = torch.arange(-K, K + 1, device=r_t.device, dtype=r_t.dtype)

    # Each wrapped image corresponds to shifting the unit-period coordinate by
    # an integer k in the covering space.
    r_plus_k_minus_mu = r_t.unsqueeze(-1) + k - mu_r_t.unsqueeze(-1)

    sigma2_r_t = sigma_r_t.square().clamp_min(eps)

    # Unnormalized wrapped-normal series terms. The Gaussian prefactor is the
    # same for every k and cancels in the final ratio.
    exp_term = torch.exp(
        -(r_plus_k_minus_mu.square()) / (2.0 * sigma2_r_t.unsqueeze(-1))
    )

    numerator = torch.sum(
        (r_plus_k_minus_mu / sigma2_r_t.unsqueeze(-1)) * exp_term,
        dim=-1,
    )
    denominator = torch.sum(exp_term, dim=-1).clamp_min(eps)

    return numerator / denominator
