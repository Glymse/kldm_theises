import torch

def d_log_wrapped_normal(
    r: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    K: int = 13,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes

        ∇_mu log WN(r | mu, sigma^2)

    for a wrapped normal on [0,1), using the truncated sum
    k = -K, ..., K.

    Formula:
        [sum_k (1/sigma^2) * (r - mu + k) * exp(-(r-mu+k)^2 / (2 sigma^2))]
        -------------------------------------------------------------------
                  [sum_k exp(-(r-mu+k)^2 / (2 sigma^2))]
    """

    while sigma.ndim < r.ndim:
        sigma = sigma.unsqueeze(-1)

    k = torch.arange(-K, K + 1, device=r.device, dtype=r.dtype)

    # unit-period wrapped difference
    diff = r.unsqueeze(-1) - mu.unsqueeze(-1) + k

    sigma2 = sigma.unsqueeze(-1) ** 2

    exp_term = torch.exp(-diff.pow(2) / (2.0 * sigma2 + eps))

    upper = ((diff / (sigma2 + eps)) * exp_term).sum(dim=-1)
    lower = exp_term.sum(dim=-1) + eps

    return upper / lower
