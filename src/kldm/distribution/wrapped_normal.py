import torch


#Based on the formula in Directional Statistics page 69 and the KLDM appendix.
#We might need mikkel or francois to verify this solution.

def d_log_wrapped_normal(
    r: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    K: int = 13,
    eps: float = 1e-8,
) -> torch.Tensor:
    while sigma.ndim < r.ndim:
        sigma = sigma.unsqueeze(-1)
    while mu.ndim < r.ndim:
        mu = mu.unsqueeze(-1)

    k = torch.arange(-K, K + 1, device=r.device, dtype=r.dtype)
    diff = r.unsqueeze(-1) - mu.unsqueeze(-1) + k
    sigma2 = sigma.unsqueeze(-1).square().clamp_min(eps)

    logw = -0.5 * diff.square() / sigma2
    logw = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
    w = torch.exp(logw)

    return (-(diff) / sigma2 * w).sum(dim=-1)
