import torch


#Based on the formula in Directional Statistics page 69 and the KLDM appendix.
#We might need mikkel or francois to verify this solution.

def d_log_wrapped_normal(
    r: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    K: int = 9,
    T: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    while sigma.ndim < r.ndim:
        sigma = sigma.unsqueeze(-1)
    while mu.ndim < r.ndim:
        mu = mu.unsqueeze(-1)

    k = torch.arange(-K, K + 1, device=r.device, dtype=r.dtype)
    diff = r.unsqueeze(-1) - mu.unsqueeze(-1) + T * k
    sigma2 = sigma.unsqueeze(-1).pow(2).clamp_min(eps)

    log_terms = -0.5 * diff.pow(2) / sigma2
    log_weights = log_terms - torch.logsumexp(log_terms, dim=-1, keepdim=True)
    weights = torch.exp(log_weights)

    grad_mu = (weights * diff / sigma2).sum(dim=-1)
    return grad_mu


def sigma_norm(
    sigma: torch.Tensor,
    T: float = 1.0,
    K: int = 9,
    sn: int = 20000,
    eps: float = 1e-8,
) -> torch.Tensor:
    sigma = sigma.reshape(-1)
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigmas * torch.randn_like(sigmas)
    x_sample = torch.remainder(x_sample + 0.5 * T, T) - 0.5 * T
    normal_ = d_log_wrapped_normal(
        r=x_sample,
        mu=torch.zeros_like(x_sample),
        sigma=sigmas.clamp_min(eps),
        K=K,
        T=T,
        eps=eps,
    )
    return (normal_ ** 2).mean(dim=0)

#∇μ​logWN    not   ∇𝑟log⁡W.
#HUSK DETTE I RAPPORT
