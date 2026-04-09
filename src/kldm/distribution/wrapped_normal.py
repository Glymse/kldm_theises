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

    return ((diff) / sigma2 * w).sum(dim=-1)



def d_log_wrapped_normal_2pi_version(
    r: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    N: int = 13,
    T: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    while sigma.ndim < r.ndim:
        sigma = sigma.unsqueeze(-1)
    while mu.ndim < r.ndim:
        mu = mu.unsqueeze(-1)

    total = torch.zeros_like(r)
    denom = torch.zeros_like(r)
    sigma2 = sigma.square().clamp_min(eps)

    for i in range(-N, N + 1):
        shifted = r - mu + T * i
        weight = torch.exp(-(shifted.square()) / (2.0 * sigma2))
        total = total + (shifted / sigma2) * weight
        denom = denom + weight

    return total / denom.clamp_min(eps)


def main() -> None:
    torch.manual_seed(7)

    r = torch.rand(8, 3) - 0.5
    mu = torch.rand(8, 3) - 0.5
    sigma = 0.05 + 0.95 * torch.rand(8, 1)
    K = 13

    ours = d_log_wrapped_normal(r=r, mu=mu, sigma=sigma, K=K)
    two_pi_version = d_log_wrapped_normal_2pi_version(r=r, mu=mu, sigma=sigma, N=K, T=1.0)

    max_abs_diff = float((ours - two_pi_version).abs().max().item())
    max_abs_diff_negated = float((ours + two_pi_version).abs().max().item())
    same_value = torch.allclose(ours, two_pi_version, atol=1e-6, rtol=1e-5)
    same_value_if_negated = torch.allclose(ours, -two_pi_version, atol=1e-6, rtol=1e-5)

    print("Wrapped normal comparison")
    print("K/N:", K)
    print("T:", 1.0)
    print("ours[0]:", ours[0])
    print("2pi_version[0]:", two_pi_version[0])
    print("max_abs_diff:", max_abs_diff)
    print("allclose_same_value:", same_value)
    print("max_abs_diff_if_negated:", max_abs_diff_negated)
    print("allclose_if_negated:", same_value_if_negated)


if __name__ == "__main__":
    main()
