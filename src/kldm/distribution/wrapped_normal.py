import os

import torch


#Based on the formula in Directional Statistics page 69 and the KLDM appendix.
#We might need mikkel or francois to verify this solution.

def d_log_wrapped_normal(
    r: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    K: int = 13,
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
    sigma_chunk_size: int = 128,
    sample_chunk_size: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    debug = os.environ.get("KLDM_SIGMA_NORM_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    sigma = sigma.reshape(-1)
    if sigma_chunk_size <= 0 or sample_chunk_size <= 0:
        raise ValueError("sigma_chunk_size and sample_chunk_size must be positive.")

    result = torch.empty_like(sigma)

    if debug:
        print(
            "sigma_norm_debug "
            f"sigmas={sigma.numel()} sn={sn} K={K} "
            f"sigma_chunk_size={sigma_chunk_size} sample_chunk_size={sample_chunk_size} "
            f"dtype={sigma.dtype} device={sigma.device}",
            flush=True,
        )

    with torch.no_grad():
        for sigma_start in range(0, sigma.numel(), sigma_chunk_size):
            sigma_end = min(sigma_start + sigma_chunk_size, sigma.numel())
            sigma_chunk = sigma[sigma_start:sigma_end].clamp_min(eps)
            accum = torch.zeros_like(sigma_chunk)
            seen = 0

            if debug:
                diff_elements = sample_chunk_size * sigma_chunk.numel() * (2 * K + 1)
                diff_mb = diff_elements * sigma.element_size() / (1024 ** 2)
                print(
                    "sigma_norm_debug "
                    f"chunk={sigma_start}:{sigma_end} "
                    f"chunk_sigmas={sigma_chunk.numel()} "
                    f"estimated_diff_tensor_mb={diff_mb:.1f}",
                    flush=True,
                )

            while seen < sn:
                current_samples = min(sample_chunk_size, sn - seen)
                sigmas = sigma_chunk.unsqueeze(0).expand(current_samples, -1)
                x_sample = sigmas * torch.randn_like(sigmas)
                x_sample = torch.remainder(x_sample + 0.5 * T, T) - 0.5 * T
                normal_ = d_log_wrapped_normal(
                    r=x_sample,
                    mu=torch.zeros_like(x_sample),
                    sigma=sigmas,
                    K=K,
                    T=T,
                    eps=eps,
                )
                accum += (normal_ ** 2).sum(dim=0)
                seen += current_samples

                if debug and (seen == current_samples or seen == sn or seen % max(sample_chunk_size * 4, 1) == 0):
                    print(
                        "sigma_norm_debug "
                        f"chunk={sigma_start}:{sigma_end} "
                        f"progress={seen}/{sn}",
                        flush=True,
                    )

            result[sigma_start:sigma_end] = accum / float(sn)

            if debug:
                print(
                    "sigma_norm_debug "
                    f"chunk={sigma_start}:{sigma_end} done",
                    flush=True,
                )

    if debug:
        print("sigma_norm_debug complete", flush=True)

    return result

#∇μ​logWN    not   ∇𝑟log⁡W.
#HUSK DETTE I RAPPORT
