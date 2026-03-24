from __future__ import annotations

import torch
from torch import nn


class ContinuousVPDiffusion(nn.Module):
    """Small variance-preserving diffusion for Euclidean modalities.

    Used for:
    - lattice parameters `l`
    - atom representations `a` in the continuous DNG variant
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _match_dims(param: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while param.dim() < target.dim():
            param = param.unsqueeze(-1)
        return param

    #If we let Beta(t) = 2, we get:

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-2.0 * t))


    #Transition kernel explicit formula
    # xt | x0 ~ N(exp(-t) * x0,   (1-Exp(-2t)I)
    def transition_kernel_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_t = self._match_dims(self.alpha(t), x0)
        sigma_t = self._match_dims(self.sigma(t), x0)
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, noise

    #Score of gaussian kernel
    #nabla_xt logP(xt|x0) = - (xt-alphat*x0) / sigma^2 =.    pred / scale
    def score_target(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = self._match_dims(self.sigma(t), x_t)
        return -eps / sigma_t
