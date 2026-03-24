from __future__ import annotations

import torch
from torch import nn


class TrivialisedDiffusionMomentum(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    @staticmethod
    def _match_dims(param: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while param.dim() < target.dim():
            param = param.unsqueeze(-1)
        return param

    @staticmethod
    def wrap(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x, 1.0)

    # v_t | v_0 ~ N(exp(-t) v0, (1-exp(-2t)) I)
    def alpha_v(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-t)

    def sigma_v(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(1.0 - torch.exp(-2.0 * t), self.eps))

    # r_t | v_t, v_0 ~ N(mu_r, sigma_r^2 I), then wrapped onto torus
    def mu_r(self, t: torch.Tensor, vt: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        c = (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))
        c = self._match_dims(c, vt)
        return c * (vt + v0)

    def sigma_r(self, t: torch.Tensor) -> torch.Tensor:
        var = 2.0 * t + 8.0 / (torch.exp(t) + 1.0) - 4.0
        return torch.sqrt(torch.clamp_min(var, self.eps))

    def perturb(
        self,
        t: torch.Tensor,
        f0: torch.Tensor,
        v0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            f_t  : perturbed torus coordinates
            v_t  : perturbed velocity
            eps_v: Gaussian noise used for v_t
        """
        if v0 is None:
            v0 = torch.zeros_like(f0)

        eps_v = torch.randn_like(v0)
        eps_r = torch.randn_like(f0)

        alpha_v = self._match_dims(self.alpha_v(t), v0)
        sigma_v = self._match_dims(self.sigma_v(t), v0)
        vt = alpha_v * v0 + sigma_v * eps_v

        mu_r = self.mu_r(t, vt, v0)
        sigma_r = self._match_dims(self.sigma_r(t), f0)
        rt = mu_r + sigma_r * eps_r

        ft = self.wrap(f0 + rt)
        return ft, vt, eps_v

    def target_score(
        self,
        t: torch.Tensor,
        vt: torch.Tensor,
        eps_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian part of KLDM velocity score:
            ∇_{v_t} log N(v_t ; exp(-t)v0, (1-exp(-2t))I) = -eps_v / sigma_v
        """
        sigma_v = self._match_dims(self.sigma_v(t), vt)
        return -eps_v / sigma_v
