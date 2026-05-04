from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Inspired by Yang Song's VP-SDE notation, but intentionally kept simple
# to fit the KLDM appendix.
# The paramization trick is covered in our theises..


class ContinuousVPDiffusion(nn.Module):
    """Small VP diffusion helper for Euclidean KLDM modalities.

    Used for:
    - lattice parameters `l`
    - atom representations `a` in the continuous DNG variant

    We use the VP-SDE

        dx = f(x, t) dt + g(t) dW_t
        f(x, t) = -0.5 beta(t) x
        g(t) = sqrt(beta(t))

    with a linear beta schedule. Its closed-form forward kernel is

        x_t | x_0 ~ N(alpha(t) x_0, sigma(t)^2 I)
        x_t = alpha(t) x_0 + sigma(t) eps, eps ~ N(0, I)

    """

    def __init__(
        self,
        eps: float = 1e-5,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        parameterization: str = "eps",
    ) -> None:
        super().__init__()
        if parameterization not in {"eps", "x0"}:
            raise ValueError("parameterization must be either 'eps' or 'x0'.")

        self.eps = float(eps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.parameterization = parameterization

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear VP-SDE noise schedule beta(t)."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the forward kernel."""
        beta_integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.pow(2)
        return torch.exp(-0.5 * beta_integral)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the forward kernel."""
        alpha_t = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - alpha_t.pow(2), min=self.eps))

    def forward_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the transition kernel.

        Given x_0 and a time t, we draw

            x_t = alpha(t) x_0 + sigma(t) eps

        where eps ~ N(0, I).
        """

        noise = torch.randn_like(x0)

        alpha_t = self._match_dims(self.alpha(t), x0)
        sigma_t = self._match_dims(self.sigma(t), x0)
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, noise

    def training_target(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return the lattice target matching the configured parameterization."""
        if self.parameterization == "eps":
            return noise #Here we just return the noise applied to the x0 target
        return x0 #Here we return the actual target that has been applied noise to

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff

    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Paper-aligned lattice reverse step.

        The network output is interpreted according to `self.parameterization`:

        - eps head:
              score = -eps_theta / sigma(t)

        - x0 head:
              score = (alpha(t) x0_theta - x_t) / sigma(t)^2

        Then we apply reverse Euler-Maruyama for the VP-SDE:

            x_{t-dt} = x_t - [f(x_t,t) - g(t)^2 score] dt
                       + g(t) sqrt(dt) z

        where f(x,t) = -0.5 beta(t)x and g(t)^2 = beta(t). The lattice branch in
        KLDM Appendix Algorithm 3/4 uses this EM update, not a PC update.
        """
        dt_t = torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype)
        beta_t = self._match_dims(self.beta(t), x_t)
        sigma_t = self._match_dims(self.sigma(t), x_t)

        if self.parameterization == "eps":
            score_x = -pred / sigma_t.clamp_min(self.eps)
            #Here we construct the usual score
        else:
            #And here we construct the paramization from franocis theises
            #
            alpha_t = self._match_dims(self.alpha(t), x_t)
            score_x = (alpha_t * pred - x_t) / sigma_t.pow(2).clamp_min(self.eps)

        noise = torch.randn_like(x_t)
        forward_drift = -0.5 * beta_t * x_t
        reverse_drift = forward_drift - beta_t * score_x
        x_prev = x_t - reverse_drift * dt_t
        x_prev = x_prev + torch.sqrt(beta_t * dt_t) * noise
        return x_prev
