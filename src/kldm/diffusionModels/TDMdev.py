from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.distribution.wrapped_normal import d_log_wrapped_normal, sigma_norm
from kldm.scoreNetwork.utils import scatter_center


class TrivialisedDiffusionDev(nn.Module):
    """
    Dev TDM variant:
    - native fractional chart with positions in [0,1) and displacements in [-0.5, 0.5)
    - epsilon-scaled unit-period diffusion using a 1 / (2π) noise scale
    - sigma_norm target normalization on the simplified wrapped-normal target
    - simplified parameterization only
    - no lambda(t) weighting in the loss
    """

    def __init__(
        self,
        eps: float = 1e-5,
        n_lambdas: int = 256,
        lambda_num_batches: int = 16,
        k_wn_score: int = 13,
        n_sigmas: int = 2000,
        compute_sigma_norm: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.time_scaling_T = 2.0
        self.scale_pos = 1.0
        self.vel_scale = 1.0 / (2.0 * math.pi)
        self.k_wn_score = int(k_wn_score)
        self.n_lambdas = int(n_lambdas)
        self.lambda_num_batches = int(lambda_num_batches)
        self.compute_sigma_norm = bool(compute_sigma_norm)
        self.simplified_parameterization = True
        self.use_lambda_weighting = False

        self.register_buffer("_lambda_t01_grid", torch.linspace(1e-4, 1.0, self.n_lambdas))
        self.register_buffer("_lambda_v_table", torch.ones(self.n_lambdas))

        if self.compute_sigma_norm:
            sigma_grid_t = torch.linspace(0.0, self.time_scaling_T, int(n_sigmas))
            sigma_values = self.wrapped_gaussian_sigma_r_t(sigma_grid_t)
            sigma_precompute_start = time.perf_counter()
            print(
                "TDMdev sigma_norm precompute start "
                f"n_sigmas={n_sigmas} k_wn_score={self.k_wn_score} "
                f"scale_pos={self.scale_pos:.6f} vel_scale={self.vel_scale:.6f}",
                flush=True,
            )
            sigma_norm_values = sigma_norm(
                sigma=sigma_values,
                T=self.scale_pos,
                K=self.k_wn_score,
                eps=self.eps,
            )
            print(
                "TDMdev sigma_norm precompute done "
                f"seconds={time.perf_counter() - sigma_precompute_start:.1f}",
                flush=True,
            )
        else:
            sigma_norm_values = torch.ones(int(n_sigmas), dtype=torch.get_default_dtype())
            print(
                "TDMdev sigma_norm precompute skipped "
                f"n_sigmas={n_sigmas} compute_sigma_norm={self.compute_sigma_norm}",
                flush=True,
            )
        self.register_buffer("_sigma_norms", sigma_norm_values)

    @staticmethod
    def wrap_positions(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x, 1.0)

    @staticmethod
    def wrap_displacements(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _wrap_internal(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrap_displacements(x)

    def gaussian_velocity_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-t)

    def gaussian_velocity_sigma_base(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def gaussian_velocity_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.vel_scale * self.gaussian_velocity_sigma_base(t)

    def _prefactor_t(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))

    def wrapped_gaussian_mu_r_t(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor,
        v0: torch.Tensor,
    ) -> torch.Tensor:
        coeff = self._prefactor_t(t)
        coeff = self._match_dims(coeff, v_t)
        return coeff * (v_t + v0)

    def wrapped_gaussian_sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        base_var = 2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0
        return self.vel_scale * torch.sqrt(torch.clamp(base_var, min=self.eps))

    def lambda_v(self, t01: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t01, dtype=self._lambda_v_table.dtype, device=t01.device)

    @torch.no_grad()
    def precompute_lambda_v_table_from_loader(
        self,
        loader,
        device: torch.device | None = None,
        num_batches: int | None = None,
        clamp_min: float = 0.2,
        clamp_max: float = 5.0,
        smooth: bool = True,
    ) -> torch.Tensor:
        del loader, device, num_batches, clamp_min, clamp_max, smooth
        self._lambda_v_table.fill_(1.0)
        return self._lambda_v_table

    def _sigma_norm_t(self, t: torch.Tensor) -> torch.Tensor:
        # Match facitKLDM's nearest-neighbor table lookup as closely as possible
        # while still clamping safely at the endpoints.
        idx = torch.round(t / self.time_scaling_T * len(self._sigma_norms)).long() - 1
        idx = idx.clamp(0, len(self._sigma_norms) - 1)
        return self._sigma_norms[idx]

    def sample_velocity_epsilon(self, ref: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(ref)
        eps = scatter_center(eps, index=index)
        return self.vel_scale * eps

    @torch.no_grad()
    def sample_prior(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = len(index)
        pos_frac = self.wrap_displacements(torch.rand((num_nodes, 3), device=index.device))
        v_frac = self.sample_velocity_epsilon(pos_frac, index=index)
        return pos_frac, v_frac

    def forward_sample(
        self,
        t: torch.Tensor,
        f0: torch.Tensor,
        index: torch.Tensor,
        v0: torch.Tensor | None = None,
        epsilon_v: torch.Tensor | None = None,
        epsilon_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.time_scaling_T * t

        if v0 is None:
            v0 = torch.zeros_like(f0)
        f0 = self.wrap_displacements(f0)

        if epsilon_v is None:
            epsilon_v = self.sample_velocity_epsilon(v0, index=index)
        else:
            epsilon_v = self.vel_scale * scatter_center(epsilon_v.to(dtype=v0.dtype, device=v0.device), index=index)

        mean_coeff_t = self._match_dims(self.gaussian_velocity_mean_coeff(t), v0)
        sigma_v_base_t = self._match_dims(self.gaussian_velocity_sigma_base(t), v0)
        v_t = mean_coeff_t * v0 + sigma_v_base_t * epsilon_v

        mu_r_t = self.wrapped_gaussian_mu_r_t(t, v_t, v0)
        sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), f0)

        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)
        else:
            epsilon_r = epsilon_r.to(dtype=f0.dtype, device=f0.device)
        epsilon_r = scatter_center(epsilon_r, index=index)

        r_t = self._wrap_internal(mu_r_t + sigma_r_t * epsilon_r)
        f_t = self._wrap_internal(f0 + r_t)



        f_t = scatter_center(f_t, index=index)



        return f_t, v_t, epsilon_v, epsilon_r, r_t


    def score_target(
        self,
        t: torch.Tensor,
        r_t: torch.Tensor,
        v_t: torch.Tensor,
        index: torch.Tensor,
        v0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.time_scaling_T * t
        v0 = torch.zeros_like(v_t) if v0 is None else v0

        mu_r_t = self._wrap_internal(self.wrapped_gaussian_mu_r_t(t, v_t, v0))
        sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), r_t)

        prefactor_t = self._match_dims(self._prefactor_t(t), r_t)
        target_pos_t = prefactor_t * d_log_wrapped_normal(
            r=r_t,
            mu=mu_r_t,
            sigma=sigma_r_t,
            K=self.k_wn_score,
            T=self.scale_pos,
            eps=self.eps,
        )
        target_pos_t = scatter_center(target_pos_t, index=index)

        sigma_norm_t = self._match_dims(torch.sqrt(self._sigma_norm_t(t)).clamp_min(self.eps), target_pos_t)
        return target_pos_t / prefactor_t.clamp_min(self.eps) / sigma_norm_t

    def construct_velocity_score(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
    ) -> torch.Tensor:
        t_internal = self.time_scaling_T * t
        prefactor = self._match_dims(self._prefactor_t(t_internal), pred_v)
        sigma_norm_t = self._match_dims(
            torch.sqrt(self._sigma_norm_t(t_internal)).clamp_min(self.eps),
            pred_v,
        )
        gaussian_velocity_sigma_sq = self._match_dims(
            self.gaussian_velocity_sigma(t_internal) ** 2,
            pred_v,
        ).clamp_min(self.eps)

        return prefactor * sigma_norm_t * pred_v - v_t / gaussian_velocity_sigma_sq

    def reverse_exp_step(
        self,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        score_v: torch.Tensor,
        index: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_t = self.wrap_displacements(f_t)

        dt_t = torch.as_tensor(self.time_scaling_T * dt, device=v_t.device, dtype=v_t.dtype)
        noise_v = self.sample_velocity_epsilon(v_t, index=index)

        exp_dt = torch.exp(dt_t)
        expm1_dt = torch.expm1(dt_t)
        expm1_2dt = torch.expm1(2.0 * dt_t)

        score_scale = torch.as_tensor(
            self.vel_scale ** 2,
            device=v_t.device,
            dtype=v_t.dtype,
        )

        # For the epsilon-scaled unit-chart model the forward OU diffusion
        # coefficient is reduced by `vel_scale`, so the reverse score drift must
        # carry the matching `vel_scale**2` factor too. Without it, sampling runs
        # far hotter than the trained forward process even when loss looks good.

        v_prev = (
            exp_dt * v_t
            + 2.0 * expm1_dt * score_v
            + torch.sqrt(expm1_2dt.clamp_min(self.eps)) * noise_v
        )

        f_prev = self._wrap_internal(f_t - dt_t * v_prev)

        return f_prev, v_prev

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
