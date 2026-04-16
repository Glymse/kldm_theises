from __future__ import annotations

import math
from pathlib import Path
import os
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
    - facit-style internal angular scaling with scale_pos = 2π
    - fractional latents returned to the network in [-0.5, 0.5)
    - facit-style sigma_norm target normalization
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
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.time_scaling_T = 2.0
        # Facit-style internal angular chart; external/public latents are still
        # fractional and wrapped back into [-0.5, 0.5) before reaching the net.
        self.scale_pos = 2.0 * math.pi
        self.k_wn_score = int(k_wn_score)
        self.n_lambdas = int(n_lambdas)
        self.lambda_num_batches = int(lambda_num_batches)
        self.simplified_parameterization = True
        self.use_lambda_weighting = False

        self.register_buffer("_lambda_t01_grid", torch.linspace(1e-4, 1.0, self.n_lambdas))
        self.register_buffer("_lambda_v_table", torch.ones(self.n_lambdas))

        sigma_grid_t = torch.linspace(0.0, self.time_scaling_T, int(n_sigmas))
        sigma_values = self.wrapped_gaussian_sigma_r_t(sigma_grid_t)
        sigma_precompute_start = time.perf_counter()
        print(
            "TDMdev sigma_norm precompute start "
            f"n_sigmas={n_sigmas} k_wn_score={self.k_wn_score} scale_pos={self.scale_pos:.6f}",
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
        self.register_buffer("_sigma_norms", sigma_norm_values)

    @staticmethod
    def wrap_positions(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x, 1.0)

    @staticmethod
    def wrap_displacements(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + 0.5, 1.0) - 0.5

    def _wrap_internal(self, x: torch.Tensor) -> torch.Tensor:
        half_period = 0.5 * self.scale_pos
        return torch.remainder(x + half_period, self.scale_pos) - half_period

    def gaussian_velocity_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-t)

    def gaussian_velocity_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def _prefactor_t(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))

    def wrapped_gaussian_mu_r_t_pre_wrap(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor,
        v0: torch.Tensor,
    ) -> torch.Tensor:
        coeff = self._prefactor_t(t)
        coeff = self._match_dims(coeff, v_t)
        return coeff * (v_t + v0)

    def wrapped_gaussian_mu_r_t(self, t: torch.Tensor, v_t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        return self.wrapped_gaussian_mu_r_t_pre_wrap(t=t, v_t=v_t, v0=v0)

    def wrapped_gaussian_sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.clamp(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0, min=self.eps)
        )

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

    @torch.no_grad()
    def sample_prior(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = len(index)
        pos_frac = self.wrap_displacements(torch.rand((num_nodes, 3), device=index.device))
        v_frac = scatter_center(torch.randn((num_nodes, 3), device=index.device), index=index) / self.scale_pos
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
        f0_internal = self.scale_pos * f0
        v0_internal = self.scale_pos * v0

        if epsilon_v is None:
            epsilon_v = torch.randn_like(v0_internal)
        epsilon_v = scatter_center(epsilon_v, index=index)

        mean_coeff_t = self._match_dims(self.gaussian_velocity_mean_coeff(t), v0_internal)
        sigma_v_t = self._match_dims(self.gaussian_velocity_sigma(t), v0_internal)
        v_t_internal = mean_coeff_t * v0_internal + sigma_v_t * epsilon_v

        mu_r_t_pre_wrap = self.wrapped_gaussian_mu_r_t_pre_wrap(t, v_t_internal, v0_internal)
        sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), f0_internal)

        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0_internal)
        epsilon_r = scatter_center(epsilon_r, index=index)

        r_t_internal = self._wrap_internal(mu_r_t_pre_wrap + sigma_r_t * epsilon_r)
        f_t_internal = self._wrap_internal(f0_internal + r_t_internal)

        r_t = r_t_internal / self.scale_pos
        f_t = f_t_internal / self.scale_pos
        v_t = v_t_internal / self.scale_pos

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
        r_t_internal = self.scale_pos * r_t
        v_t_internal = self.scale_pos * v_t
        v0_internal = self.scale_pos * v0

        mu_r_t_pre_wrap = self.wrapped_gaussian_mu_r_t_pre_wrap(t, v_t_internal, v0_internal)
        mu_r_t = self._wrap_internal(mu_r_t_pre_wrap)
        sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), r_t_internal)

        prefactor_t = self._match_dims(self._prefactor_t(t), r_t_internal)
        target_pos_t = prefactor_t * d_log_wrapped_normal(
            r=r_t_internal,
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
        v_t_internal = self.scale_pos * v_t
        prefactor = self._match_dims(self._prefactor_t(t_internal), pred_v)
        sigma_norm_t = self._match_dims(
            torch.sqrt(self._sigma_norm_t(t_internal)).clamp_min(self.eps),
            pred_v,
        )
        gaussian_velocity_sigma_sq = self._match_dims(
            self.gaussian_velocity_sigma(t_internal) ** 2,
            pred_v,
        ).clamp_min(self.eps)

        return prefactor * sigma_norm_t * pred_v - v_t_internal / gaussian_velocity_sigma_sq

    def reverse_exp_step(
        self,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        score_v: torch.Tensor,
        index: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_t = self.wrap_displacements(f_t)
        f_t_internal = self.scale_pos * f_t
        v_t_internal = self.scale_pos * v_t

        dt_t = torch.as_tensor(
            self.time_scaling_T * dt,
            device=v_t.device,
            dtype=v_t.dtype,
        )
        noise_v = scatter_center(torch.randn_like(v_t_internal), index=index)

        exp_dt = torch.exp(dt_t)
        expm1_dt = torch.expm1(dt_t)
        expm1_2dt = torch.expm1(2.0 * dt_t)
        noise_scale = torch.sqrt(expm1_2dt.clamp_min(self.eps))

        v_prev_internal = exp_dt * v_t_internal + 2.0 * expm1_dt * score_v + noise_scale * noise_v
        f_prev_internal = self._wrap_internal(f_t_internal - dt_t * v_prev_internal)

        return f_prev_internal / self.scale_pos, v_prev_internal / self.scale_pos

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
