from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.distribution import d_log_wrapped_normal
from kldm.diffusionModels.lambda_t import (
    interpolate_lambda_table,
    precompute_lambda_time_grid_from_loader,
)
from kldm.scoreNetwork.utils import scatter_center

################################################################
####### NOTE:                                               ####
#######       Time is first mapped from normalized time     ####
#######       t01 in [0,1] to KLDM internal time by         ####
#######       t = tf * t01.        [Appendix T = 2]         ####
################################################################



class TrivialisedDiffusion(nn.Module):
    """
    trivialised diffusion for positions + velocities.
    """
    def __init__(
            self,
            eps: float = 1e-5,
            n_lambdas: int = 256,
            lambda_num_batches: int = 16) -> None:
        super().__init__()
        self.eps = float(eps)
        self.time_scaling_T = 2
        self.n_lambdas = int(n_lambdas)
        self.lambda_num_batches = int(lambda_num_batches)
        self.register_buffer("_lambda_t01_grid", torch.linspace(1e-4, 1.0, self.n_lambdas))
        self.register_buffer("_lambda_v_table", torch.ones(self.n_lambdas))

    # -------------------------------------------------
    #  Wrapping function.
    # -------------------------------------------------

    @staticmethod
    def wrap_positions(x: torch.Tensor) -> torch.Tensor:
        """Wrap unit-cell fractional coordinates into [0, 1)."""
        return torch.remainder(x, 1.0)

    #displacements usually should be in [-0.5,0.5), see report.
    #
    #Måske ikke nødvendigt.
    @staticmethod
    def wrap_displacements(x: torch.Tensor) -> torch.Tensor:
        """Wrap signed periodic displacements into [-0.5, 0.5)."""
        return torch.remainder(x + 0.5, 1.0) - 0.5

    # -------------------------------------------------
    # Velocity sampling
    # -------------------------------------------------

    # v_t | v_0 ~ N(exp(-t) v_0, (1 - exp(-2t)) I)
    def gaussian_velocity_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the Gaussian velocity forward kernel."""
        return torch.exp(-t)

    def gaussian_velocity_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the Gaussian velocity forward kernel."""
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def wrapped_gaussian_mu_r_t_pre_wrap(self, t: torch.Tensor, v_t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Unwrapped wrapped-Gaussian mean in the internal unit-period chart."""
        coeff = (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))  # Eq. (22)
        coeff = self._match_dims(coeff, v_t)
        return coeff * (v_t + v0)

    def wrapped_gaussian_mu_r_t(self, t: torch.Tensor, v_t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for the pre-wrap mean helper."""
        return self.wrapped_gaussian_mu_r_t_pre_wrap(t=t, v_t=v_t, v0=v0)

    def wrapped_gaussian_sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.clamp(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0, min=self.eps)
        )  # Eq. (23)

    def lambda_v(self, t01: torch.Tensor) -> torch.Tensor:
        """
        Interpolate λ(t) for normalized time t01 in [0,1].
        """
        return interpolate_lambda_table(self._lambda_v_table, t01)

    # NO GRAD HERE!!!, THIS IS PURELY PRECOMPUTED!!!
    @torch.no_grad()
    def precompute_lambda_v_table_from_loader(
        self,
        loader,
        device: torch.device | None = None,
        num_batches: int | None = None,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        smooth: bool = False,
    ) -> torch.Tensor:
        """
        Precompute λ(t) using the Torsional Diffusion Eq. (4) kernel-score norm.

        By default this is the raw paper-style estimate with no smoothing or clipping.
        """
        lambda_table = precompute_lambda_time_grid_from_loader(
            diffusion=self,
            loader=loader,
            t01_grid=self._lambda_t01_grid,
            num_batches=self.lambda_num_batches if num_batches is None else int(num_batches),
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            smooth=smooth,
            device=self._lambda_v_table.device if device is None else device,
        )
        self._lambda_v_table.copy_(lambda_table.to(self._lambda_v_table))
        return self._lambda_v_table


    #TODO: We do not center the distribution around = 0 yet. Ask francois.

    def forward_sample(
        self,
        t: torch.Tensor,
        f0: torch.Tensor,
        index: torch.Tensor,
        v0: torch.Tensor | None = None,
        epsilon_v: torch.Tensor | None = None,
        epsilon_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


        #Now we do T = [0,2] time scaling in order for TDM to converge.
        t = self.time_scaling_T * t

        """
        The transition kernel is defined as follow:
            p_t|0 (ft, vt | f0, v0) = WN(r | wrapped_gaussian_mu_r_t, wrapped_gaussian_sigma_r_t) * Nv(vt | mu_v_t, sigma_v)

            transition kernel =          sample r_t              *         sample v_t

            We sample v_t, use it to move f_0 on manifold, to samlpe f_t
        """

        #######################
        ###    SAMPLE v_t   ###
        #######################

        #Vi sætter v0 = 0, [Design choice] at time t = 0
        if v0 is None:
            v0 = torch.zeros_like(f0)                               #Design choice: Initial zero velocities
        # Keep all state variables in native unit-period coordinates.
        # Introducing angular 2π scaling here changes the optimization problem.
        f0 = self.wrap_displacements(f0)

        #TODO: Scatter center mean free, også det de gør i KLDM

        #Sample normal noise for velocity                       # Nv is a normal distribution such that ∑vi = 0
        if epsilon_v is None:
            epsilon_v = torch.randn_like(v0)
        epsilon_v = scatter_center(epsilon_v, index=index) #Zero mean

        gaussian_velocity_mean_coeff_t = self._match_dims(self.gaussian_velocity_mean_coeff(t), v0)
        gaussian_velocity_sigma_t = self._match_dims(self.gaussian_velocity_sigma(t), v0)

        #Sample v_t, given initial velocity.
        v_t = gaussian_velocity_mean_coeff_t * v0 + gaussian_velocity_sigma_t * epsilon_v

        ######################################
        ###    Calculate displacement ft   ###
        ######################################
        #Now we calculate f_t = f_0 * expm(r_t), where r_t follows a wrapped Gaussian.
        wrapped_gaussian_mu_r_t_pre_wrap = self.wrapped_gaussian_mu_r_t_pre_wrap(t, v_t, v0)

        wrapped_gaussian_sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), f0)

        #Sample normal noise on epsilon
        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)                        # Nr is a normal distribution such that ∑vi = 0
        epsilon_r = scatter_center(epsilon_r, index=index)


        # Native fractional pipeline: wrap in period 1 directly.
        # Hidden angular scaling here would change both target norms and reverse dynamics.
        r_t = self.wrap_displacements(wrapped_gaussian_mu_r_t_pre_wrap + wrapped_gaussian_sigma_r_t * epsilon_r)

        #Now we calculate displacement, and while we stay on the manifold.
        f_t = self.wrap_displacements(f0 + r_t)

        #Center again
        #f_t = scatter_center(f_t, index=index) DEACTIVED WHILE FRANCOIS ANSWERS MAIL.
        #Then we wrap to ensure we stay within [0,1]
        #f_t = self.wrap_positions(f_t)

        return f_t, v_t, epsilon_v, epsilon_r, r_t

    def score_target(
        self,
        t: torch.Tensor,
        # epsilon_v: torch.Tensor, not needed due to our initial velocity assumption
        r_t: torch.Tensor,
        v_t: torch.Tensor,
        index: torch.Tensor,
        v0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the simplified TDM target predicted by the network.

        In the simplified parameterization we train on the stripped-down
        wrapped-normal term only. The prefactor is reinserted later in
        `construct_velocity_score(...)` when reconstructing Eq. (19).
        """

        #We do time scaling.
        t = self.time_scaling_T * t

        #Design choice, makes the target quite simple to calculate.
        v0 = torch.zeros_like(v_t) if v0 is None else v0

        #Now we find target of the wrapped normal fractional distribution
        # Keep target construction in exactly the same unit-period chart as training.
        wrapped_gaussian_mu_r_t_pre_wrap = self.wrapped_gaussian_mu_r_t_pre_wrap(t, v_t, v0)

        # Keep the wrapped mean in the centered unit chart so the finite image truncation in
        # d_log_wrapped_normal(...) stays numerically stable. Verify this TODO
        wrapped_gaussian_mu_r_t = self.wrap_displacements(wrapped_gaussian_mu_r_t_pre_wrap)

        wrapped_gaussian_sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), r_t)



        wrapped_gaussian_target = d_log_wrapped_normal(
            r=r_t,
            mu=wrapped_gaussian_mu_r_t,
            sigma=wrapped_gaussian_sigma_r_t,
        )



        #Center the target
        wrapped_gaussian_target = scatter_center(wrapped_gaussian_target, index=index)

        #target = gaussian_velocity_target + wrapped_gaussian_target
        return wrapped_gaussian_target #Simplified version, might explain later.

        #return target_s

    def construct_velocity_score(
        self,
        t: torch.Tensor,
        v_t: torch.Tensor,
        pred_v: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the full KLDM velocity score from the simplified prediction."""
        t_internal = self.time_scaling_T * t
        prefactor = self._match_dims(
            (1.0 - torch.exp(-t_internal)) / (1.0 + torch.exp(-t_internal)),
            pred_v,
        )
        gaussian_velocity_sigma_sq = self._match_dims(
            self.gaussian_velocity_sigma(t_internal) ** 2,
            pred_v,
        ).clamp_min(self.eps)

        # Native period-1 reconstruction: no hidden chart conversion here.
        return prefactor * pred_v - v_t / gaussian_velocity_sigma_sq

    def reverse_exp_step(
        self,
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        score_v: torch.Tensor,
        index: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One exponential-integrator reverse step for the TDM process."""
        f_t = self.wrap_displacements(f_t)

        dt_t = torch.as_tensor(
            self.time_scaling_T * dt,
            device=v_t.device,
            dtype=v_t.dtype,
        )
        # Reverse noise must stay in the same unit-period scaling as v_t and score_v.
        noise_v = scatter_center(torch.randn_like(v_t), index=index)

        exp_dt = torch.exp(dt_t)
        exp_2dt_minus_1 = torch.exp(2.0 * dt_t) - 1.0
        expm1_dt = torch.expm1(dt_t)
        expm1_2dt = torch.expm1(2.0 * dt_t)
        noise_scale = torch.sqrt(expm1_2dt.clamp_min(self.eps))

        # Keep reverse drift native too; mixing angular-sized score/noise here is an easy way
        # to silently change the sampler.
        v_prev = exp_dt * v_t + 2.0 * expm1_dt * score_v + noise_scale * noise_v

        f_prev = self.wrap_displacements(f_t - dt_t * v_prev)

        return f_prev, v_prev


    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
