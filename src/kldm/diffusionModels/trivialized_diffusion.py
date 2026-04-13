from __future__ import annotations

from pathlib import Path
import math
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.distribution import d_log_wrapped_normal_2pi_version
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
            n_lambdas: int = 2000,
            lambda_num_samples: int = 20000,
            lambda_chunk_size: int = 128,
            k_wn_score: int = 13) -> None:
        super().__init__()
        self.eps = float(eps)
        self.time_scaling_T = 2
        self.scale_pos = 2.0 * math.pi
        self.k_wn_score = int(k_wn_score)
        self.n_lambdas = int(n_lambdas)
        self.lambda_num_samples = int(lambda_num_samples)
        self.lambda_chunk_size = int(lambda_chunk_size)
        lambda_grid = torch.linspace(1e-4, 1.0, self.n_lambdas)
        lambda_table = self._precompute_lambda_table(lambda_grid, num_samples=self.lambda_num_samples)
        self.register_buffer("_lambda_v_table", lambda_table)

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

    def wrap_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Wrap scaled periodic variables into [-pi, pi)."""
        return torch.remainder(x + torch.pi, self.scale_pos) - torch.pi

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

    def wrapped_gaussian_mu_r_t(self, t: torch.Tensor, v_t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        coeff = (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))  # Eq. (22)
        coeff = self._match_dims(coeff, v_t)
        return coeff * (v_t + v0)

    def wrapped_gaussian_sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.clamp(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0, min=self.eps)
        )  # Eq. (23)

    def _precompute_lambda_table(
        self,
        t01_grid: torch.Tensor,
        num_samples: int = 20000,
    ) -> torch.Tensor:
        """
        Estimate λ(t) = 1 / E || ∇_μ log WN ||^2 on a normalized time grid.

        This is computed for the raw simplified wrapped-normal target in the
        same 2π-periodic convention used by score_target().
        """
        sigma_grid = self.wrapped_gaussian_sigma_r_t(self.time_scaling_T * t01_grid)
        chunk_size = self.lambda_chunk_size
        total = torch.zeros_like(sigma_grid)
        count = 0

        for start in range(0, num_samples, chunk_size):
            current_chunk = min(chunk_size, num_samples - start)
            sigmas = sigma_grid[None, :].repeat(current_chunk, 1)
            x_sample = sigmas * torch.randn_like(sigmas)
            x_sample = self.wrap_angles(x_sample)
            normal = d_log_wrapped_normal_2pi_version(
                r=x_sample,
                mu=torch.zeros_like(x_sample),
                sigma=sigmas,
                N=self.k_wn_score,
                T=self.scale_pos,
            )
            total = total + normal.square().sum(dim=0)
            count += current_chunk

        expected_sq_norm = total / max(count, 1)
        return 1.0 / expected_sq_norm.clamp_min(self.eps)

    def lambda_v(self, t01: torch.Tensor) -> torch.Tensor:
        """
        Interpolate λ(t) for normalized time t01 in [0,1].
        """
        scaled = torch.clamp(t01, 0.0, 1.0) * (self.n_lambdas - 1)
        idx_lo = torch.floor(scaled).long()
        idx_hi = torch.clamp(idx_lo + 1, max=self.n_lambdas - 1)
        frac = (scaled - idx_lo.to(scaled.dtype)).to(self._lambda_v_table.dtype)
        lambda_lo = self._lambda_v_table[idx_lo]
        lambda_hi = self._lambda_v_table[idx_hi]
        return lambda_lo + (lambda_hi - lambda_lo) * frac


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
        v0 = self.scale_pos * v0

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
        f0 = self.scale_pos * self.wrap_displacements(f0)

        wrapped_gaussian_mu_r_t = self.wrapped_gaussian_mu_r_t(t, v_t, v0)
        wrapped_gaussian_mu_r_t = self.wrap_angles(wrapped_gaussian_mu_r_t) #To stay in [-0.5, 0.5] or [-pi, pi] equvialant.

        wrapped_gaussian_sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), f0)

        #Sample normal noise on epsilon
        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)                        # Nr is a normal distribution such that ∑vi = 0
        epsilon_r = scatter_center(epsilon_r, index=index)


        #FACIT VERISON, OLD VERSION, CHAT MIGHT SAY IT IS A PROBLEM
        r_t = self.wrap_angles(wrapped_gaussian_mu_r_t + wrapped_gaussian_sigma_r_t * epsilon_r)

        #Now we calculate displacement, and while we stay on the manifold.
        #We keep the internal TDM position latent in the centered unit-period chart.
        f_t = self.wrap_angles(f0 + r_t)

        #Center again
        #f_t = scatter_center(f_t, index=index) DEACTIVED WHILE FRANCOIS ANSWERS MAIL.
        #Then we wrap to ensure we stay within [0,1]
        #f_t = self.wrap_positions(f_t)

        return f_t / self.scale_pos, v_t / self.scale_pos, epsilon_v, epsilon_r, r_t

    def score_target(
        self,
        t: torch.Tensor,
        # epsilon_v: torch.Tensor, not needed due to our initial velocity assumption
        r_t: torch.Tensor,
        v_t: torch.Tensor,
        index: torch.Tensor,
        v0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the TDM velocity training target used by KLDM. """

        #We do time scaling.
        t = self.time_scaling_T * t

        #Design choice, makes the target quite simple to calculate.
        v0 = torch.zeros_like(v_t) if v0 is None else v0
        v0 = self.scale_pos * v0
        v_t = self.scale_pos * v_t

        #Now we find target of the wrapped normal fractional distribution
        wrapped_gaussian_mu_r_t = self.wrapped_gaussian_mu_r_t(t, v_t, v0)
        #NOT WRITTEN IN APPENDIX, BUT ASK FRANCOIS / MIKKEL IF IT WOULD MAKE SENSE TO DO
        #But would make sense since usually it works i [-pi, pi]
        """TODO: CHECK"""
        wrapped_gaussian_mu_r_t = self.wrap_angles(wrapped_gaussian_mu_r_t)
        """ this """

        wrapped_gaussian_sigma_r_t = self._match_dims(self.wrapped_gaussian_sigma_r_t(t), r_t)


        """
        wrapped_gaussian_target = self._match_dims((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t)), r_t) * d_log_wrapped_normal(
            r=r_t,
            mu=wrapped_gaussian_mu_r_t,
            sigma=wrapped_gaussian_sigma_r_t,
        )
        """ #FULL TARGET

        wrapped_gaussian_target = d_log_wrapped_normal_2pi_version(
            r=r_t,
            mu=wrapped_gaussian_mu_r_t,
            sigma=wrapped_gaussian_sigma_r_t,
            N=self.k_wn_score,
            T=self.scale_pos,
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
        """Construct the full KLDM velocity score from the network prediction."""
        t_internal = self.time_scaling_T * t
        v_t = self.scale_pos * v_t
        prefactor = self._match_dims(
            (1.0 - torch.exp(-t_internal)) / (1.0 + torch.exp(-t_internal)),
            pred_v,
        )
        gaussian_velocity_sigma_sq = self._match_dims(
            self.gaussian_velocity_sigma(t_internal) ** 2,
            pred_v,
        ).clamp_min(self.eps)

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
        f_t = self.scale_pos * f_t
        v_t = self.scale_pos * v_t
        dt_t = torch.as_tensor(
            self.time_scaling_T * dt,
            device=v_t.device,
            dtype=v_t.dtype,
        )
        noise_v = scatter_center(torch.randn_like(v_t), index=index)

        exp_dt = torch.exp(dt_t)
        exp_2dt_minus_1 = torch.exp(2.0 * dt_t) - 1.0
        expm1_dt = torch.expm1(dt_t)
        expm1_2dt = torch.expm1(2.0 * dt_t)
        noise_scale = torch.sqrt(expm1_2dt.clamp_min(self.eps))

        v_prev = exp_dt * v_t + 2.0 * expm1_dt * score_v + noise_scale * noise_v


        #Keep the internal reverse-time position latent in the centered unit-period chart.
        f_prev = self.wrap_angles(f_t - dt_t * v_prev)

        return f_prev / self.scale_pos, v_prev / self.scale_pos


    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
