from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch_geometric.utils import scatter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.data import CSPTask, DNGTask, resolve_data_root

from kldm.distribution import d_log_wrapped_normal
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
            eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.time_scaling_T = 2

    # -------------------------------------------------
    #  Wrapping function.
    # -------------------------------------------------

    @staticmethod
    def wrap_fractional(x: torch.Tensor) -> torch.Tensor:
        """Wrap coordinates into [0, 1)."""
        return torch.remainder(x, 1.0)

    #displacements usually should be in [-0.5,0.5), see report.
    @staticmethod
    def wrap_signed_unit(x: torch.Tensor) -> torch.Tensor:
        """Wrap signed periodic displacements into [-0.5, 0.5)."""
        return torch.remainder(x + 0.5, 1.0) - 0.5

    # -------------------------------------------------
    # Velocity sampling
    # -------------------------------------------------

    # v_t | v_0 ~ N(exp(-t) v_0, (1 - exp(-2t)) I)
    def alpha_v(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the forward kernel."""
        return torch.exp(-t)

    def sigma_v(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the forward kernel."""
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def mu_r_t(self, t: torch.Tensor, v_t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        coeff = (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))  # Eq. (22)
        coeff = self._match_dims(coeff, v_t)
        return coeff * (v_t + v0)

    def sigma_r_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.clamp(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0, min=self.eps)
        )  # Eq. (23)


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
            p_t|0 (ft, vt | f0, v0) = WN(r, | mu_r_t, sigma_r_t) * Nv(vt | mu_v_t, sigma_v_r)

            transition kernel =          sample r_t              *         sample v_t

            We sample v_t, use it to move f_0 on manifold, to samlpe f_t
        """

        #######################
        ###    SAMPLE v_t   ###
        #######################

        #Vi sætter v0 = 0, [Design choice] at time t = 0
        if v0 is None:
            v0 = torch.zeros_like(f0)                               #Design choice: Initial zero velocities

        #TODO: Scatter center mean free, også det de gør i KLDM

        #Sample normal noise for velocity                       # Nv is a normal distribution such that ∑vi = 0
        if epsilon_v is None:
            epsilon_v = torch.randn_like(v0)
        epsilon_v = scatter_center(epsilon_v, index=index) #Zero mean

        #Alpha_v_t = exp(-t)
        alpha_v_t = self._match_dims(self.alpha_v(t), v0)       #Equation 22
        #Sigma_v_t = 1-exp(-2t)
        sigma_v_t = self._match_dims(self.sigma_v(t), v0)       #Equation 23

        #Sample v_t, given initial velocity.
        v_t = alpha_v_t * v0 + sigma_v_t * epsilon_v            #Equation 16: Reparamization sample of Nv(vt | mu_v_t, sigma_v_r)

        ######################################
        ###    Calculate displacement ft   ###
        ######################################
        #Now we calculate f_t = f_0 * expm(r_t),  where r_t = WN(r, | mu_r_t, sigma_r_t)

        #First we sample r_t = WN(r, | mu_r_t, sigma_r_t)
        mu_r_t = self.mu_r_t(t, v_t, v0)
        sigma_r_t = self._match_dims(self.sigma_r_t(t), f0)

        #Sample normal noise on epsilon
        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)                        # Nr is a normal distribution such that ∑vi = 0
        epsilon_r = scatter_center(epsilon_r, index=index)


        """ OLD VERSION, CHAT MIGHT SAY IT IS A PROBLEM
        r_t = self.wrap_signed_unit(mu_r_t + sigma_r_t * epsilon_r)  # Signed wrapped displacement

        #Now we calculate displacement, and while we stay on the manifold.
        f_t = self.wrap_fractional(f0 + r_t)                               # Map back to [0, 1)

        #Center again
        f_t = scatter_center(f_t, index=index)
        f_t = self.wrap_fractional(f_t)
        """
        #Move to [-0.5, 0.5]

        #PSEUDO: rt = w(µrt(t, v0, vt) + σrt, (t, v0, vt) · ϵrt ) ▷ w indicates the wrap function. ft = w(f0 + rt) ft = center(ft) ▷ center(·) keeps the center of gravity fixed
        r_t = self.wrap_signed_unit(mu_r_t + sigma_r_t * epsilon_r)
        #Remove mean
        r_t = scatter_center(r_t, index=index)
        r_t = self.wrap_signed_unit(r_t)
        f_t = self.wrap_fractional(f0 + r_t)



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
        """Return the TDM velocity training target used by KLDM. """

        #We do time scaling.
        t = self.time_scaling_T * t

        #Design choice, makes the target quite simple to calculate.
        v0 = torch.zeros_like(v_t) if v0 is None else v0

        sigma_v_t = self._match_dims(self.sigma_v(t), v_t)

        #Simplified target of normal velocity distribution
        gaussian_target = -v_t / sigma_v_t.clamp_min(self.eps).pow(2)

        #Now we find target of the wrapped normal fractional distribution
        mu_r_t = self.mu_r_t(t, v_t, v0)


        sigma_r_t = self._match_dims(self.sigma_r_t(t), r_t)
        wrapped_normal_target = self._match_dims((1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t)), r_t) * d_log_wrapped_normal(
            r=r_t,
            mu=mu_r_t,
            sigma=sigma_r_t,
        )
        wrapped_normal_target = scatter_center(wrapped_normal_target, index=index)

        target = gaussian_target + wrapped_normal_target
        return target
        #return target_s


    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
