from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.data import CSPTask, DNGTask, resolve_data_root



################################################################
####### NOTE:                                               ####
#######       Time is first mapped from normalized time     ####
#######       t01 in [0,1] to KLDM internal time by         ####
#######       t = tf * t01.        [Leap of fate]           ####
#######
#######       Fractional coordinates are not used directly  ####
#######       in raw [0,1] space when computing the wrapped ####
#######       normal score. KLDM first wraps them to a      ####
#######       centered periodic representation and then     ####
#######       rescales them by scale_pos = 2*pi.            ####
#######
#######       So the wrapped normal score is computed in    ####
#######       this 2*pi-periodic internal coordinate space, ####
#######       which is equivalent to a centered angle-like  ####
#######       representation, and only mapped back to       ####
#######       fractional coordinates afterward when returned####
################################################################



class TrivialisedDiffusion(nn.Module):
    """TDM forward process for positions + velocities.

    Variables:
        f0 : clean fractional coordinates
        v0 = 0: clean initial velocity

    Forward process:
        1) sample noisy velocity
        2) sample position displacement conditioned on velocity
        3) wrap back to the torus

    Returns:
        f_t            : noisy wrapped positions
        v_t            : noisy velocities
        score_v_target : Gaussian velocity score target
        pos_target     : wrapped-position target (here kept minimal)
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

    wrap = lambda self, x: torch.remainder(x, 1.0)


    # -------------------------------------------------
    # Velocity transition kernel schedulers
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

    def forward_sample(
        self,
        t: torch.Tensor,
        f0: torch.Tensor,
        v0: torch.Tensor | None = None,
        epsilon_v: torch.Tensor | None = None,
        epsilon_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        #Now we do T = [0,2] time scaling.
        t = self.time_scaling_T * t

        """
        The transition kernel is defined as follow:
            p_t|0 (ft, vt | f0, v0) = WN(r, | mu_r_t, sigma_r_t) * Nv(vt | mu_v_t, sigma_v_r)

            transitioner kernel =     sample r_t                 *           sample v_t

            We sample v_t, use it to move on manifold, to samlpe f_t
        """

        #######################
        ###    SAMPLE v_t   ###
        #######################

        #Vi sætter v0 = 0, [Design choice] at time t = 0
        if v0 is None:
            v0 = torch.zeros_like(f0)                           #Equation: Initial zero velocities

        #Sample the Nv(0,I) noise on velocty, epislon_v, i
        if epsilon_v is None:
            epsilon_v = torch.randn_like(v0)                        #Equation  Nv is a normal distribution such that ∑vi = 0

        #Alpha_v_t = exp(-t)
        alpha_v_t = self._match_dims(self.alpha_v(t), v0)       #Equation 22
        #Sigma_v_t = 1-exp(-2t)
        sigma_v_t = self._match_dims(self.sigma_v(t), v0)       #Equation 23

        #Sample v_t
        v_t = alpha_v_t * v0 + sigma_v_t * epsilon_v            #Equation 16: Reparamization sample of Nv(vt | mu_v_t, sigma_v_r)

        ######################################
        ###    Calculate displacement ft   ###
        ######################################
        #Now we calculate f_t = f_0 * expm(r_t),  where r_t = WN(r, | mu_r_t, sigma_r_t)

        #First we sample r_t = WN(r, | mu_r_t, sigma_r_t)
        mu_r_t = self.mu_r_t(t, v_t, v0)
        sigma_r_t = self._match_dims(self.sigma_r_t(t), f0)

        if epsilon_r is None:
            epsilon_r = torch.randn_like(f0)

        r_t = self.wrap(mu_r_t + sigma_r_t * epsilon_r)         #From pseudo code algorithm 1

        #Now we calculate displacement, and stay on the manifold.
        f_t = self.wrap(f0  + r_t)                              #Equation: C derivation

        return f_t, v_t, epsilon_v, epsilon_r

    def score_target(self, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:



        return NotImplementedError()

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff
