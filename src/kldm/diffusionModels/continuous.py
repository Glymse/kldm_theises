from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kldm.data import CSPTask, DNGTask, resolve_data_root


class ContinuousVPDiffusion(nn.Module):
    """Small variance-preserving diffusion for Euclidean modalities.

    Used for:
    - lattice parameters `l`
    - atom representations `a` in the continuous DNG variant

    We use the closed-form forward process of a simple VP SDE. For a clean input
    x_0 and diffusion time t, the marginal distribution is:

        x_t | x_0 ~ N(alpha(t) x_0, sigma(t)^2 I)

    so the noisy sample can be written as:

        x_t = alpha(t) x_0 + sigma(t) eps,    eps ~ N(0, I)

    This is the object we need during training: we can draw x_t directly without
    numerically simulating an SDE path, and we can also write down the score
    target in closed form.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    # For the VP SDE
    #
    #   dx = -beta x dt + sqrt(2 beta) dW_t
    #
    # with constant beta = 1, the marginal mean coefficient becomes exp(-t)
    # and the variance becomes 1 - exp(-2t). In this file we expose those two
    # pieces directly as alpha(t) and sigma(t).
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient of the forward kernel."""
        return torch.exp(-t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the forward kernel."""
        return torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * t), min=self.eps))

    def forward_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from the closed-form VP transition kernel.

        Given x_0 and a time t, we draw

            x_t = alpha(t) x_0 + sigma(t) eps

        where eps ~ N(0, I).
        """
        if noise is None:
            noise = torch.randn_like(x0)  # Sample epsilon ~ N(0, I).

        alpha_t = self._match_dims(self.alpha(t), x0)
        sigma_t = self._match_dims(self.sigma(t), x0)
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, noise

    def score_target(self, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Return the exact score target for the Gaussian forward kernel.

        Since

            x_t | x_0 ~ N(alpha(t) x_0, sigma(t)^2 I),

        the score of that conditional density is

            grad_{x_t} log p(x_t | x_0) = -eps / sigma(t),

        when x_t is parameterized as alpha(t) x_0 + sigma(t) eps.
        """

        #epsilon becomes [eps, 1,1,1,....] to broadcast to sigma
        sigma_t = self._match_dims(self.sigma(t), eps)

        return -eps / sigma_t

    @staticmethod
    def _match_dims(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand batch-wise coefficients until they broadcast with `x`."""
        while coeff.ndim < x.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff







def _print_header(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _preview(name: str, tensor: torch.Tensor, rows: int = 3) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}")
    if tensor.ndim == 0:
        print(tensor)
    elif tensor.ndim == 1:
        print(tensor[: min(rows, tensor.shape[0])])
    else:
        print(tensor[: min(rows, tensor.shape[0])])


def _run_case(
    *,
    title: str,
    diffusion: ContinuousVPDiffusion,
    x0: torch.Tensor,
    t: torch.Tensor,
    preview_rows: int = 3,
) -> None:
    """Run one forward-diffusion example with the real class implementation."""
    x_t, eps = diffusion.forward_sample(t, x0)
    score = diffusion.score_target(t, eps)

    alpha_t = diffusion._match_dims(diffusion.alpha(t), x0)
    sigma_t = diffusion._match_dims(diffusion.sigma(t), x0)
    x_t_check = alpha_t * x0 + sigma_t * eps
    score_check = -eps / sigma_t

    _print_header(title)
    _preview("Before diffusion (x0)", x0, preview_rows)
    _preview("Noise used (eps)", eps, preview_rows)
    _preview("After diffusion (x_t)", x_t, preview_rows)
    _preview("Target score", score, preview_rows)

    print("\nMax abs diff between x_t and formula:")
    print((x_t - x_t_check).abs().max().item())
    print("\nMax abs diff between score and formula:")
    print((score - score_check).abs().max().item())


def main() -> None:
    torch.manual_seed(7)

    # Use the actual diffusion class defined above, then run it on real batches
    # coming from the CSP and DNG MatterGen-backed loaders.
    diffusion = ContinuousVPDiffusion()
    root = resolve_data_root()
    base_t = torch.tensor([1], dtype=torch.float32)

    print(f"root={root}")
    print(f"t={base_t.item():.2f}, alpha(t)={diffusion.alpha(base_t).item():.4f}, sigma(t)={diffusion.sigma(base_t).item():.4f}")

    csp_batch = next(iter(CSPTask().dataloader(root=root, split="train", batch_size=2, shuffle=False, download=True)))
    _run_case(
        title="CASE 1: CSP / diffuse lattice on real data",
        diffusion=diffusion,
        x0=csp_batch.l,
        t=base_t.expand(csp_batch.l.shape[0]),
    )

    # DNG atom features are node-level, so we repeat the graph time for each node.
    dng_batch = next(iter(DNGTask().dataloader(root=root, split="train", batch_size=2, shuffle=False, download=True)))
    node_t = base_t.expand(dng_batch.num_graphs)[dng_batch.batch]
    _run_case(
        title="CASE 2: DNG / diffuse atom features on real data",
        diffusion=diffusion,
        x0=dng_batch.h,
        t=node_t,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
