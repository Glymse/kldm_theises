from __future__ import annotations

import math
import importlib
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[1]
FACIT_ROOT = WORKSPACE_ROOT / "src" / "facitKLDM" / "kldm-main-git"

if str(FACIT_ROOT) not in sys.path:
    sys.path.insert(0, str(FACIT_ROOT))


def _build_shared_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Two tiny graphs with centered random coordinates in [-0.5, 0.5).
    index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long, device=device)
    pos = torch.rand((len(index), 3), device=device) - 0.5
    t_graph = torch.tensor([[0.25], [0.80]], dtype=torch.get_default_dtype(), device=device)
    t_node = t_graph[index].squeeze(-1)
    return index, pos, t_node


def _spawn_models(device: torch.device):
    FacitTDM = importlib.import_module("src_kldm.model.tdm").TDM
    TrivialisedDiffusionDev = importlib.import_module("kldm.diffusionModels.TDMdev").TrivialisedDiffusionDev

    facit = FacitTDM(
        scale_pos=2.0 * math.pi,
        k_wn_score=13,
        tf=2.0,
        simplified_parameterization=True,
        n_sigmas=400,
    ).to(device)
    ours = TrivialisedDiffusionDev(
        eps=1e-6,
        k_wn_score=13,
        n_sigmas=400,
    ).to(device)
    return ours.eval(), facit.eval()


def test_forward_targets_match_facit_under_unit_scaling() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1234)
    ours, facit = _spawn_models(device)
    index, pos01, t_node = _build_shared_inputs(device)

    # Build one shared stochastic realization in facit units, then convert.
    v0_unscaled = torch.zeros_like(pos01)
    eps_v_unscaled = facit.velocity_distribution.sample(index)
    eps_r_unscaled = facit.velocity_distribution.sample(index)

    pos_facit01 = torch.remainder(pos01 + 0.5, 1.0) - 0.5
    # Force facit to use the exact same stochastic draws as our unit-scaled path.
    queued_samples = [eps_v_unscaled, eps_r_unscaled]

    def _sample_queue(index: torch.Tensor) -> torch.Tensor:
        assert len(queued_samples) > 0
        out = queued_samples.pop(0)
        assert out.shape[0] == len(index)
        return out

    facit.init_velocity_distribution.sample = lambda idx: torch.zeros((len(idx), 3), device=device)
    facit.velocity_distribution.sample = _sample_queue

    (_, _), target_facit = facit.training_targets(t_node.unsqueeze(-1), pos_facit01, index=index)

    _, v_t, _, _, r_t = ours.forward_sample(
        t=t_node,
        f0=pos01,
        index=index,
        v0=v0_unscaled / (2.0 * math.pi),
        epsilon_v=eps_v_unscaled,
        epsilon_r=eps_r_unscaled,
    )
    target_ours = ours.score_target(t=t_node, r_t=r_t, v_t=v_t, index=index)

    # Numerical drift comes mostly from sigma_norm precompute approximation.
    torch.testing.assert_close(target_ours, target_facit, atol=2e-2, rtol=2e-2)


def test_reverse_em_step_matches_facit_after_rescaling() -> None:
    device = torch.device("cpu")
    torch.manual_seed(5678)
    ours, facit = _spawn_models(device)
    index, _, t_node = _build_shared_inputs(device)

    # Shared latent state and network prediction.
    pos_unit = torch.rand((len(index), 3), device=device) - 0.5
    v_unit = (torch.rand((len(index), 3), device=device) - 0.5) * 0.2
    pred_unit = torch.randn_like(v_unit)

    dt = 1.0 / 200.0
    t_graph = torch.tensor([[0.40], [0.65]], dtype=torch.get_default_dtype(), device=device)
    t_facit = t_graph

    score_ours = ours.construct_velocity_score(
        t=t_graph[index].squeeze(-1),
        v_t=v_unit,
        pred_v=pred_unit,
    )

    torch.manual_seed(99)
    pos_prev_ours, v_prev_ours = ours.reverse_exp_step(
        f_t=pos_unit,
        v_t=v_unit,
        score_v=score_ours,
        index=index,
        dt=dt,
    )

    torch.manual_seed(99)
    pos_prev_facit, v_prev_facit = facit.reverse_step_em(
        t=t_facit,
        v_t=v_unit,
        pos_t=pos_unit,
        pred_v_t=pred_unit,
        dt=torch.as_tensor(-dt, dtype=t_facit.dtype, device=device),
        node_index=index,
    )

    # Both models operate in unit-period coordinates at their public interface.
    torch.testing.assert_close(pos_prev_ours, pos_prev_facit, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_prev_ours, v_prev_facit, atol=3e-5, rtol=3e-5)
