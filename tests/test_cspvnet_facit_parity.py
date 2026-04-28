from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

if "torch_scatter" not in sys.modules:
    scatter_mod = types.ModuleType("torch_scatter")

    def _scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, reduce: str = "sum", dim_size: int | None = None):
        if dim != 0:
            raise NotImplementedError("Test fallback only supports dim=0.")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0

        out_shape = (dim_size, *src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

        if reduce in {"sum", "add", "mean"}:
            counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            for row, idx in zip(src, index.tolist(), strict=False):
                out[idx] = out[idx] + row
                counts[idx] = counts[idx] + 1
            if reduce == "mean":
                counts = counts.clamp_min(1)
                expand = counts.view(-1, *([1] * (src.ndim - 1)))
                out = out / expand
            return out
        raise NotImplementedError(f"Test fallback does not support reduce={reduce!r}.")

    def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None):
        return _scatter(src, index=index, dim=dim, reduce="mean", dim_size=dim_size)

    scatter_mod.scatter = _scatter
    scatter_mod.scatter_mean = _scatter_mean
    sys.modules["torch_scatter"] = scatter_mod

from kldm.scoreNetwork.scoreNetwork import CSPVNet as LocalCSPVNet


REPO_ROOT = Path(__file__).resolve().parents[1]
FACIT_ROOT = REPO_ROOT / "src" / "facitKLDM" / "kldm-main-git"
FACIT_ARCH_PATH = FACIT_ROOT / "src_kldm" / "nn" / "arch.py"
LOCAL_ARCH_PATH = REPO_ROOT / "src" / "kldm" / "scoreNetwork" / "scoreNetwork.py"

if str(FACIT_ROOT) not in sys.path:
    sys.path.insert(0, str(FACIT_ROOT))

from src_kldm.nn.arch import CSPVNet as FacitCSPVNet  # noqa: E402


def _normalize_arch_source(source: str) -> str:
    normalized = source.replace("from src_kldm.nn.embedding", "from kldm.scoreNetwork.embedding")
    normalized = normalized.replace("from src_kldm.nn.utils", "from kldm.scoreNetwork.utils")
    return normalized.strip()


def _build_local_net() -> LocalCSPVNet:
    return LocalCSPVNet(
        hidden_dim=512,
        time_dim=256,
        num_layers=6,
        h_dim=100,
        num_freqs=128,
        ln=True,
        smooth=False,
        pred_h=False,
        pred_l=True,
        pred_v=True,
        zero_cog=True,
    )


def _build_facit_net() -> FacitCSPVNet:
    return FacitCSPVNet(
        hidden_dim=512,
        time_dim=256,
        num_layers=6,
        h_dim=100,
        num_freqs=128,
        ln=True,
        smooth=False,
        pred_h=False,
        pred_l=True,
        pred_v=True,
        zero_cog=True,
    )


def test_cspvnet_arch_source_matches_facit_mp20() -> None:
    facit_source = FACIT_ARCH_PATH.read_text(encoding="utf-8")
    local_source = LOCAL_ARCH_PATH.read_text(encoding="utf-8")
    assert _normalize_arch_source(local_source) == _normalize_arch_source(facit_source)


def test_cspvnet_state_dict_layout_matches_facit_mp20() -> None:
    local_net = _build_local_net()
    facit_net = _build_facit_net()

    local_state = local_net.state_dict()
    facit_state = facit_net.state_dict()

    assert list(local_state.keys()) == list(facit_state.keys())
    assert {key: tuple(value.shape) for key, value in local_state.items()} == {
        key: tuple(value.shape) for key, value in facit_state.items()
    }


def test_cspvnet_forward_output_shapes_match_facit_mp20() -> None:
    torch.manual_seed(0)

    local_net = _build_local_net()
    facit_net = _build_facit_net()

    num_graphs = 2
    nodes_per_graph = 3
    num_nodes = num_graphs * nodes_per_graph

    t = torch.rand(num_graphs, 1)
    pos = torch.rand(num_nodes, 3)
    v = torch.rand(num_nodes, 3)
    h = torch.randint(1, 100, (num_nodes,), dtype=torch.long)
    l = torch.rand(num_graphs, 6)
    node_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    edge_pairs = []
    for graph_id in range(num_graphs):
        start = graph_id * nodes_per_graph
        end = start + nodes_per_graph
        for src in range(start, end):
            for dst in range(start, end):
                if src != dst:
                    edge_pairs.append((src, dst))
    edge_node_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    local_out = local_net(
        t=t,
        pos=pos,
        v=v,
        h=h,
        l=l,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )
    facit_out = facit_net(
        t=t,
        pos=pos,
        v=v,
        h=h,
        l=l,
        node_index=node_index,
        edge_node_index=edge_node_index,
    )

    assert set(local_out.keys()) == set(facit_out.keys()) == {"v", "l"}
    assert local_out["v"].shape == facit_out["v"].shape == (num_nodes, 3)
    assert local_out["l"].shape == facit_out["l"].shape == (num_graphs, 6)


if __name__ == "__main__":
    test_cspvnet_arch_source_matches_facit_mp20()
    test_cspvnet_state_dict_layout_matches_facit_mp20()
    test_cspvnet_forward_output_shapes_match_facit_mp20()
    print("cspvnet parity checks passed")
