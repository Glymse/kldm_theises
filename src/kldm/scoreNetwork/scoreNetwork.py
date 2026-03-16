from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def _build_fully_connected_edges(batch: torch.Tensor | None, num_nodes: int) -> torch.Tensor:
	"""Build a per-graph fully-connected edge index without self-loops."""
	if num_nodes == 0:
		return torch.empty((2, 0), dtype=torch.long)

	if batch is None:
		idx = torch.arange(num_nodes, dtype=torch.long)
		row = idx.repeat_interleave(num_nodes)
		col = idx.repeat(num_nodes)
		mask = row != col
		return torch.stack([row[mask], col[mask]], dim=0)

	edges: list[torch.Tensor] = []
	unique_graphs = torch.unique(batch)
	for graph_id in unique_graphs.tolist():
		node_idx = torch.where(batch == graph_id)[0]
		n = int(node_idx.numel())
		if n <= 1:
			continue

		row = node_idx.repeat_interleave(n)
		col = node_idx.repeat(n)
		mask = row != col
		edges.append(torch.stack([row[mask], col[mask]], dim=0))

	if not edges:
		return torch.empty((2, 0), dtype=torch.long, device=batch.device)

	return torch.cat(edges, dim=1)


class SimpleScoreGNN(nn.Module):
	"""Very simple GNN score network for CSP and de-novo batches.

	Expected `Data`/`Batch` fields:
	- `pos`: [N, 3] float coordinates
	- `h`: [N] integer atomic numbers
	- optional `batch`: [N] graph ids (provided by PyG DataLoader)
	"""

	def __init__(
		self,
		num_atom_types: int = 128,
		hidden_dim: int = 128,
		num_layers: int = 3,
		extra_feature_dim: int = 0,
	) -> None:
		super().__init__()

		self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
		self.extra_feature_dim = int(extra_feature_dim)
		self.input_proj = nn.Linear(hidden_dim + 3 + self.extra_feature_dim, hidden_dim)

		self.convs = nn.ModuleList()
		for _ in range(max(1, num_layers)):
			self.convs.append(GCNConv(hidden_dim, hidden_dim))

		self.output_proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, 3),
		)

	def forward(self, data: Data, extra_node_features: torch.Tensor | None = None) -> torch.Tensor:
		if not hasattr(data, "pos") or not hasattr(data, "h"):
			raise ValueError("Input data must contain 'pos' and 'h' fields")

		pos = data.pos
		h = data.h.long()
		batch = getattr(data, "batch", None)

		edge_index = _build_fully_connected_edges(batch=batch, num_nodes=pos.shape[0]).to(pos.device)

		x_parts = [pos, self.atom_embed(h)]
		if self.extra_feature_dim > 0:
			if extra_node_features is None:
				extra_node_features = torch.zeros(
					(pos.shape[0], self.extra_feature_dim), device=pos.device, dtype=pos.dtype
				)
			x_parts.append(extra_node_features)

		x = torch.cat(x_parts, dim=-1)
		x = self.input_proj(x)

		for conv in self.convs:
			x = conv(x, edge_index)
			x = F.silu(x)

		# Per-node 3D score/noise prediction.
		return self.output_proj(x)
