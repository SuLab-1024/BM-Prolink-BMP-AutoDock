from __future__ import annotations
"""Encode molecular graphs with a graph isomorphism network (GIN).

Inputs are atom features, an adjacency matrix, and an atom mask.
After updating node features, average the valid atoms and apply the graph
readout. An atom mask of 1 denotes a valid atom and 0 denotes padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugGNN(nn.Module):

    def __init__(
        self,
        atom_dim: int = 78,       # atom feature dim (from RDKit)
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_GINConv(hidden_dim, hidden_dim, dropout))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        atom_feats: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.atom_proj(atom_feats)

        for layer in self.layers:
            h = layer(h, adj)

        mask_f = mask.float().unsqueeze(-1)
        h_pool = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.readout(h_pool)


class _GINConv(nn.Module):
    """
    Aggregate node features with the adjacency matrix, then apply the MLP, residual
    addition, ReLU, and layer normalization.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (B, N, D)  — node features
        adj: (B, N, N)  — adjacency (0/1, possibly with self-loops)
        """
        neighbor_msg = torch.bmm(adj, h)
        h_update = (1 + self.eps) * h + neighbor_msg
        h_update = self.mlp(h_update)
        return self.norm(self.act(h + h_update))
