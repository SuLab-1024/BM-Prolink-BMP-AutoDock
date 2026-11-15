from __future__ import annotations
"""Bidirectional differential cross-attention between drug and protein representations.

Each direction forms two query sets and computes two attention distributions
against shared keys. Their weighted difference aggregates the shared values.
Apply GroupNorm, scaling, and output projection, then add the original query.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DIFF_ATTN_HEADS, DROPOUT, D_MODEL


class DifferentialAttention(nn.Module):
    """Single-direction differential cross-attention."""

    def __init__(self, d_model: int = D_MODEL, num_heads: int = DIFF_ATTN_HEADS,
                 dropout: float = DROPOUT, layer_idx: int = 1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = d_model // num_heads          # d per head
        self.scale      = math.sqrt(self.head_dim)      # √d (full head dim)
        self.layer_idx  = layer_idx

        # Compute the base coefficient using the layer index, starting at 1.
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))

        # Learnable vectors for λ parameterisation (per head)
        H, d = num_heads, self.head_dim
        self.lambda_q1 = nn.Parameter(torch.zeros(H, d))
        self.lambda_k1 = nn.Parameter(torch.zeros(H, d))
        self.lambda_q2 = nn.Parameter(torch.zeros(H, d))
        self.lambda_k2 = nn.Parameter(torch.zeros(H, d))
        nn.init.normal_(self.lambda_q1, std=0.1)
        nn.init.normal_(self.lambda_k1, std=0.1)
        nn.init.normal_(self.lambda_q2, std=0.1)
        nn.init.normal_(self.lambda_k2, std=0.1)

        # Q projected to 2×d_model so we can split Q1/Q2
        self.q_proj   = nn.Linear(d_model, 2 * d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.v_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

        # Scaled SubLN: GroupNorm with one group per attention head
        self.sub_ln   = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

        self.attn_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)

    def _lambda(self) -> torch.Tensor:
        """Compute scalar λ per head: shape (1, H, 1, 1) for broadcasting."""
        lam = (torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1)) -
               torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1)) +
               self.lambda_init)
        return lam.view(1, self.num_heads, 1, 1)

    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, Lq, D)
            key:   (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk) bool — True = padding position

        Returns:
            output:  (B, Lq, D)
            attn_w:  (B, H, Lq, Lk) differential attention weights
        """
        B, Lq, D = query.shape
        Lk       = key.shape[1]
        H, d     = self.num_heads, self.head_dim

        q_full = self.q_proj(query)
        k      = self.k_proj(key).view(B, Lk, H, d).transpose(1, 2)
        v      = self.v_proj(value).view(B, Lk, H, d).transpose(1, 2)

        q1, q2 = q_full.chunk(2, dim=-1)
        q1 = q1.view(B, Lq, H, d).transpose(1, 2)
        q2 = q2.view(B, Lq, H, d).transpose(1, 2)

        a1 = torch.matmul(q1, k.transpose(-2, -1)) / self.scale
        a2 = torch.matmul(q2, k.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            # (B,Lk) → (B,1,1,Lk), broadcast over heads and queries
            inf_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            a1 = a1.masked_fill(inf_mask, float("-inf"))
            a2 = a2.masked_fill(inf_mask, float("-inf"))

        a1 = F.softmax(a1, dim=-1)
        a2 = F.softmax(a2, dim=-1)
        # Set attention weights to zero when all keys are masked.
        a1 = torch.nan_to_num(a1, nan=0.0)
        a2 = torch.nan_to_num(a2, nan=0.0)

        lam    = self._lambda()
        diff_w = a1 - lam * a2
        diff_w = self.attn_drop(diff_w)

        out = torch.matmul(diff_w, v)

        # Reshape: (B,H,Lq,d) → (B, d_model, Lq) → GroupNorm → back
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Lq, D)
        out = self.sub_ln(out.transpose(1, 2)).transpose(1, 2)
        out = (1.0 - self.lambda_init) * out

        out = self.out_proj(out)
        return query + out, diff_w.detach()


class BidirectionalDiffCrossAttention(nn.Module):
    """Compute drug-to-protein and protein-to-drug differential cross-attention.

    Use layer_idx to set the base coefficient for both directions.
    """

    def __init__(self, d_model: int = D_MODEL, num_heads: int = DIFF_ATTN_HEADS,
                 dropout: float = DROPOUT, layer_idx: int = 1):
        super().__init__()
        self.drug2prot = DifferentialAttention(d_model, num_heads, dropout, layer_idx=layer_idx)
        self.prot2drug = DifferentialAttention(d_model, num_heads, dropout, layer_idx=layer_idx)

    def forward(self,
                h_drug: torch.Tensor,
                h_prot: torch.Tensor,
                drug_mask: torch.Tensor | None = None,
                prot_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_drug: (B, Ld, D)
            h_prot: (B, Lp, D)
            drug_mask / prot_mask: (B, L) bool, True = padding

        Returns:
            h_drug_out, h_prot_out, attn_d2p, attn_p2d
        """
        h_drug_out, attn_d2p = self.drug2prot(
            query=h_drug, key=h_prot, value=h_prot, key_padding_mask=prot_mask)
        h_prot_out, attn_p2d = self.prot2drug(
            query=h_prot, key=h_drug, value=h_drug, key_padding_mask=drug_mask)
        return h_drug_out, h_prot_out, attn_d2p, attn_p2d
