from __future__ import annotations
"""
Multiscale dilated convolution for single-modality sequence refinement.

Applied after BDCA. Each branch has a kernel size of 3, with dilation rates
of 1, 2, and 4 and corresponding receptive fields of 3, 5, and 9 tokens.

Three parallel branches are combined by a weighted sum, with one learned scalar
per scale normalized across the three branches by softmax. The fused output is
normalized and added to the input before the feedforward residual update.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedBranch(nn.Module):
    """One depthwise-separable dilated conv branch."""

    def __init__(self, d_model: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2   # same-length padding
        # Depthwise conv (groups=d_model) → pointwise
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel,
                            padding=pad, dilation=dilation,
                            groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.pw(F.gelu(self.dw(h)))
        h = h.transpose(1, 2)
        return self.drop(self.norm(h))


class MultiScaleConvRefinerLayer(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.branch1 = _DilatedBranch(d_model, kernel=3, dilation=1, dropout=dropout)
        self.branch2 = _DilatedBranch(d_model, kernel=3, dilation=2, dropout=dropout)
        self.branch4 = _DilatedBranch(d_model, kernel=3, dilation=4, dropout=dropout)

        self.scale_weights = nn.Parameter(torch.ones(3) / 3)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)

        w = torch.softmax(self.scale_weights, dim=0)
        fused = w[0] * b1 + w[1] * b2 + w[2] * b4

        out = x + self.drop(self.norm(fused))
        out = out + self.ffn(out)
        return out


class MultiScaleConvRefiner(nn.Module):
    """Stack refinement layers and apply final layer normalization."""

    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiScaleConvRefinerLayer(d_model, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
