"""ST-GCN spatial-temporal graph convolution blocks (pose baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    """One spatial graph conv + temporal conv block.

    Spatial step aggregates node features over a fixed normalized adjacency;
    temporal step is a strided 1-D conv along time. Residual + BN + ReLU.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        adjacency: torch.Tensor,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)  # (V, V)

        # Spatial: 1x1 conv mixes channels before graph aggregation.
        self.gconv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # Temporal: conv over time only (kernel along T, width 1 over V).
        pad = (temporal_kernel - 1) // 2
        self.tconv = nn.Conv2d(out_ch, out_ch, kernel_size=(temporal_kernel, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if in_ch == out_ch:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, V) -> (B, C', T, V)."""
        res = self.residual(x)
        h = self.gconv(x)
        # Graph aggregation: out[..., v] = sum_w A[v, w] * h[..., w].
        h = torch.einsum("bctw,vw->bctv", h, self.adjacency)
        h = self.tconv(h)
        h = self.bn(h)
        return self.relu(h + res)
