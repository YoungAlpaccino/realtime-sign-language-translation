"""CTR-GCN: channel-wise topology refinement GCN (proposed spatial backbone).

Compact, ONNX-exportable variant of Chen et al. (ICCV'21): a shared static
topology is refined per-channel from temporally-pooled pairwise feature
differences, then used to aggregate node features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CTRGCNBlock(nn.Module):
    """Channel-wise topology refinement graph conv block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        adjacency: torch.Tensor,
        rel_reduction: int = 8,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)  # (V, V)
        self.out_ch = out_ch

        mid = max(out_ch // rel_reduction, 1)
        self.theta = nn.Conv2d(in_ch, mid, kernel_size=1)
        self.phi = nn.Conv2d(in_ch, mid, kernel_size=1)
        # Expand the per-pair refinement back to per-output-channel topology.
        self.expand = nn.Conv2d(mid, out_ch, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))  # start as pure static topology

        self.transform = nn.Conv2d(in_ch, out_ch, kernel_size=1)

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

        # Temporally-pooled node descriptors for topology refinement.
        t1 = self.theta(x).mean(dim=2)  # (B, mid, V)
        t2 = self.phi(x).mean(dim=2)    # (B, mid, V)
        diff = torch.tanh(t1.unsqueeze(-1) - t2.unsqueeze(-2))  # (B, mid, V, V)
        refine = self.expand(diff)      # (B, out_ch, V, V)

        # Per-channel topology = shared static A + learned refinement.
        a = self.adjacency.unsqueeze(0).unsqueeze(0) + self.alpha * refine  # (B, out, V, V)

        h = self.transform(x)  # (B, out, T, V)
        h = torch.einsum("bctw,bcvw->bctv", h, a)
        h = self.tconv(h)
        h = self.bn(h)
        return self.relu(h + res)
