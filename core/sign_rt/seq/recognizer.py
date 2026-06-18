"""End-to-end CSLR recognizer: graph backbone -> Conformer -> CTC head.

Input is a window of normalized keypoints (B, 3, T, V); output is per-frame CTC
log-probs (B, T, vocab). This is the single graph exported to ONNX and served on
all three surfaces (edge / server / browser).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..graph.ctr_gcn import CTRGCNBlock
from ..graph.st_gcn import STGCNBlock
from ..pose.graph import build_adjacency
from ..pose.holistic import N_KEYPOINTS
from .conformer import ConformerEncoder
from .ctc_head import CTCHead


class Recognizer(nn.Module):
    """Keypoint window -> gloss CTC log-probs."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        spatial_backbone: str = "ctr_gcn",
        graph_channels: tuple[int, ...] = (64, 128),
        conformer_layers: int = 6,
        conformer_heads: int = 4,
        right_context: int = 8,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        adj = torch.from_numpy(build_adjacency())
        block_cls = CTRGCNBlock if spatial_backbone == "ctr_gcn" else STGCNBlock

        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in graph_channels:
            blocks.append(block_cls(c_in, c_out, adj))
            c_in = c_out
        self.graph = nn.ModuleList(blocks)

        self.proj = nn.Linear(c_in, d_model)
        self.encoder = ConformerEncoder(
            d_model=d_model,
            n_layers=conformer_layers,
            n_heads=conformer_heads,
            right_context=right_context,
        )
        self.ctc = CTCHead(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C=3, T, V=N_KEYPOINTS) -> log-probs (B, T, vocab)."""
        for blk in self.graph:
            x = blk(x)
        x = x.mean(dim=3)            # spatial pool over nodes -> (B, C, T)
        x = x.transpose(1, 2)        # (B, T, C)
        x = self.proj(x)             # (B, T, d_model)
        x, _ = self.encoder(x)
        return self.ctc(x)           # (B, T, vocab) log-probs

    @staticmethod
    def dummy_input(batch: int = 1, t: int = 64) -> torch.Tensor:
        """A correctly-shaped random input for export / parity / smoke tests."""
        return torch.randn(batch, 3, t, N_KEYPOINTS)
