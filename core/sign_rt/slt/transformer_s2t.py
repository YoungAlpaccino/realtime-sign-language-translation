"""Gloss-free Sign2Text: encoder-decoder directly from keypoint features."""
from __future__ import annotations

import torch
import torch.nn as nn


class Sign2TextTransformer(nn.Module):
    """Translate keypoint-feature sequences to text without gloss supervision."""

    def __init__(self, feat_dim: int, text_vocab: int, d_model: int = 512) -> None:
        super().__init__()
        # TODO: feature projection -> Transformer encoder-decoder -> lm head
        raise NotImplementedError

    def forward(self, feats: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, feat_dim) -> (B, T_tgt, text_vocab) logits."""
        raise NotImplementedError
