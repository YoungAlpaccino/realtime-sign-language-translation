"""Two-stage SLT: seq2seq Transformer translating recognized gloss -> text."""
from __future__ import annotations

import torch
import torch.nn as nn


class Gloss2TextTransformer(nn.Module):
    """Encoder-decoder Transformer over gloss tokens (mirrors Sign-Language-Transformers)."""

    def __init__(self, gloss_vocab: int, text_vocab: int, d_model: int = 512) -> None:
        super().__init__()
        # TODO: nn.Transformer encoder-decoder + token/pos embeddings + lm head
        raise NotImplementedError

    def forward(self, gloss_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """-> (B, T_tgt, text_vocab) logits."""
        raise NotImplementedError
