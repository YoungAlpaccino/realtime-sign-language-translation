"""CTC head: gloss logits + blank-collapse decoding utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """Linear projection to gloss vocabulary (+ blank) for CTC."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # vocab_size includes the CTC blank symbol.
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> log-probs (B, T, vocab_size)."""
        return self.proj(x).log_softmax(dim=-1)


def ctc_collapse(ids: list[int], blank: int = 0) -> list[int]:
    """Standard CTC collapse: merge repeats, drop blanks.

    PORTED to TS (ctc_collapse.ts) — keep numerically/logically identical.

    Step 1: collapse runs of identical labels to a single label.
    Step 2: drop the blank symbol.
    """
    out: list[int] = []
    prev: int | None = None
    for i in ids:
        if i == prev:
            continue
        prev = i
        if i != blank:
            out.append(i)
    return out
