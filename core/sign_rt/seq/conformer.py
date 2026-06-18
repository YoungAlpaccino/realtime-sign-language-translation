"""Conformer temporal encoder with optional cached/chunked attention for streaming."""
from __future__ import annotations

import torch
import torch.nn as nn


class _FeedForward(nn.Module):
    """Half-step feed-forward module (Macaron-style)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvModule(nn.Module):
    """Conformer convolution module: pointwise -> GLU -> depthwise -> BN -> SiLU -> pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=pad, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.ln(x).transpose(1, 2)        # (B, D, T)
        h = nn.functional.glu(self.pw1(h), dim=1)
        h = self.act(self.bn(self.dw(h)))
        h = self.pw2(h)
        return self.drop(h.transpose(1, 2))   # (B, T, D)


class _MultiHeadSelfAttention(nn.Module):
    """Self-attention written with plain ops so it exports with a dynamic T axis.

    (nn.MultiheadAttention bakes the sequence length into a Reshape, which breaks
    variable-length / streaming inference in onnxruntime-web.)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (B, h, T, dh)
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, T, T)
        if attn_mask is not None:
            att = att + attn_mask                   # broadcast (T, T)
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(b, t, self.h * self.dh)
        return self.out(out)


class ConformerBlock(nn.Module):
    """FFN -> MHSA -> Conv -> FFN with a final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = _FeedForward(d_model, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.conv = _ConvModule(d_model, dropout=dropout)
        self.ffn2 = _FeedForward(d_model, dropout=dropout)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.attn_ln(x), attn_mask=attn_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.out_ln(x)


class ConformerEncoder(nn.Module):
    """Conformer encoder over graph-pooled per-frame features.

    For streaming, supports a bounded right-context (look-ahead capped at ~320 ms)
    via a banded attention mask. Full-sequence by default; `right_context` < 0
    means unbounded look-ahead.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        right_context: int = 8,  # frames of look-ahead (~320 ms @ 25 FPS)
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.right_context = right_context
        self.layers = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

    def _lookahead_mask(self, t: int, device: torch.device) -> torch.Tensor | None:
        """Additive (-inf) mask allowing each frame to attend up to `right_context` ahead."""
        if self.right_context is None or self.right_context < 0:
            return None
        idx = torch.arange(t, device=device)
        allowed = idx.unsqueeze(1) + self.right_context >= idx.unsqueeze(0)  # (T_q, T_k)
        mask = torch.zeros(t, t, device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(
        self, x: torch.Tensor, cache: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """x: (B, T, d_model) -> (encoded (B, T, d_model), updated cache)."""
        mask = self._lookahead_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return x, (cache or {})
