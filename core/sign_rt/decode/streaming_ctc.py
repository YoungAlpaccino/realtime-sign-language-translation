"""Streaming CTC decoding with rolling-buffer windowing.

Latency-sensitive: PORTED to TS — collapse rule must match exactly.

This is the greedy (best-path) streaming decoder used by the live surfaces: it
keeps the last emitted label across chunk boundaries so repeats that straddle a
window are not double-counted. A full beam search + KenLM shallow fusion is the
offline/Phase-2 path (see kenlm_rescore.py); the greedy decoder is what runs in
real time on edge/server/browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamingCTCDecoder:
    """Chunked greedy CTC decoder maintaining state across overlapping windows."""

    beam_width: int = 8  # reserved for beam search; greedy path ignores it
    blank: int = 0
    _committed: list[int] = field(default_factory=list)
    _prev: int | None = None

    def push(self, log_probs: np.ndarray) -> list[int]:
        """Feed a chunk of (T_chunk, vocab) log-probs; return the partial gloss so far.

        Greedy best path with cross-chunk de-duplication: argmax per frame, drop
        repeats (including across the previous chunk's tail), drop blanks.
        """
        ids = np.asarray(log_probs).argmax(axis=-1).astype(int).tolist()
        for i in ids:
            if i == self._prev:
                continue
            self._prev = i
            if i != self.blank:
                self._committed.append(i)
        return list(self._committed)

    def reset(self) -> None:
        """Clear state at segment boundaries."""
        self._committed = []
        self._prev = None
