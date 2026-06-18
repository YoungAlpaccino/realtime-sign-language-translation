"""KenLM n-gram shallow-fusion rescoring of CTC beams."""
from __future__ import annotations


class KenLMRescorer:
    """Shallow-fusion rescorer: combine acoustic CTC score with n-gram LM score."""

    def __init__(self, lm_path: str, alpha: float = 0.5, beta: float = 0.0) -> None:
        # alpha: LM weight; beta: word-insertion bonus.
        # TODO: load KenLM model from lm_path
        self.alpha = alpha
        self.beta = beta
        raise NotImplementedError

    def score(self, tokens: list[str]) -> float:
        """Return LM log-prob for a gloss token sequence."""
        raise NotImplementedError
