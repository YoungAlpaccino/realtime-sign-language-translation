"""Translation metrics: BLEU-1..4 / chrF / ROUGE-L (sacrebleu) for SLT outputs."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp", required=True)
    p.add_argument("--ref", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: sacrebleu corpus_bleu / corpus_chrf; print all metrics
    raise NotImplementedError


if __name__ == "__main__":
    main()
