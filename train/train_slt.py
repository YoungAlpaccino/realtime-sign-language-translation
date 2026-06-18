"""Train translation (SLT): two-stage gloss->text or gloss-free Sign2Text."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cslr.yaml")
    p.add_argument("--mode", choices=["two_stage", "gloss_free"], default="two_stage")
    p.add_argument("--data-root", default="data/")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: seq2seq training; report BLEU-1..4 / chrF / ROUGE-L
    raise NotImplementedError


if __name__ == "__main__":
    main()
