"""Train continuous recognition (CSLR): CTR-GCN + Conformer + CTC on PHOENIX-2014T."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cslr.yaml")
    p.add_argument("--data-root", default="data/")
    p.add_argument("--streaming", action="store_true", help="train with bounded look-ahead")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: CTC training loop with AMP; KenLM rescoring at eval
    # TODO: target dev WER <= 21; verify CTC peak time-alignment
    raise NotImplementedError


if __name__ == "__main__":
    main()
