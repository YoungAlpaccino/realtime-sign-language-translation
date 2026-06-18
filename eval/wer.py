"""WER evaluation for CSLR (jiwer), with del/ins/sub breakdown; offline vs streaming."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hyp", required=True, help="hypotheses file")
    p.add_argument("--ref", required=True, help="references file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: jiwer.compute_measures(ref, hyp); print WER + S/D/I breakdown
    raise NotImplementedError


if __name__ == "__main__":
    main()
