"""Dataset preparation: download metadata, extract Holistic keypoint caches.

Builds pose caches for WLASL/MS-ASL/PHOENIX-2014T/CSL-Daily/How2Sign.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=["wlasl", "msasl", "phoenix14t", "csl_daily", "how2sign"],
        required=True,
    )
    p.add_argument("--out", default="data/")
    p.add_argument("--cache-keypoints", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: verify license acceptance; extract frames; run Holistic; write codec caches
    raise NotImplementedError


if __name__ == "__main__":
    main()
