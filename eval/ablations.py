"""Ablation driver: augmentation, look-ahead, LM fusion, face keypoints, pose-vs-RGB."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ablation",
        choices=["aug", "lookahead", "lm", "face_kpts", "pose_vs_rgb"],
        required=True,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: sweep the chosen ablation; report deltas with seed variance
    raise NotImplementedError


if __name__ == "__main__":
    main()
