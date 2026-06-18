"""Per-surface / per-stage latency benchmark (p50/p95) and sustained FPS."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", choices=["pi", "server", "browser"], default="pi")
    p.add_argument("--minutes", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: time capture/pose/recognizer/SLT stages; report p50/p95 + FPS
    raise NotImplementedError


if __name__ == "__main__":
    main()
