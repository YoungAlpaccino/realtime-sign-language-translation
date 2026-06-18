"""Train isolated sign recognition (WLASL-2000): I3D RGB or ST-GCN pose baseline."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cslr.yaml")
    p.add_argument("--model", choices=["i3d", "stgcn"], default="stgcn")
    p.add_argument("--data-root", default="data/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build dataset/dataloader, model, optimizer, AMP loop
    # TODO: report top-1 / top-5 on official WLASL-2000 split
    raise NotImplementedError


if __name__ == "__main__":
    main()
