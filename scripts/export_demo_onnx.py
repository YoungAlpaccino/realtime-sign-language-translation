"""Export a (random-weight) recognizer to ONNX so the three surfaces are runnable
before Phase-2 training exists.

    python scripts/export_demo_onnx.py

Writes artifacts/onnx/recognizer_<backbone>.onnx for st_gcn and ctr_gcn and
verifies Python<->ONNX parity (the Phase-0 gate). Replace with trained weights
once checkpoints land.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from sign_rt.io.onnx_export import check_parity, export_onnx  # noqa: E402
from sign_rt.seq.recognizer import Recognizer  # noqa: E402

VOCAB = 1296  # matches configs/cslr.yaml vocab_size placeholder


def main() -> None:
    torch.manual_seed(0)
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "onnx"
    for backbone in ("st_gcn", "ctr_gcn"):
        model = Recognizer(vocab_size=VOCAB, spatial_backbone=backbone)
        model.eval()
        x = Recognizer.dummy_input(t=64)
        with torch.no_grad():
            ref = model(x)
        path = export_onnx(model, x, out_dir / f"recognizer_{backbone}.onnx")
        delta = check_parity(path, x, ref)
        status = "OK" if delta < 1e-2 else "FAIL"
        print(f"{backbone:8s} -> {path.name}  parity delta={delta:.2e}  [{status}]")


if __name__ == "__main__":
    main()
