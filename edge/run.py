"""Edge device runner: capture -> Holistic -> ONNX recognizer -> WS publish.

Thin real-time path for the Raspberry Pi. Publishes {t, keypoints, partial_gloss}.
Keypoints-only by default (privacy).
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="ws://localhost:8000/ws", help="WS endpoint")
    p.add_argument("--onnx", default="artifacts/onnx/recognizer.onnx")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--window-frames", type=int, default=64)
    p.add_argument("--keypoints-only", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: open OpenCV capture
    # TODO: init HolisticExtractor + onnxruntime InferenceSession
    # TODO: loop: read frame -> holistic -> normalize -> rolling buffer
    #            -> streaming CTC -> encode keypoints -> WS publish
    raise NotImplementedError


if __name__ == "__main__":
    main()
