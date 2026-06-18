"""Edge device runner: capture -> Holistic -> ONNX recognizer -> WS publish.

Thin real-time path for the Raspberry Pi. Publishes {t, keypoints, partial_gloss}.
Keypoints-only by default (privacy).

Requires the edge extras: `pip install mediapipe opencv-python websockets`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from sign_rt.decode.streaming_ctc import StreamingCTCDecoder  # noqa: E402
from sign_rt.io.keypoint_codec import encode  # noqa: E402
from sign_rt.pose.holistic import N_KEYPOINTS, HolisticExtractor, normalize_keypoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="ws://localhost:8000/ws", help="WS endpoint")
    p.add_argument("--onnx", default="artifacts/onnx/recognizer_ctr_gcn.onnx")
    p.add_argument("--session", default="edge-1")
    p.add_argument("--token", default="", help="JWT; mint one via POST /api/token")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--window-frames", type=int, default=64)
    p.add_argument("--infer-every", type=int, default=8)
    p.add_argument("--keypoints-only", action="store_true", default=True)
    return p.parse_args()


def _pack_input(buffer: list[np.ndarray]) -> np.ndarray:
    """Stack a rolling buffer of (N_KEYPOINTS, 3) frames into (1, 3, T, V)."""
    arr = np.stack(buffer, axis=0)            # (T, V, 3)
    arr = np.transpose(arr, (2, 0, 1))        # (3, T, V)
    return arr[None].astype(np.float32)       # (1, 3, T, V)


async def run(args: argparse.Namespace) -> None:
    import cv2  # type: ignore
    import onnxruntime as ort
    import websockets

    extractor = HolisticExtractor()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    decoder = StreamingCTCDecoder()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    url = f"{args.backend}?session={args.session}&token={args.token}"
    buffer: list[np.ndarray] = []
    frame_no = 0

    async with websockets.connect(url) as ws:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kpts = normalize_keypoints(extractor(frame))
            buffer.append(kpts)
            if len(buffer) > args.window_frames:
                buffer.pop(0)

            partial: list[int] = []
            if frame_no % args.infer_every == 0 and buffer:
                (log_probs,) = session.run(["log_probs"], {"keypoints": _pack_input(buffer)})
                partial = decoder.push(log_probs[0])

            payload = encode(kpts) if args.keypoints_only else b""
            await ws.send(
                json.dumps(
                    {
                        "type": "frame",
                        "t": time.time(),
                        "keypoints": payload.hex(),
                        "partial_gloss": partial,
                    }
                )
            )
            frame_no += 1

    cap.release()


def main() -> None:
    args = parse_args()
    assert N_KEYPOINTS == 145  # layout guard shared with the recognizer
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
