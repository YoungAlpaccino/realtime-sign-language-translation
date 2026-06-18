"""Compact keypoint codec for the WebSocket wire format.

PORTED to TS — encode/decode must be byte-compatible (keypoint_codec.ts).

Wire format (little-endian):
    offset 0  : uint8  magic[0] = 'S' (0x53)
    offset 1  : uint8  magic[1] = 'R' (0x52)
    offset 2  : uint8  version  = 1
    offset 3  : uint8  ndim     = 3      (x, y, visibility)
    offset 4  : uint16 n_keypoints
    offset 6+ : int16  body, row-major (n_keypoints * ndim) values,
                each = round(clamp(value, -CLAMP, CLAMP) * SCALE)

Quantization resolution is 1/SCALE ≈ 2.4e-4 — lossless within tolerance for
shoulder-normalized coordinates. Privacy: payload carries keypoints only,
never pixels.
"""
from __future__ import annotations

import numpy as np

MAGIC0 = 0x53  # 'S'
MAGIC1 = 0x52  # 'R'
VERSION = 1
NDIM = 3
HEADER_BYTES = 6

SCALE = 4096.0
CLAMP = 8.0  # normalized coords + visibility comfortably fit in [-8, 8]
_INT16_MIN = -32768
_INT16_MAX = 32767


def encode(kpts: np.ndarray) -> bytes:
    """Pack a (N_KEYPOINTS, 3) float array into a compact byte payload.

    Privacy: payload carries keypoints only, never pixels.
    """
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != NDIM:
        raise ValueError(f"expected (N, {NDIM}) keypoints, got {arr.shape}")
    n = arr.shape[0]

    flat = np.nan_to_num(arr, nan=0.0, posinf=CLAMP, neginf=-CLAMP).reshape(-1)
    q = np.clip(np.rint(np.clip(flat, -CLAMP, CLAMP) * SCALE), _INT16_MIN, _INT16_MAX)
    q = q.astype("<i2")

    header = bytes([MAGIC0, MAGIC1, VERSION, NDIM]) + int(n).to_bytes(2, "little")
    return header + q.tobytes()


def decode(payload: bytes) -> np.ndarray:
    """Inverse of encode(): bytes -> (N_KEYPOINTS, 3) float array."""
    if len(payload) < HEADER_BYTES:
        raise ValueError("payload too short for header")
    if payload[0] != MAGIC0 or payload[1] != MAGIC1:
        raise ValueError("bad magic")
    if payload[2] != VERSION:
        raise ValueError(f"unsupported version {payload[2]}")
    ndim = payload[3]
    n = int.from_bytes(payload[4:6], "little")

    body = np.frombuffer(payload, dtype="<i2", offset=HEADER_BYTES, count=n * ndim)
    return (body.astype(np.float32) / SCALE).reshape(n, ndim)
