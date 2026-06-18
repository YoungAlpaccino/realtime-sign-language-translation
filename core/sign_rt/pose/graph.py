"""Spatial-graph construction for the skeleton GCN.

Natural skeleton edges + symmetric hand bones, plus face-contour links.
The adjacency is baked into the ONNX graph as a constant buffer, so only the
Python side builds it (the browser never reconstructs it).
"""
from __future__ import annotations

import numpy as np

from .holistic import (
    FACE_OFFSET,
    LHAND_OFFSET,
    N_FACE,
    N_KEYPOINTS,
    RHAND_OFFSET,
)

# MediaPipe hand bone topology (indices within a single 21-point hand).
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm arch
]

# Upper-body pose edges (MediaPipe pose indices) — signing is upper-body.
_POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms + shoulders
    (11, 23), (12, 24), (23, 24),                      # torso
    (15, 17), (15, 19), (15, 21),                      # left-hand stub
    (16, 18), (16, 20), (16, 22),                      # right-hand stub
    (0, 2), (0, 5), (9, 10),                           # face anchors on pose
]

# Cross-block links so hands/face are connected to the body graph.
_POSE_L_WRIST = 15
_POSE_R_WRIST = 16
_POSE_NOSE = 0


def _edge_list() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = list(_POSE_EDGES)

    for off in (LHAND_OFFSET, RHAND_OFFSET):
        edges += [(a + off, b + off) for a, b in _HAND_EDGES]

    # Stitch wrists to hand roots, nose to face root.
    edges.append((_POSE_L_WRIST, LHAND_OFFSET))
    edges.append((_POSE_R_WRIST, RHAND_OFFSET))
    edges.append((_POSE_NOSE, FACE_OFFSET))

    # Face contour ring.
    edges += [(FACE_OFFSET + i, FACE_OFFSET + (i + 1) % N_FACE) for i in range(N_FACE)]
    return edges


def build_adjacency() -> np.ndarray:
    """Return the (N, N) symmetric-normalized spatial adjacency matrix.

    Includes natural body edges, per-hand bone topology, and a face-contour ring.
    Self-loops are added, then symmetric normalization D^-1/2 (A+I) D^-1/2.
    """
    n = N_KEYPOINTS
    a = np.zeros((n, n), dtype=np.float32)
    for i, j in _edge_list():
        a[i, j] = 1.0
        a[j, i] = 1.0

    a += np.eye(n, dtype=np.float32)  # self-loops
    deg = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return (d_inv_sqrt[:, None] * a) * d_inv_sqrt[None, :]
