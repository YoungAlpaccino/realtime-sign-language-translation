"""MediaPipe Holistic wrapper and keypoint normalization.

Keep 33 pose + 2x21 hand + a 70-point face contour subset. Normalize to a
body-centered, shoulder-width-scaled frame.

Keypoint layout (row index in the (N_KEYPOINTS, 3) array):
    [0   : 33 ) pose       (MediaPipe pose landmark indices 0..32)
    [33  : 54 ) left hand  (MediaPipe hand landmark indices 0..20)
    [54  : 75 ) right hand (MediaPipe hand landmark indices 0..20)
    [75  : 145) face       (70-point curated contour subset)

Each row is (x, y, visibility). x/y are image-normalized in [0, 1] before
`normalize_keypoints`; visibility in [0, 1]. Missing landmarks are NaN.
"""
from __future__ import annotations

import numpy as np

# Curated keypoint counts (see RESEARCH.md §6).
N_POSE = 33
N_HAND = 21
N_FACE = 70
N_KEYPOINTS = N_POSE + 2 * N_HAND + N_FACE  # 145

# Row offsets for each block (PORTED to TS — keep in sync).
POSE_OFFSET = 0
LHAND_OFFSET = N_POSE                 # 33
RHAND_OFFSET = N_POSE + N_HAND        # 54
FACE_OFFSET = N_POSE + 2 * N_HAND     # 75

# MediaPipe pose landmark indices used for the body-centered frame.
L_SHOULDER = 11
R_SHOULDER = 12

_EPS = 1e-6


class HolisticExtractor:
    """Wrap MediaPipe Holistic to produce a fixed-size keypoint array per frame.

    MediaPipe is an optional, edge-only dependency, so it is imported lazily:
    the core library, backend, and tests do not require it to be installed.
    """

    def __init__(self, model_complexity: int = 1) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:  # pragma: no cover - edge-only path
            raise ImportError(
                "mediapipe is required for HolisticExtractor; install it on the "
                "edge/capture device (`pip install mediapipe`)."
            ) from exc
        self.model_complexity = model_complexity
        self._holistic = mp.solutions.holistic.Holistic(
            model_complexity=model_complexity,
            refine_face_landmarks=True,
        )

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return (N_KEYPOINTS, 3) array of (x, y, visibility) for one frame."""
        import cv2  # type: ignore  # pragma: no cover - edge-only path

        results = self._holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        kpts = np.full((N_KEYPOINTS, 3), np.nan, dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kpts[POSE_OFFSET + i] = (lm.x, lm.y, lm.visibility)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                kpts[LHAND_OFFSET + i] = (lm.x, lm.y, 1.0)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                kpts[RHAND_OFFSET + i] = (lm.x, lm.y, 1.0)
        if results.face_landmarks:
            # Curated 70-point subset: evenly sample the dense face mesh.
            face = results.face_landmarks.landmark
            idx = np.linspace(0, len(face) - 1, N_FACE).astype(int)
            for j, fi in enumerate(idx):
                lm = face[fi]
                kpts[FACE_OFFSET + j] = (lm.x, lm.y, 1.0)
        return kpts


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Body-center and shoulder-width-scale keypoints.

    Args:
        kpts: (N_KEYPOINTS, 3) raw keypoints (x, y, visibility), possibly NaN.
    Returns:
        (N_KEYPOINTS, 3) normalized keypoints. PORTED to TS — keep in sync.

    Procedure (must match normalization.ts exactly):
      1. center = midpoint of the two shoulders (x, y).
      2. scale  = Euclidean distance between the two shoulders; if it is
         non-finite or ~0, fall back to 1.0.
      3. x,y    = (x - center) / scale; visibility passed through.
      4. any non-finite value becomes 0.0.
    """
    out = np.array(kpts, dtype=np.float32, copy=True)

    ls = out[L_SHOULDER, :2]
    rs = out[R_SHOULDER, :2]
    center = (ls + rs) / 2.0
    scale = float(np.sqrt(((ls - rs) ** 2).sum()))
    if not np.isfinite(scale) or scale < _EPS:
        scale = 1.0
    if not np.all(np.isfinite(center)):
        center = np.zeros(2, dtype=np.float32)

    out[:, 0] = (out[:, 0] - center[0]) / scale
    out[:, 1] = (out[:, 1] - center[1]) / scale
    out[~np.isfinite(out)] = 0.0
    return out
