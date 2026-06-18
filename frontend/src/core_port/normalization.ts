// Ported from core/sign_rt/pose/holistic.py::normalize_keypoints.
// MUST stay numerically aligned with the Python implementation (parity test).

export const N_POSE = 33;
export const N_HAND = 21;
export const N_FACE = 70;
export const N_KEYPOINTS = N_POSE + 2 * N_HAND + N_FACE; // 145

// Row offsets for each block (keep in sync with holistic.py).
export const POSE_OFFSET = 0;
export const LHAND_OFFSET = N_POSE; // 33
export const RHAND_OFFSET = N_POSE + N_HAND; // 54
export const FACE_OFFSET = N_POSE + 2 * N_HAND; // 75

// MediaPipe pose landmark indices used for the body-centered frame.
export const L_SHOULDER = 11;
export const R_SHOULDER = 12;

const EPS = 1e-6;
const NDIM = 3;

/**
 * Body-center and shoulder-width-scale a flat (N_KEYPOINTS * 3) buffer of
 * (x, y, visibility) rows. Returns a new Float32Array of the same length.
 * Procedure must match normalize_keypoints() in holistic.py exactly.
 */
export function normalizeKeypoints(kpts: Float32Array): Float32Array {
  const out = new Float32Array(kpts); // copy

  const lx = out[L_SHOULDER * NDIM + 0];
  const ly = out[L_SHOULDER * NDIM + 1];
  const rx = out[R_SHOULDER * NDIM + 0];
  const ry = out[R_SHOULDER * NDIM + 1];

  let cx = (lx + rx) / 2;
  let cy = (ly + ry) / 2;
  let scale = Math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2);
  if (!Number.isFinite(scale) || scale < EPS) scale = 1.0;
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
    cx = 0;
    cy = 0;
  }

  const n = out.length / NDIM;
  for (let i = 0; i < n; i++) {
    const base = i * NDIM;
    out[base + 0] = (out[base + 0] - cx) / scale;
    out[base + 1] = (out[base + 1] - cy) / scale;
    // visibility (base + 2) passes through.
    for (let c = 0; c < NDIM; c++) {
      if (!Number.isFinite(out[base + c])) out[base + c] = 0;
    }
  }
  return out;
}
