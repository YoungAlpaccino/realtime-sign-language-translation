// Canvas skeleton overlay. Draws normalized keypoints; privacy mode never needs
// the RGB frame, only the points.
import { LHAND_OFFSET, N_KEYPOINTS, RHAND_OFFSET } from "../core_port/normalization";

// A few illustrative bones (subset of the Python adjacency) for the overlay.
const POSE_EDGES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
];
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function edges(): [number, number][] {
  const all = [...POSE_EDGES];
  for (const off of [LHAND_OFFSET, RHAND_OFFSET]) {
    for (const [a, b] of HAND_EDGES) all.push([a + off, b + off]);
  }
  return all;
}

const EDGES = edges();

/**
 * Draw a flat (N_KEYPOINTS*3) raw (image-normalized 0..1) keypoint buffer.
 * Pass raw (pre-normalization) coords so points map to canvas pixels.
 */
export function drawSkeleton(ctx: CanvasRenderingContext2D, kpts: Float32Array): void {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(80,200,255,0.9)";
  ctx.lineWidth = 2;
  for (const [a, b] of EDGES) {
    const ax = kpts[a * 3], ay = kpts[a * 3 + 1];
    const bx = kpts[b * 3], by = kpts[b * 3 + 1];
    if (!isFinite(ax) || !isFinite(bx)) continue;
    ctx.beginPath();
    ctx.moveTo(ax * width, ay * height);
    ctx.lineTo(bx * width, by * height);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(255,180,60,0.95)";
  for (let i = 0; i < N_KEYPOINTS; i++) {
    const x = kpts[i * 3], y = kpts[i * 3 + 1];
    if (!isFinite(x) || !isFinite(y)) continue;
    ctx.beginPath();
    ctx.arc(x * width, y * height, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
}
