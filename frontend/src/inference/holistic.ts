// Browser MediaPipe Holistic wrapper.
// Produces the SAME (N_KEYPOINTS, 3) layout as core/sign_rt/pose/holistic.py so
// the ported normalization + the ONNX recognizer see identical inputs.
import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
} from "@mediapipe/tasks-vision";

import {
  FACE_OFFSET,
  LHAND_OFFSET,
  N_FACE,
  N_KEYPOINTS,
  POSE_OFFSET,
  RHAND_OFFSET,
} from "../core_port/normalization";

const WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

type LM = { x: number; y: number; visibility?: number };

export async function createHolistic(): Promise<HolisticLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_CDN);
  return HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
  });
}

function copyBlock(dst: Float32Array, landmarks: LM[] | undefined, offset: number, count: number) {
  if (!landmarks) return; // leave as NaN -> zeroed by normalization
  const n = Math.min(landmarks.length, count);
  for (let i = 0; i < n; i++) {
    const lm = landmarks[i];
    const base = (offset + i) * 3;
    dst[base] = lm.x;
    dst[base + 1] = lm.y;
    dst[base + 2] = lm.visibility ?? 1.0;
  }
}

/** Pack a HolisticLandmarker result into a flat (N_KEYPOINTS * 3) buffer. */
export function packResult(result: HolisticLandmarkerResult): Float32Array {
  const out = new Float32Array(N_KEYPOINTS * 3).fill(NaN);
  const r = result as unknown as {
    poseLandmarks?: LM[];
    leftHandLandmarks?: LM[];
    rightHandLandmarks?: LM[];
    faceLandmarks?: LM[];
  };

  copyBlock(out, r.poseLandmarks, POSE_OFFSET, 33);
  copyBlock(out, r.leftHandLandmarks, LHAND_OFFSET, 21);
  copyBlock(out, r.rightHandLandmarks, RHAND_OFFSET, 21);

  // Evenly sample the dense face mesh down to N_FACE points (matches Python).
  const face = r.faceLandmarks;
  if (face && face.length > 0) {
    for (let j = 0; j < N_FACE; j++) {
      const fi = Math.round((j * (face.length - 1)) / (N_FACE - 1));
      const lm = face[fi];
      const base = (FACE_OFFSET + j) * 3;
      out[base] = lm.x;
      out[base + 1] = lm.y;
      out[base + 2] = 1.0;
    }
  }
  return out;
}
