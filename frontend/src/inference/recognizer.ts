// onnxruntime-web recognizer: rolling buffer of normalized keypoint frames ->
// per-frame CTC log-probs -> greedy collapsed gloss ids.
import * as ort from "onnxruntime-web";

import { ctcCollapse } from "../core_port/ctc_collapse";
import { N_KEYPOINTS } from "../core_port/normalization";

const DEFAULT_MODEL = "/api/artifacts/onnx/recognizer_ctr_gcn.onnx";

export class Recognizer {
  private session: ort.InferenceSession | null = null;
  private buffer: Float32Array[] = [];
  private readonly window: number;
  private readonly blank: number;

  constructor(window = 64, blank = 0) {
    this.window = window;
    this.blank = blank;
  }

  async load(modelUrl = DEFAULT_MODEL): Promise<void> {
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
  }

  /** Append one normalized (N_KEYPOINTS*3) frame to the rolling buffer. */
  push(frame: Float32Array): void {
    this.buffer.push(frame);
    if (this.buffer.length > this.window) this.buffer.shift();
  }

  /** Run the recognizer over the current window; returns collapsed gloss ids. */
  async infer(): Promise<number[]> {
    if (!this.session || this.buffer.length === 0) return [];
    const t = this.buffer.length;
    const v = N_KEYPOINTS;

    // Pack to (1, C=3, T, V): channel-major as the PyTorch model expects.
    const data = new Float32Array(3 * t * v);
    for (let ti = 0; ti < t; ti++) {
      const frame = this.buffer[ti];
      for (let vi = 0; vi < v; vi++) {
        for (let c = 0; c < 3; c++) {
          data[c * t * v + ti * v + vi] = frame[vi * 3 + c];
        }
      }
    }

    const input = new ort.Tensor("float32", data, [1, 3, t, v]);
    const out = await this.session.run({ keypoints: input });
    const logProbs = out.log_probs;
    const [, T, vocab] = logProbs.dims as number[];
    const arr = logProbs.data as Float32Array;

    const ids: number[] = [];
    for (let ti = 0; ti < T; ti++) {
      let best = 0;
      let bestVal = -Infinity;
      for (let k = 0; k < vocab; k++) {
        const val = arr[ti * vocab + k];
        if (val > bestVal) {
          bestVal = val;
          best = k;
        }
      }
      ids.push(best);
    }
    return ctcCollapse(ids, this.blank);
  }

  reset(): void {
    this.buffer = [];
  }
}
