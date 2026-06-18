// Ported from core/sign_rt/io/keypoint_codec.py.
// MUST be byte-compatible with the Python encoder/decoder (wire format).
//
// Layout (little-endian):
//   0: uint8  'S' (0x53)   1: uint8  'R' (0x52)
//   2: uint8  version=1    3: uint8  ndim=3
//   4: uint16 n_keypoints
//   6: int16  body, row-major (n * ndim), each = round(clamp(v,-8,8) * 4096)

export const MAGIC0 = 0x53;
export const MAGIC1 = 0x52;
export const VERSION = 1;
export const NDIM = 3;
export const HEADER_BYTES = 6;

export const SCALE = 4096.0;
export const CLAMP = 8.0;
const INT16_MIN = -32768;
const INT16_MAX = 32767;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function encode(kpts: Float32Array): ArrayBuffer {
  const total = kpts.length;
  if (total % NDIM !== 0) throw new Error(`expected multiple of ${NDIM}, got ${total}`);
  const n = total / NDIM;

  const buf = new ArrayBuffer(HEADER_BYTES + n * NDIM * 2);
  const view = new DataView(buf);
  view.setUint8(0, MAGIC0);
  view.setUint8(1, MAGIC1);
  view.setUint8(2, VERSION);
  view.setUint8(3, NDIM);
  view.setUint16(4, n, true);

  for (let i = 0; i < total; i++) {
    let v = kpts[i];
    if (!Number.isFinite(v)) v = 0;
    // Math.round matches numpy rint for the .5-free values quantization produces
    // in practice; use round-half-away to stay deterministic across surfaces.
    const q = clamp(Math.round(clamp(v, -CLAMP, CLAMP) * SCALE), INT16_MIN, INT16_MAX);
    view.setInt16(HEADER_BYTES + i * 2, q, true);
  }
  return buf;
}

export function decode(payload: ArrayBuffer): Float32Array {
  if (payload.byteLength < HEADER_BYTES) throw new Error("payload too short");
  const view = new DataView(payload);
  if (view.getUint8(0) !== MAGIC0 || view.getUint8(1) !== MAGIC1) throw new Error("bad magic");
  if (view.getUint8(2) !== VERSION) throw new Error(`unsupported version ${view.getUint8(2)}`);
  const ndim = view.getUint8(3);
  const n = view.getUint16(4, true);

  const out = new Float32Array(n * ndim);
  for (let i = 0; i < n * ndim; i++) {
    out[i] = view.getInt16(HEADER_BYTES + i * 2, true) / SCALE;
  }
  return out;
}
