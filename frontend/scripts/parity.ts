// Cross-language parity check: runs the TS core_port implementations against a
// fixture produced by the Python side (scripts/gen_parity_fixture.py) and exits
// non-zero on any mismatch. Run with: node --experimental-strip-types parity.ts
//
// Invoked from the pytest test `tests/test_parity.py::test_cross_language_parity`.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { ctcCollapse } from "../src/core_port/ctc_collapse.ts";
import { normalizeKeypoints } from "../src/core_port/normalization.ts";
import { decode as codecDecode } from "../src/core_port/keypoint_codec.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../src/core_port/__fixtures__/parity.json");

type Fixture = {
  collapse: { input: number[]; blank: number; expected: number[] }[];
  normalize: { input: number[]; expected: number[] }[];
  codec: { payload_hex: string; expected: number[] }[];
};

const fx = JSON.parse(readFileSync(fixturePath, "utf8")) as Fixture;

let failures = 0;
const fail = (msg: string) => {
  console.error("MISMATCH:", msg);
  failures++;
};

function arrEq(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
function arrClose(a: ArrayLike<number>, b: number[], tol: number): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < b.length; i++) if (Math.abs(a[i] - b[i]) > tol) return false;
  return true;
}

for (const c of fx.collapse) {
  const got = ctcCollapse(c.input, c.blank);
  if (!arrEq(got, c.expected)) fail(`collapse ${JSON.stringify(c.input)} -> ${JSON.stringify(got)}`);
}

for (const c of fx.normalize) {
  const got = normalizeKeypoints(new Float32Array(c.input));
  if (!arrClose(got, c.expected, 1e-4)) fail(`normalize row mismatch`);
}

for (const c of fx.codec) {
  const bytes = Uint8Array.from(c.payload_hex.match(/.{2}/g)!.map((h) => parseInt(h, 16)));
  const got = codecDecode(bytes.buffer);
  if (!arrClose(got, c.expected, 1e-3)) fail(`codec decode mismatch`);
}

if (failures > 0) {
  console.error(`PARITY FAILED: ${failures} mismatch(es)`);
  process.exit(1);
}
console.log("PARITY OK: all cases match");
