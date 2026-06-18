// Ported from core/sign_rt/seq/ctc_head.py::ctc_collapse.
// MUST match Python logic exactly (shared fixture parity test).
//
// Step 1: collapse runs of identical labels to a single label.
// Step 2: drop the blank symbol.

export function ctcCollapse(ids: number[], blank = 0): number[] {
  const out: number[] = [];
  let prev: number | null = null;
  for (const i of ids) {
    if (i === prev) continue;
    prev = i;
    if (i !== blank) out.push(i);
  }
  return out;
}
