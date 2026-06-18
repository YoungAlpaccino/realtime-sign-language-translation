# Roadmap — sign-rt

Phased build process from the research doc (§8). Each phase lists tasks, deliverables, and a "done when" gate.

## Phase 0 — Scaffold

- [ ] Set up monorepo with `core/`, `backend/`, `frontend/`, `edge/`
- [ ] Implement keypoint codec (encode/decode compact keypoint frames)
- [ ] Implement ONNX export/load round-trip for a dummy graph
- [ ] Wire CI running `pytest` + `tsc`
- [ ] Add a fixture clip and a three-surface parity test stub

**Deliverable:** working skeleton repo + CI.
**Done when:** a dummy graph exports to ONNX and produces identical logits in Python and onnxruntime-web on a fixture clip.

## Phase 1 — Baseline

- [ ] Implement I3D / S3D RGB pipeline for isolated recognition
- [ ] Implement vanilla ST-GCN pose pipeline for isolated recognition
- [ ] Train both end-to-end on WLASL-2000 official split
- [ ] Report top-1 / top-5 accuracy

**Deliverable:** two reproduced isolated baselines.
**Done when:** both train end-to-end and report top-1/top-5 within published range on the official split.

## Phase 2 — Core model

- [ ] Build CTR-GCN spatial blocks + Conformer temporal encoder
- [ ] Add CTC head with blank-collapse decoding
- [ ] Integrate KenLM shallow-fusion rescoring of CTC beams
- [ ] Train CSLR on PHOENIX-2014T (full-sequence / offline)
- [ ] Verify CTC peak time-alignment to gloss order

**Deliverable:** offline CSLR recognizer.
**Done when:** dev WER ≤ 21 with KenLM rescoring and CTC peaks time-align to ground-truth gloss order.

## Phase 3 — Real-time path

- [ ] Convert Conformer to cached/chunked attention (bounded right-context)
- [ ] Implement streaming CTC beam search with overlap-add windowing
- [ ] Quantize recognizer to INT8
- [ ] Deploy on Raspberry Pi 5 and benchmark

**Deliverable:** streaming on-device recognizer.
**Done when:** end-to-end Pi latency p95 ≤ 65 ms at ≥ 24 FPS and streaming dev WER within +2 of offline.

## Phase 4 — Three surfaces

- [ ] FastAPI WS hub + JWT auth + SQLite/SQLModel transcript store
- [ ] React canvas overlay: live caption + skeleton + A/B (pose-vs-RGB) viewer
- [ ] Browser-local inference via onnxruntime-web
- [ ] Verify numerical parity across edge / server / browser

**Deliverable:** three running surfaces.
**Done when:** the same ONNX runs on edge, server, and browser with max logit Δ < 1e-2 and the UI shows live gloss + translated text.

## Phase 5 — Rigorous eval

- [ ] Add SLT: two-stage (gloss→text) + gloss-free (Sign2Text)
- [ ] Ablations: augmentation, look-ahead (0/160/320/∞ ms), LM, face keypoints, pose-vs-RGB
- [ ] Signer-independent splits
- [ ] Per-surface latency tables

**Deliverable:** full evaluation suite.
**Done when:** the evaluation protocol (§9) is fully reproduced from one command and ablation deltas are reported with seeds.

## Phase 6 — Write-up

- [ ] Generate Pareto figures (accuracy-vs-latency)
- [ ] Collect qualitative caption examples incl. failure cases
- [ ] Package reproducibility artifact (weights + ONNX + fixtures)
- [ ] Record demo video

**Deliverable:** paper draft + reproducible repo + demo.
**Done when:** the paper draft, repo, and demo video are complete and numbers regenerate from scripts.
