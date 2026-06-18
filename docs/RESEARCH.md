# 07 — Real-Time Continuous Sign-Language Recognition & Translation (on-device)

> Pose-first, edge-capable pipeline that turns a live signing video stream into gloss and natural-language text in real time — isolated → continuous → translation, on a Pi or in the browser.

## 1. Why this is cool

Most sign-language demos recognize *isolated* signs (one word at a time, segmented for you). Real signing is **continuous**: signs co-articulate, fingerspelling bursts in, and there is no whitespace between "words". This project tackles the hard version — **continuous sign-language recognition (CSLR)** and **sign-language translation (SLT)** — and does it under a real-time, on-device budget.

The key bet is **pose-first**. Instead of streaming RGB to a heavy 3D-CNN, we extract body/hand/face keypoints once (MediaPipe Holistic) and run a lightweight spatial-temporal graph model on the skeleton. That buys three things at once:

- **Speed**: keypoints are ~300 floats/frame, so the recognition head is tiny and runs on a Raspberry Pi.
- **Privacy**: the WebSocket can publish keypoints, not faces — a real concern for a deaf-user-facing tool.
- **Robustness**: pose normalization absorbs clothing, lighting, and background variation that wreck RGB models.

It also reuses the keypoint stack from project 06 (pose/hand estimation), so the same `core/` library serves the edge runner, the FastAPI backend, and an `onnxruntime-web` browser demo — train once in PyTorch, export ONNX, serve everywhere.

## 2. Research novelty & contributions

- **Streaming CTC decoder with bounded latency.** We propose a chunked, look-ahead-limited ST-GCN→Conformer→CTC recognizer that emits gloss with a fixed ≤320 ms look-ahead, and show it recovers ≥95% of the WER of the full-sequence offline model on PHOENIX-2014T while running causally. Target: offline dev WER ≈ 19–20, streaming dev WER ≤ 21.
- **Pose-vs-RGB honesty study at fixed compute.** A controlled comparison of pose-based (CTR-GCN) vs RGB (I3D/S3D) CSLR at matched FLOPs and matched latency, reporting the accuracy-per-millisecond Pareto frontier rather than only best-case accuracy. Expected: pose within 2–3 WER of RGB at ~8× lower latency.
- **Gloss-free vs gloss-supervised translation.** We contribute a side-by-side of (a) a two-stage CSLR→gloss→text pipeline and (b) a gloss-free Sign2Text Transformer on the same splits, quantifying the BLEU-4 cost of dropping gloss annotations (typically 2–4 BLEU on PHOENIX-2014T).
- **Keypoint augmentation suite for low-resource generalization.** A reproducible set of skeleton augmentations (bone-length jitter, temporal warp, hand-dropout, mirroring with handedness relabeling) that improves WLASL-2000 top-1 by ≥3 points and reduces signer-dependent overfitting.
- **Three-surface parity benchmark.** Identical ONNX weights evaluated on Pi-class edge, FastAPI server, and browser WASM/WebGPU, reporting numerical agreement (max logit Δ) and per-surface latency — a reproducibility artifact most CSLR papers omit.

## 3. System architecture

```
                          ┌──────────────────────────────────────────────────┐
                          │                  core/ (Python)                   │
                          │  pose: MediaPipe Holistic wrapper + normalization │
                          │  graph: ST-GCN / CTR-GCN spatial-temporal blocks  │
                          │  seq:   Conformer encoder + CTC head (recognizer) │
                          │  slt:   seq2seq Transformer (gloss→text / S2T)    │
                          │  decode: streaming CTC beam, KenLM rescoring      │
                          │  io:    ONNX export/load, keypoint codec          │
                          └───────────────┬───────────────────┬──────────────┘
                                          │ imported by        │ exported as ONNX
              ┌───────────────────────────┘                   └───────────────────────────┐
              ▼                                                                             ▼
  ┌───────────────────────────┐        ┌──────────────────────────────┐      ┌──────────────────────────────┐
  │   edge/ (Raspberry Pi)    │        │      backend/ (FastAPI)       │      │   frontend/ (React 19 + TS)  │
  │  camera capture (OpenCV)  │        │  WS hub (fan-in/fan-out)      │      │  webcam → MediaPipe / wasm    │
  │  Holistic keypoints       │  WS    │  JWT auth, sessions           │  WS  │  onnxruntime-web inference    │
  │  ONNX Runtime (CTC)       │ ─────► │  SQLite/SQLModel transcripts  │ ◄──► │  canvas: skeleton + caption   │
  │  publish gloss + kpts     │ kpts/  │  SLT translate endpoint       │ text │  gloss timeline, A/B viewer   │
  │                           │ gloss  │  serves ONNX + LM artifacts   │      │  privacy: kpts-only mode      │
  └───────────────────────────┘        └──────────────────────────────┘      └──────────────────────────────┘
```

**Prose.** The `core/` Python package is the single source of truth: pose normalization, the graph backbone, the Conformer+CTC recognizer, the translation decoder, and the streaming beam search all live there and are unit-tested in isolation. The **edge** runner is the thin real-time path — capture a frame, run Holistic, normalize, feed the rolling buffer to the ONNX recognizer, and publish a compact message `{t, keypoints, partial_gloss}` over WebSocket. The **backend** is a WS hub plus persistence: it authenticates clients (JWT), fans keypoint/gloss streams to subscribers, runs the heavier gloss→text translation on demand, and stores transcripts in SQLite via SQLModel. The **frontend** can either consume the edge stream (low-power devices) or do everything locally in the browser via `onnxruntime-web`, drawing the skeleton and a live caption on a canvas overlay. Critical, latency-sensitive logic (keypoint normalization, the CTC collapse/merge rule, beam pruning) is **ported Python↔TS** and covered by a shared fixture test so the two implementations stay numerically aligned.

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Training | PyTorch | ST-GCN/CTR-GCN, Conformer, seq2seq Transformer; AMP |
| Pose extraction | MediaPipe Holistic | body 33 + 2×21 hands + 468 face → curated subset |
| Core CV/array | OpenCV, NumPy | capture, geometry, keypoint codec |
| Inference | ONNX Runtime (CPU/CUDA), onnxruntime-web | one graph, three surfaces |
| Language model | KenLM (n-gram) | shallow-fusion rescoring of CTC beams |
| Backend | FastAPI, SQLModel, SQLite | WS hub, JWT auth, transcript store |
| Frontend | React 19 + TypeScript + Vite | canvas overlay, A/B viewer |
| Browser inference | onnxruntime-web (WASM + WebGPU) | client-side recognizer |
| Edge | Raspberry Pi 5 / device runner | capture → infer → WS publish |
| Eval/tooling | pytest, jiwer, sacrebleu | WER, BLEU-4/chrF, fixtures |

## 5. Datasets

| Dataset | Task | One-line note |
|---|---|---|
| WLASL (2000) | Isolated (ASL) | 2000-gloss word-level benchmark; signer-dependent splits a known pitfall. |
| MS-ASL | Isolated (ASL) | 1000 classes, in-the-wild YouTube clips; tests robustness. |
| RWTH-PHOENIX-Weather 2014T | Continuous + translation (DGS) | The CSLR/SLT standard; gloss + German text, weather domain. |
| CSL-Daily | Continuous + translation (CSL) | Large daily-life Chinese SL corpus; sentence-level glosses + text. |
| How2Sign | Continuous + translation (ASL) | Large multimodal instructional ASL; sentence-level translation. |
| MS-ASL / WLASL pose caches | Pretraining | Holistic keypoints precomputed for fast graph-model pretraining. |

Note on licensing/ethics: all corpora are research-licensed; PHOENIX and CSL-Daily involve identifiable signers, so face keypoints are used numerically (never re-rendered) and a keypoints-only publish mode is the default for any live capture.

## 6. Models & algorithms

**Pose front-end.** MediaPipe Holistic → keep 33 pose + 2×21 hand + a 70-point face contour subset (eyes/mouth matter for grammatical markers). Normalize to a body-centered, shoulder-width-scaled frame; build the spatial graph (natural skeleton edges + symmetric hand bones).

**Isolated recognition (baseline + proposed).**
- *Baseline:* I3D / S3D on RGB (the WLASL reference).
- *Proposed:* CTR-GCN (channel-wise topology refinement GCN) on keypoints + temporal attention pooling.

**Continuous recognition (CSLR).**
- *Backbone:* ST-GCN/CTR-GCN spatial blocks → temporal Conformer encoder.
- *Loss/decoder:* CTC with the standard blank-collapse; **streaming chunked inference** with a bounded right-context Conformer (cached attention) for causal operation; beam search + KenLM shallow fusion.
- *Baselines:* VAC / SMKD-style CTC CNN-BiLSTM; CorrNet RGB.

**Translation (SLT).**
- *Two-stage:* CSLR gloss → seq2seq Transformer (gloss→text), mirrors Sign-Language-Transformers.
- *Gloss-free:* Sign2Text encoder-decoder directly from keypoint features (no gloss supervision).

**Algorithmic specifics that matter:** rolling-buffer windowing with overlap-add for the streaming encoder; CTC peak alignment to time-stamp glosses for the UI timeline; mirroring augmentation with explicit handedness relabeling so left/right-dominant signers are not corrupted.

## 7. Real-time / edge budget

| Surface | Stage | p50 | p95 | Notes |
|---|---|---|---|---|
| Pi 5 (CPU) | Holistic keypoints | 28 ms | 42 ms | dominant cost; 640×480 |
| Pi 5 (CPU) | CTR-GCN+Conformer CTC | 9 ms | 16 ms | 64-frame window, INT8 |
| Pi 5 end-to-end | capture→partial gloss | ~40 ms | ~62 ms | ≈ 24–30 FPS sustained |
| Server (CUDA) | recognizer | 3 ms | 6 ms | batched WS clients |
| Server | gloss→text (SLT) | 35 ms | 70 ms | per sentence, on segment close |
| Browser (WebGPU) | recognizer | 11 ms | 22 ms | onnxruntime-web; WASM ~2.5× slower |

Model size: recognizer ≈ 5.8 M params, ≈ 6 MB ONNX (FP16) / ≈ 3 MB INT8; SLT Transformer ≈ 24 M params, ≈ 48 MB. Streaming look-ahead capped at 320 ms (≈ 8 frames @ 25 FPS). Hardware: Raspberry Pi 5 (8 GB) for edge, single RTX-class GPU for training/server, commodity laptop GPU for browser WebGPU.

## 8. Build process

**Phase 0 — Scaffold.** Monorepo with `core/`, `backend/`, `frontend/`, `edge/`; ONNX export/load round-trip test; keypoint codec; CI running pytest + tsc. *Done when* a dummy graph exports to ONNX and produces identical logits in Python and onnxruntime-web on a fixture clip.

**Phase 1 — Baseline.** Reproduce isolated WLASL-2000 with I3D (RGB) and a vanilla ST-GCN (pose). *Done when* both train end-to-end and report top-1/top-5 within published range on the official split.

**Phase 2 — Core model.** CTR-GCN + Conformer + CTC for CSLR on PHOENIX-2014T; full-sequence (offline) decoding. *Done when* dev WER ≤ 21 with KenLM rescoring and CTC peaks time-align to ground-truth gloss order.

**Phase 3 — Real-time path.** Convert the Conformer to cached/chunked attention; streaming beam search; quantize to INT8; deploy on Pi. *Done when* end-to-end Pi latency p95 ≤ 65 ms at ≥ 24 FPS and streaming dev WER within +2 of offline.

**Phase 4 — Three surfaces.** FastAPI WS hub + JWT + SQLite transcripts; React canvas overlay with live caption and A/B (pose-vs-RGB) viewer; browser-local inference. *Done when* the same ONNX runs on edge, server, and browser with max logit Δ < 1e-2 and the UI shows live gloss + translated text.

**Phase 5 — Rigorous eval.** Add SLT (two-stage + gloss-free); ablations (augmentation, look-ahead, LM); signer-independent splits; latency tables per surface. *Done when* the evaluation protocol (§9) is fully reproduced from one command and ablation deltas are reported with seeds.

**Phase 6 — Write-up.** Pareto figures, qualitative caption examples (including failure cases), reproducibility artifact (weights + ONNX + fixtures). *Done when* the paper draft, repo, and demo video are complete and numbers regenerate from scripts.

## 9. Evaluation protocol & metrics

- **Isolated (WLASL-2000, MS-ASL):** top-1 / top-5 accuracy on the official splits; report both signer-dependent and signer-independent where defined.
- **Continuous (PHOENIX-2014T, CSL-Daily):** **WER** (jiwer) on official dev/test, plus deletion/insertion/substitution breakdown; report offline vs streaming WER.
- **Translation (PHOENIX-2014T, CSL-Daily, How2Sign):** **BLEU-1..4** (sacrebleu), **ROUGE-L**, **chrF**; two-stage vs gloss-free.
- **Latency:** p50/p95 per surface and per stage (capture, pose, recognizer, SLT), FPS sustained over 5-minute streams.
- **Ablations:** (a) augmentation suite on/off; (b) look-ahead 0/160/320/∞ ms vs WER; (c) KenLM shallow fusion on/off; (d) face keypoints in/out; (e) pose vs RGB at matched FLOPs.
- **Baselines:** I3D/S3D (isolated), VAC/SMKD/CorrNet (CSLR), Sign-Language-Transformers (SLT).
- **Reproducibility:** fixed seeds (×3), official splits only, no test-set tuning; numerical parity check across the three surfaces.

## 10. Suggested repo structure

```
sign-rt/
├── core/
│   ├── pose/            # holistic wrapper, normalization, graph build
│   ├── graph/           # st_gcn.py, ctr_gcn.py
│   ├── seq/             # conformer.py, ctc_head.py
│   ├── slt/             # transformer_s2t.py, gloss2text.py
│   ├── decode/          # streaming_ctc.py, kenlm_rescore.py
│   ├── io/              # onnx_export.py, keypoint_codec.py
│   └── tests/           # fixtures, parity tests
├── backend/
│   ├── app/             # main.py, ws_hub.py, auth.py, models.py (SQLModel)
│   └── tests/
├── frontend/
│   ├── src/             # App.tsx, inference/ (onnxruntime-web), overlay/
│   └── src/core_port/   # normalization.ts, ctc_collapse.ts (ported)
├── edge/
│   └── runner.py        # capture → holistic → onnx → ws publish
├── train/               # train_isolated.py, train_cslr.py, train_slt.py
├── eval/                # wer.py, bleu.py, latency_bench.py, ablations.py
├── artifacts/           # onnx/, kenlm/, checkpoints/
└── README.md
```

## 11. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Holistic latency dominates Pi budget | Misses FPS target | Downscale capture, run pose every frame but recognizer on rolling buffer; consider lighter hand-only model |
| Signer-dependent leakage inflates accuracy | Invalid claims | Always report signer-independent splits; fixed official protocol |
| Gloss annotations scarce / noisy | Limits CSLR quality | Offer gloss-free SLT path; treat gloss as auxiliary, not required |
| CTC streaming hurts WER | Real-time unusable | Bounded look-ahead ablation; cached-attention Conformer; LM rescoring |
| Python↔TS drift in decode logic | Surfaces disagree | Shared fixtures + parity test in CI (max logit Δ gate) |
| Dialect / regional sign variation | Poor generalization | Per-corpus reporting; document scope; avoid cross-language claims |
| Privacy of identifiable signers | Ethical/legal | Keypoints-only publish default; never re-render faces; opt-in RGB |

## 12. Stretch goals

- **Fingerspelling sub-decoder** that switches to a character-level model during detected spelling bursts.
- **WebGPU INT8** browser path for laptop-class real-time without a server.
- **Speech output** (TTS) of the translated text for hearing interlocutors — full two-way loop.
- **Continual adaptation:** few-shot personalization to a single signer's idiolect from a 2-minute calibration.
- **Multi-language gloss** shared encoder across ASL/DGS/CSL with language-conditioned decoder.

## 13. Publication plan

- **Top venue:** IEEE Transactions on Multimedia — full system + streaming CSLR + three-surface reproducibility.
- **Mid venue / workshop:** CVPR/ICCV sign-language or assistive-tech workshop for the streaming-latency Pareto study.
- **Conference → journal path:** workshop paper on streaming CTC with bounded latency → extended journal version (Elsevier *Pattern Recognition* or *CVIU*) adding SLT, gloss-free comparison, and full ablations.
- **Key figures:** (1) architecture + three-surface diagram; (2) accuracy-vs-latency Pareto (pose vs RGB); (3) look-ahead vs WER curve; (4) qualitative caption strip with CTC time alignment, including a failure case; (5) per-surface latency/parity table.
