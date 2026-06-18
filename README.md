# sign-rt — Real-Time Continuous Sign-Language Recognition & Translation

> Pose-first, edge-capable pipeline that turns a live signing video stream into gloss and natural-language text in real time — isolated → continuous → translation, on a Pi or in the browser.

## Overview

Real signing is continuous: signs co-articulate, fingerspelling bursts in, and there is no whitespace between "words". `sign-rt` tackles continuous sign-language recognition (CSLR) and sign-language translation (SLT) under a real-time, on-device budget. The bet is **pose-first**: extract body/hand/face keypoints once (MediaPipe Holistic) and run a lightweight spatial-temporal graph model on the skeleton — fast, private, and robust. One PyTorch-trained model is exported to ONNX and served on three surfaces: a Raspberry Pi edge runner, a FastAPI backend, and a browser demo via `onnxruntime-web`.

## Architecture

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

## Quickstart

```bash
# 1. Python env + deps (mediapipe is optional, edge-only).
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the test suite (core pieces, models, ONNX parity, backend, Python<->TS parity).
pytest                                                # 14 tests, ~12s

# 3. Export the (random-weight) recognizer ONNX so all three surfaces are runnable.
python scripts/export_demo_onnx.py                    # -> artifacts/onnx/recognizer_*.onnx

# 4. Backend (serves ONNX + LM artifacts, WS hub, JWT, transcripts).
uvicorn backend.app.main:app --reload --app-dir .

# 5. Browser demo (webcam -> MediaPipe -> onnxruntime-web -> skeleton + caption).
cd frontend && npm install && npm run dev             # http://localhost:5173

# 6. Edge runner (Raspberry Pi / any camera) — requires `pip install mediapipe`.
python edge/run.py --backend ws://localhost:8000/ws
```

### Status

Target **A (runnable end-to-end demo)** is implemented: the build/test infra,
the portable core (pose normalization, CTC collapse, keypoint codec, spatial
graph), the ST-GCN / CTR-GCN + Conformer + CTC recognizer, ONNX export with
Python↔TS↔ONNX parity, the FastAPI WS hub + JWT + SQLite transcripts, and the
browser webcam demo all run today. **The served recognizer uses random weights**
— gloss/caption are structurally live but not yet meaningful. Producing real
WER/BLEU numbers is Target B (training on PHOENIX-2014T / WLASL etc.); see
[ROADMAP.md](ROADMAP.md) Phases 1–6.

## Links

- [docs/RESEARCH.md](docs/RESEARCH.md) — full research design doc.
- [ROADMAP.md](ROADMAP.md) — phased build plan and task checklists.
