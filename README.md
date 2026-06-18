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

## Quickstart (TODO)

```bash
# TODO: python -m venv .venv && source .venv/bin/activate
# TODO: pip install -r requirements.txt
# TODO: pytest -q
# TODO: uvicorn backend.app.main:app --reload
# TODO: cd frontend && npm install && npm run dev
# TODO: python edge/run.py --backend ws://localhost:8000/ws
```

## Links

- [docs/RESEARCH.md](docs/RESEARCH.md) — full research design doc.
- [ROADMAP.md](ROADMAP.md) — phased build plan and task checklists.
