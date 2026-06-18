"""Smoke tests for sign-rt scaffold.

Real tests to add as the project is fleshed out (see RESEARCH.md §9):
  - test_onnx_parity: max logit Δ < 1e-2 across Python / onnxruntime-web on fixture clip
  - test_ctc_collapse_parity: Python ctc_collapse == TS ctc_collapse on shared fixtures
  - test_normalization_parity: Python normalize_keypoints == TS on shared fixtures
  - test_keypoint_codec_roundtrip: encode/decode is lossless within quant tolerance
  - test_wer_phoenix_dev: dev WER <= 21 with KenLM rescoring
  - test_streaming_vs_offline_wer: streaming within +2 WER of offline
  - test_isolated_wlasl_topk: top-1/top-5 within published range
  - test_slt_bleu: BLEU-4 within expected range (two-stage vs gloss-free)
  - test_edge_latency_p95: end-to-end Pi p95 <= 65 ms at >= 24 FPS
"""
from __future__ import annotations


def test_imports() -> None:
    import sign_rt

    assert sign_rt.__version__


def test_smoke() -> None:
    assert True
