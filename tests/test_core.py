"""Core library tests: pure pieces, models, ONNX parity, streaming decode."""
from __future__ import annotations

import numpy as np
import torch

from sign_rt.decode.streaming_ctc import StreamingCTCDecoder
from sign_rt.io.keypoint_codec import decode, encode
from sign_rt.io.onnx_export import check_parity, export_onnx
from sign_rt.pose.graph import build_adjacency
from sign_rt.pose.holistic import L_SHOULDER, N_KEYPOINTS, R_SHOULDER, normalize_keypoints
from sign_rt.seq.ctc_head import ctc_collapse
from sign_rt.seq.recognizer import Recognizer


def test_ctc_collapse_merges_and_drops_blank():
    assert ctc_collapse([0, 0, 1, 1, 2, 0, 0, 2, 2], blank=0) == [1, 2, 2]
    assert ctc_collapse([], blank=0) == []
    assert ctc_collapse([0, 0, 0], blank=0) == []
    assert ctc_collapse([5, 5, 5], blank=0) == [5]


def test_normalize_scales_shoulders_to_unit():
    rng = np.random.RandomState(0)
    k = rng.rand(N_KEYPOINTS, 3).astype(np.float32)
    k[L_SHOULDER, :2] = [0.4, 0.5]
    k[R_SHOULDER, :2] = [0.6, 0.5]
    k[3] = np.nan  # missing landmark
    nk = normalize_keypoints(k)
    assert np.isfinite(nk).all()
    d = float(np.hypot(*(nk[L_SHOULDER, :2] - nk[R_SHOULDER, :2])))
    assert abs(d - 1.0) < 1e-5
    assert np.allclose(nk[3], 0.0)  # fully-NaN landmark row is zeroed


def test_codec_roundtrip_within_tolerance():
    rng = np.random.RandomState(1)
    k = (rng.rand(N_KEYPOINTS, 3).astype(np.float32) - 0.5) * 4
    back = decode(encode(k))
    assert back.shape == k.shape
    assert np.abs(back - k).max() < 1e-3


def test_adjacency_symmetric_normalized():
    a = build_adjacency()
    assert a.shape == (N_KEYPOINTS, N_KEYPOINTS)
    assert np.allclose(a, a.T)
    assert np.isfinite(a).all()
    assert (a >= 0).all()


def test_streaming_decoder_dedups_across_chunks():
    dec = StreamingCTCDecoder(blank=0)
    vocab = 4
    # chunk 1 argmax -> [1,1,2]; chunk 2 -> [2,0,3]
    lp1 = np.full((3, vocab), -9.0)
    lp1[0, 1] = lp1[1, 1] = lp1[2, 2] = 0.0
    lp2 = np.full((3, vocab), -9.0)
    lp2[0, 2] = lp2[1, 0] = lp2[2, 3] = 0.0
    dec.push(lp1)
    out = dec.push(lp2)
    assert out == [1, 2, 3]  # straddling repeat of 2 collapsed


def test_recognizer_forward_and_onnx_parity(tmp_path):
    torch.manual_seed(0)
    model = Recognizer(vocab_size=50, d_model=64, spatial_backbone="ctr_gcn", conformer_layers=2)
    model.eval()
    x = Recognizer.dummy_input(t=24)
    with torch.no_grad():
        ref = model(x)
    assert ref.shape == (1, 24, 50)
    # rows are log-probs -> exp sums to 1
    assert torch.allclose(ref.exp().sum(-1), torch.ones(1, 24), atol=1e-4)

    onnx_path = export_onnx(model, x, tmp_path / "rec.onnx")
    delta = check_parity(onnx_path, x, ref)
    assert delta < 1e-2, f"parity delta {delta}"

    # dynamic time axis: a different T must still match
    x2 = Recognizer.dummy_input(t=40)
    with torch.no_grad():
        ref2 = model(x2)
    assert check_parity(onnx_path, x2, ref2) < 1e-2
